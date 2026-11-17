"""
V2 forecast: after each scrape, predict every active (edge, direction) for each trained
horizon and store it in `forecast` (made_at = origin slot, target_ts = origin + h*30 min).
Units not modelled (always free in training) are not written — readers treat them as free.
Keeps FORECAST_KEEP_DAYS of history (for later evaluation of live forecasts).

    venv/bin/python scripts/v2/predict_v2.py            # origin = latest scraped slot
"""
from __future__ import annotations

import json
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb

from common import ROOT
from db import connect
from features_v2 import FREE, SLOT_HIST_DAYS, _slot_hist, build_rows, extend, load_matrix

MODELS = ROOT / "models_v2"
HISTORY_DAYS = 8          # week_ago and slot_hist_mean (SLOT_HIST_DAYS) need 7 days + horizon
FORECAST_KEEP_DAYS = 14


def main():
    t0 = time.time()
    if not (MODELS / "meta.json").exists():
        raise SystemExit("no models_v2/meta.json — train first (train_v2.py)")
    meta = json.load(open(MODELS / "meta.json"))
    if not meta["horizons"]:
        raise SystemExit("models_v2/meta.json has no trained horizons — keep collecting, retrain later")
    assert SLOT_HIST_DAYS + 1 <= HISTORY_DAYS, "HISTORY_DAYS too short for slot_hist_mean"
    with connect() as conn:
        hi = conn.execute("SELECT max(ts) FROM traffic_obs").fetchone()[0]
        panel = load_matrix(conn, hi - timedelta(days=HISTORY_DAYS), hi)
    act = pd.read_csv(MODELS / "active_units.csv")
    act = act[act.active]
    mask = pd.MultiIndex.from_frame(panel.units[["edge_id", "dir"]]).isin(
        pd.MultiIndex.from_frame(act[["edge_id", "dir"]]))
    t = panel.S.shape[1] - 1                     # origin = latest observed slot
    # empty future slots so the TARGET slot exists in the matrix: features (slot_hist_mean,
    # slot_hist_n) are then computed exactly as in training instead of coming out empty
    panel = extend(panel, max(meta["horizons"]))
    hist = _slot_hist(panel.S, panel.slots)

    rows = []
    for h in meta["horizons"]:
        X = build_rows(panel, h, [t], with_target=False, hist=hist, unit_mask=mask)
        if not len(X):
            continue
        reg = xgb.XGBRegressor()
        reg.load_model(MODELS / f"xgb_reg_h{h}.json")
        score = np.clip(reg.predict(X[meta["features"]]), FREE, 9.5)
        p_cong = np.full(len(X), np.nan)
        if str(h) in meta.get("thresholds", {}) and (MODELS / f"xgb_cls_h{h}.json").exists():
            cls = xgb.XGBClassifier()
            cls.load_model(MODELS / f"xgb_cls_h{h}.json")
            p_cong = cls.predict_proba(X[meta["features"]])[:, 1]
        u = panel.units.iloc[X["unit"].values]
        rows.append(pd.DataFrame({"made_at": hi, "target_ts": hi + timedelta(minutes=30 * h),
                                  "edge_id": u["edge_id"].values, "dir": u["dir"].values,
                                  "score": score.round(3), "p_congested": np.round(p_cong, 4)}))
    if not rows:
        print("nothing to forecast")
        return
    df = pd.concat(rows, ignore_index=True)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM forecast WHERE made_at = %s", (hi,))
        cur.execute("DELETE FROM forecast WHERE made_at < %s", (hi - timedelta(days=FORECAST_KEEP_DAYS),))
        with cur.copy("COPY forecast (made_at, target_ts, edge_id, dir, score, p_congested) FROM STDIN") as cp:
            for r in df.itertuples(index=False):
                cp.write_row((r.made_at, r.target_ts, int(r.edge_id), int(r.dir), float(r.score),
                              None if np.isnan(r.p_congested) else float(r.p_congested)))
    print(f"forecast from {hi:%Y-%m-%d %H:%M} UTC: {len(df):,} rows, horizons {meta['horizons']}, "
          f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
