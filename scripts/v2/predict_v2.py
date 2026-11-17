"""
V2 forecast: after each scrape, predict every active (edge, direction) for each trained
horizon and store it in `forecast` (made_at = origin slot, target_ts = origin + h*30 min).
Units not modelled (always free in training) are not written — readers treat them as free.
Keeps FORECAST_KEEP_DAYS of history (for later evaluation of live forecasts).
Per horizon the model is served only if it beat the naive "as now" forecast on the test
period of its last training (report.json); otherwise the naive forecast is written.
What was served goes to models_v2/serving.json (and /api/report).

    venv/bin/python scripts/v2/predict_v2.py            # origin = latest scraped slot
"""
from __future__ import annotations

import json
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb

from common import ROOT
from db import connect
from features_v2 import FREE, SLOT_HIST_DAYS, _slot_hist, build_rows, extend, load_matrix, naive_forecast

MODELS = ROOT / "models_v2"
HISTORY_DAYS = 8          # week_ago and slot_hist_mean (SLOT_HIST_DAYS) need 7 days + horizon
FORECAST_KEEP_DAYS = 14


def choose(report: dict | None, h: int) -> tuple[bool, bool]:
    """(serve regressor, serve classifier) for horizon h: only where the model beat the naive
    "as now" forecast on the held-out test period of its last training. Early on (< ~1 day of
    data) the model cannot know the daily pattern and loses; serving it would make the map
    worse than doing nothing. After a retrain where it wins, it is served automatically."""
    r = ((report or {}).get("horizons") or {}).get(str(h)) or {}
    reg, cls = r.get("regression") or {}, r.get("classification") or {}
    use_reg = bool(reg) and reg["xgboost"]["mae"] < reg["naive_now"]["mae"]
    use_cls = bool(cls) and cls["xgboost"]["f1"] > cls["naive_now"]["f1"]
    return use_reg, use_cls


def main():
    t0 = time.time()
    if not (MODELS / "meta.json").exists():
        raise SystemExit("no models_v2/meta.json — train first (train_v2.py)")
    meta = json.load(open(MODELS / "meta.json"))
    if not meta["horizons"]:
        raise SystemExit("models_v2/meta.json has no trained horizons — keep collecting, retrain later")
    assert SLOT_HIST_DAYS + 1 <= HISTORY_DAYS, "HISTORY_DAYS too short for slot_hist_mean"
    rep_f = MODELS / "report.json"
    report = json.load(open(rep_f)) if rep_f.exists() else None
    cong = float(meta.get("congested_score", 5.5))
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

    rows, served = [], {}
    for h in meta["horizons"]:
        X = build_rows(panel, h, [t], with_target=False, hist=hist, unit_mask=mask)
        if not len(X):
            continue
        use_reg, use_cls = choose(report, h)
        use_cls = use_cls and str(h) in meta.get("thresholds", {}) and (MODELS / f"xgb_cls_h{h}.json").exists()
        if use_reg:
            reg = xgb.XGBRegressor()
            reg.load_model(MODELS / f"xgb_reg_h{h}.json")
            score = np.clip(reg.predict(X[meta["features"]]), FREE, 9.5)
        else:
            score = np.clip(naive_forecast(X), FREE, 9.5)
        if use_cls:
            cls = xgb.XGBClassifier()
            cls.load_model(MODELS / f"xgb_cls_h{h}.json")
            p_cong = cls.predict_proba(X[meta["features"]])[:, 1]
        else:                                    # 0/1: congested by the served score
            p_cong = (score >= cong).astype(float)
        served[str(h)] = {"score": "xgboost" if use_reg else "naive_now",
                          "p_congested": "xgboost" if use_cls else "naive_now"}
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
    tmp = MODELS / "serving.json.tmp"           # atomic: the API may read it at any moment
    with open(tmp, "w") as f:
        json.dump({"made_at": hi.isoformat(), "horizons": served}, f, indent=2)
    os.replace(tmp, MODELS / "serving.json")
    print(f"forecast from {hi:%Y-%m-%d %H:%M} UTC: {len(df):,} rows, horizons {meta['horizons']}, "
          f"served: " + ", ".join(f"h{h}={v['score']}" for h, v in served.items())
          + f", {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
