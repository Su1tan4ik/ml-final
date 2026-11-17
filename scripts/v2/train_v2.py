"""
V2 training: forecast traffic score of each (edge, direction) h slots ahead (h=1 -> 30 min).

Honest-evaluation rules carried over from V1:
  * out-of-time split by TARGET time (last TEST_SHARE of slots = test)
  * "active" units decided on the train period only (always-free units are not modelled;
    for them the forecast is the free-flow value)
  * baselines: naive persistence (score now) and causal time-of-day profile (slot_hist_mean)
  * classifier threshold tuned on a time-ordered validation tail of train, never on test
  * no leaky fills: missing values stay NaN (XGBoost handles them)

    venv/bin/python scripts/v2/train_v2.py                      # horizons 1,2,4
    venv/bin/python scripts/v2/train_v2.py --horizons 1 --days 14
Outputs: models_v2/xgb_reg_h{h}.json, xgb_cls_h{h}.json, meta.json, report.json, active_units.csv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

from common import ROOT
from db import connect
from features_v2 import FEATURES, FREE, GOOD_SLOT_SHARE, _slot_hist, build_rows, load_matrix, naive_forecast

OUT = ROOT / "models_v2"
TEST_SHARE = 0.2
VAL_SHARE = 0.15
CONGESTED = 5.5          # orange or worse
FREE_BELOW = 2.5
INACTIVE_FREE_SHARE = 0.99
MIN_UNIT_OBS = 10
MIN_SLOTS = 4            # below this nothing can be split into train/test at all
MAX_TRAIN_ROWS = 4_000_000

REG_PARAMS = dict(n_estimators=600, max_depth=8, learning_rate=0.06, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                  early_stopping_rounds=40, n_jobs=-1)
CLS_PARAMS = dict(REG_PARAMS, eval_metric="aucpr")


def reg_metrics(y, p):
    return {"mae": round(float(mean_absolute_error(y, p)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            "r2": round(float(r2_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
            "n": int(len(y))}


def cls_metrics(y, pred, proba=None):
    m = {"f1": round(float(f1_score(y, pred, zero_division=0)), 4),
         "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
         "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
         "positive_rate": round(float(np.mean(y)), 4)}
    if proba is not None and 0 < y.sum() < len(y):
        m["roc_auc"] = round(float(roc_auc_score(y, proba)), 4)
    return m


fallback = naive_forecast   # prediction without the model: now -> typical -> free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--days", type=float, help="use only the last N days")
    ap.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_ROWS)
    a = ap.parse_args()
    with connect() as conn:
        lo, hi = conn.execute("SELECT min(ts), max(ts) FROM traffic_obs").fetchone()
        if lo is None:
            raise SystemExit("no data in traffic_obs")
        if a.days:
            lo = max(lo, hi - timedelta(days=a.days))
        panel = load_matrix(conn, lo, hi, min_ok_share=GOOD_SLOT_SHARE)   # partial scrapes = missing
    run(panel, a.horizons, a.max_train_rows, OUT)


def publish(stage, out):
    """Move the freshly trained files from `stage` into `out` one by one with os.replace
    (atomic per file; meta.json last) and drop model files of horizons not trained this
    time. predict_v2 runs every 30 min in parallel with training and must never read a
    half-written model or an old classifier left over from an earlier run."""
    new = sorted(stage.iterdir(), key=lambda f: f.name == "meta.json")
    for f in new:
        os.replace(f, out / f.name)
    names = {f.name for f in new}
    for f in out.glob("xgb_*_h*.json"):
        if f.name not in names:
            f.unlink()
    shutil.rmtree(stage, ignore_errors=True)


def run(panel, horizons, max_train_rows=MAX_TRAIN_ROWS, out=OUT):
    t0 = time.time()
    final, out = out, out / ".staging"           # write here, publish() at the end
    shutil.rmtree(out, ignore_errors=True)
    lo, hi = panel.slots[0], panel.slots[-1]
    S = panel.S
    T = S.shape[1]
    have = (~np.isnan(S)).any(0)
    print(f"panel: {S.shape[0]:,} units x {T} slots ({lo:%m-%d %H:%M} .. {hi:%m-%d %H:%M} UTC), "
          f"{have.sum()} slots with data")
    if have.sum() < MIN_SLOTS:
        raise SystemExit(f"only {have.sum()} slots with data — need at least {MIN_SLOTS}. Keep collecting.")
    small = T < 48                               # < 1 day: shorter warm-up, looser unit filter
    start = 1 if small else 5                    # first origin slot (lags before it are NaN anyway)

    split = int(T * (1 - TEST_SHARE))            # first test TARGET slot
    tr = S[:, :split]
    n_obs = (~np.isnan(tr)).sum(1)
    with np.errstate(invalid="ignore"):
        free_share = np.nansum(tr < FREE_BELOW, 1) / np.maximum(n_obs, 1)
    min_obs = min(MIN_UNIT_OBS, max(2, int(0.6 * have[:split].sum())))
    active = (n_obs >= min_obs) & (free_share < INACTIVE_FREE_SHARE)
    print(f"active units (train period): {active.sum():,} / {len(active):,}")

    hist = _slot_hist(S, panel.slots)
    out.mkdir(parents=True, exist_ok=True)
    report = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "data": {"from": lo.isoformat(), "to": hi.isoformat(), "slots": int(T),
                       "slots_with_data": int(have.sum()), "units": int(S.shape[0]),
                       "active_units": int(active.sum()), "test_from": str(panel.slots[split]),
                       "hours_covered": round(float(have.sum()) / 2, 1), "min_unit_obs": int(min_obs),
                       "small_data": bool(have.sum() < 48 * 7)},
              "congested_score": CONGESTED, "horizons": {}}

    for h in horizons:
        th = time.time()
        tr_orig = np.arange(start, split - h)
        te_orig = np.arange(max(start, split - h), T - h)
        if len(tr_orig) < (2 if small else 4) or len(te_orig) < 1:
            print(f"h={h}: not enough slots, skip")
            continue
        est = active.sum() * len(tr_orig) * have.mean()
        frac = min(1.0, max_train_rows / max(est, 1))
        train = build_rows(panel, h, tr_orig, hist=hist, unit_mask=active, frac=frac, seed=h)
        test = build_rows(panel, h, te_orig, hist=hist, unit_mask=active)
        if len(train) < 1000 or len(test) < 100:
            print(f"h={h}: too few rows (train {len(train)}, test {len(test)}), skip")
            continue
        train = train.sort_values("t", kind="stable")
        cut = int(len(train) * (1 - VAL_SHARE))
        fit, val = train.iloc[:cut], train.iloc[cut:]

        reg = xgb.XGBRegressor(**REG_PARAMS)
        reg.fit(fit[FEATURES], fit["y"], eval_set=[(val[FEATURES], val["y"])], verbose=False)
        pred = reg.predict(test[FEATURES])

        ycls_fit, ycls_val = (fit["y"] >= CONGESTED).astype(int), (val["y"] >= CONGESTED).astype(int)
        res_h = {"rows": {"train": len(fit), "val": len(val), "test": len(test), "sample_frac": round(frac, 3)}}
        y = test["y"].values
        naive = fallback(test)
        prof = test["slot_hist_mean"].fillna(test["cur"]).fillna(FREE).values
        cong = y >= CONGESTED
        res_h["regression"] = {
            "xgboost": reg_metrics(y, pred), "naive_now": reg_metrics(y, naive),
            "time_profile": reg_metrics(y, prof),
            "on_congested_targets": {"xgboost": reg_metrics(y[cong], pred[cong]),
                                     "naive_now": reg_metrics(y[cong], naive[cong])} if cong.sum() >= 20 else None,
            "best_iteration": int(reg.best_iteration)}

        if ycls_fit.sum() >= 20 and ycls_val.sum() >= 5:
            cls = xgb.XGBClassifier(**CLS_PARAMS)
            cls.fit(fit[FEATURES], ycls_fit, eval_set=[(val[FEATURES], ycls_val)], verbose=False)
            pv = cls.predict_proba(val[FEATURES])[:, 1]
            grid = np.arange(0.1, 0.91, 0.025)
            thr = float(grid[np.argmax([f1_score(ycls_val, pv >= g, zero_division=0) for g in grid])])
            pt = cls.predict_proba(test[FEATURES])[:, 1]
            res_h["classification"] = {"threshold": round(thr, 3),
                                       "xgboost": cls_metrics(cong.astype(int), pt >= thr, pt),
                                       "naive_now": cls_metrics(cong.astype(int), naive >= CONGESTED)}
            cls.save_model(out / f"xgb_cls_h{h}.json")
        else:
            res_h["classification"] = None
            print(f"h={h}: too few congested examples for a classifier")

        imp = pd.Series(reg.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
        res_h["feature_gain_share"] = (imp / imp.sum()).round(4).sort_values(ascending=False).head(12).to_dict()
        reg.save_model(out / f"xgb_reg_h{h}.json")
        report["horizons"][str(h)] = res_h
        r = res_h["regression"]
        c = res_h["classification"]
        print(f"h={h} ({30 * h} min): MAE xgb {r['xgboost']['mae']} vs naive {r['naive_now']['mae']} "
              f"vs profile {r['time_profile']['mae']} | R2 {r['xgboost']['r2']} vs {r['naive_now']['r2']}"
              + (f" | F1 {c['xgboost']['f1']} vs naive {c['naive_now']['f1']} (thr {c['threshold']})" if c else "")
              + f" | {time.time() - th:.0f}s")

    units = panel.units[["edge_id", "dir"]].assign(active=active)
    units.to_csv(out / "active_units.csv", index=False)
    json.dump({"features": FEATURES, "congested_score": CONGESTED, "free_score": FREE,
               "thresholds": {h: v["classification"]["threshold"]
                              for h, v in report["horizons"].items() if v.get("classification")},
               "horizons": [int(h) for h in report["horizons"]]},
              open(out / "meta.json", "w"), indent=2)
    json.dump(report, open(out / "report.json", "w"), indent=2, ensure_ascii=False)
    publish(out, final)
    print(f"saved to {final} in {time.time() - t0:.0f}s")
    return report


if __name__ == "__main__":
    main()
