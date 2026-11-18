"""
Offline checks for the V2 model code (no DB, no network):
  1. no leakage: changing any data AFTER the origin slot must not change features
  2. live forecasting (origin = last observed slot) builds the same features as training
  3. the map gets the model only where it beat the naive "as now" forecast
  4. end-to-end training on a synthetic city (rush hours + AR noise + missing values)

    cd scripts/v2 && ../../venv/bin/python test_v2.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from features_v2 import FEATURES, SLOT_HIST_DAYS, Panel, _adjacency, _slot_hist, build_rows, extend
from train_v2 import run


def synthetic_panel(n_units=400, days=10, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    slots = pd.date_range("2026-11-04", periods=48 * days, freq="30min", tz="UTC")
    local_h = np.asarray((slots.hour + 5) % 24 + slots.minute / 60)
    weekend = np.asarray((slots + pd.Timedelta(hours=5)).dayofweek >= 5)
    rush = np.exp(-((local_h - 8.5) ** 2) / 0.8) + np.exp(-((local_h - 18.5) ** 2) / 1.2)
    rush = rush * np.where(weekend, 0.3, 1.0)
    intensity = rng.gamma(1.5, 2.0, n_units)
    intensity[: n_units // 3] = 0                     # a third of roads always free
    S = np.empty((n_units, len(slots)), np.float32)
    e = np.zeros(n_units)
    for j in range(len(slots)):
        e = 0.7 * e + rng.normal(0, 0.5, n_units)     # persistent disturbances
        S[:, j] = np.clip(1.5 + intensity * rush[j] + np.where(intensity > 0, e, 0), 1.5, 9.5)
    S[rng.random(S.shape) < 0.1] = np.nan             # missing observations
    S[:, 100] = np.nan                                 # a failed scrape run
    units = pd.DataFrame({"edge_id": np.arange(n_units), "dir": 0, "source": np.arange(n_units),
                          "target": np.arange(1, n_units + 1), "highway": np.int8(3), "oneway": 0,
                          "length_m": 100.0, "lanes": np.nan, "lat": 43.25, "lon": 76.9})
    weather = pd.DataFrame({"temp_c": np.float32(5.0), "precip_1h": np.float32(0.0)}, index=range(len(slots)))
    return Panel(S, units, slots, weather, _adjacency(units))


def test_no_leakage():
    p = synthetic_panel(n_units=60, days=9)
    rng = np.random.default_rng(1)
    for h in (1, 2, 4):
        for t in (60, 200, 300):
            a = build_rows(p, h, [t], with_target=False, hist=_slot_hist(p.S, p.slots))
            S2 = p.S.copy()
            S2[:, t + 1:] = rng.uniform(1.5, 9.5, S2[:, t + 1:].shape)   # rewrite the future
            p2 = Panel(S2, p.units, p.slots, p.weather, p.adj)
            b = build_rows(p2, h, [t], with_target=False, hist=_slot_hist(S2, p.slots))
            pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])
    print("ok: features use only data up to the origin slot (h=1,2,4)")


def test_serving_matches_training():
    """predict_v2 forecasts from the LAST observed slot, so the target lies beyond the
    matrix; after extend() its features must equal those training saw for the same moment."""
    p = synthetic_panel(n_units=60, days=10)
    T = p.S.shape[1]
    for h in (1, 2, 4):
        t = T - 1 - h                                    # target still inside the full panel
        ref = build_rows(p, h, [t], with_target=False, hist=_slot_hist(p.S, p.slots))
        now = Panel(p.S[:, :t + 1], p.units, p.slots[:t + 1], p.weather.iloc[:t + 1], p.adj)
        live = extend(now, 4)                            # what predict_v2 does
        got = build_rows(live, h, [t], with_target=False, hist=_slot_hist(live.S, live.slots))
        pd.testing.assert_frame_equal(ref[FEATURES].reset_index(drop=True), got[FEATURES].reset_index(drop=True))
        assert got["slot_hist_mean"].notna().mean() > 0.5
        # the window bounds slot history: at most 5 weekdays / 2 weekend days in 7 days
        assert got["slot_hist_n"].max() <= 5
    assert SLOT_HIST_DAYS == 7
    print("ok: live-forecast features == training features (h=1,2,4)")


def test_serving_choice():
    from predict_v2 import choose
    m = lambda x, n: {"xgboost": {"mae": x}, "naive_now": {"mae": n}}
    f = lambda x, n: {"xgboost": {"f1": x}, "naive_now": {"f1": n}}
    rep = {"horizons": {"1": {"regression": m(0.588, 0.379), "classification": None},   # half a day of data
                        "2": {"regression": m(0.30, 0.40), "classification": f(0.7, 0.6)},
                        "4": {"regression": m(0.50, 0.60), "classification": f(0.5, 0.6)}}}
    assert choose(rep, 1) == (False, False)
    assert choose(rep, 2) == (True, True)
    assert choose(rep, 4) == (True, False)
    assert choose(None, 1) == (False, False) and choose(rep, 8) == (False, False)
    print("ok: model served only where it beats the naive forecast")


def test_training():
    p = synthetic_panel()
    with tempfile.TemporaryDirectory() as d:
        rep = run(p, [1, 2], out=Path(d))
        for h, r in rep["horizons"].items():
            x, n = r["regression"]["xgboost"]["mae"], r["regression"]["naive_now"]["mae"]
            assert x < n, f"h={h}: model ({x}) should beat naive ({n}) on synthetic rush hours"
            assert (Path(d) / f"xgb_reg_h{h}.json").exists()
    print("ok: end-to-end training on synthetic data, model beats naive")


if __name__ == "__main__" and "--db" not in __import__("sys").argv:
    test_no_leakage()
    test_serving_matches_training()
    test_serving_choice()
    test_training()


def test_db_end_to_end():
    """
    Separate DB `traffic_test` in the same container: real graph + synthetic observations
    on 1 500 real edges -> train_v2 -> predict_v2 -> forecast rows. Never touches `traffic`.
    """
    import os
    import subprocess
    import sys
    from datetime import timedelta

    env = dict(os.environ, PG_DB="traffic_test")
    dex = ["docker", "exec", "-i", "traffic_db"]
    subprocess.run(dex + ["dropdb", "-U", "traffic", "--if-exists", "traffic_test"], check=True)
    subprocess.run(dex + ["createdb", "-U", "traffic", "traffic_test"], check=True)
    root = Path(__file__).resolve().parents[2]
    for f in ("db/schema.sql", "db/functions.sql"):
        subprocess.run(dex + ["psql", "-q", "-U", "traffic", "-d", "traffic_test"],
                       stdin=open(root / f), check=True, stdout=subprocess.DEVNULL)
    py = sys.executable
    subprocess.run([py, "load_graph_db.py"], env=env, check=True)

    os.environ["PG_DB"] = "traffic_test"
    from db import connect
    with connect() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT edge_id FROM road_edge WHERE highway IN ('primary','secondary') ORDER BY edge_id LIMIT 1500")]
    p = synthetic_panel(n_units=len(ids), days=9)
    with connect() as conn, conn.cursor() as cur:
        with cur.copy("COPY traffic_obs (ts, edge_id, dir, score, n_px) FROM STDIN") as cp:
            for j, ts in enumerate(p.slots):
                for i in np.flatnonzero(~np.isnan(p.S[:, j])):
                    cp.write_row((ts.to_pydatetime(), ids[i], 0, float(p.S[i, j]), 50))
        for ts in p.slots:
            cur.execute("INSERT INTO scrape_run (ts, zoom, finished_at) VALUES (%s, 16, now())", (ts.to_pydatetime(),))
    out = root / "data" / "v2" / "test_models"
    subprocess.run([py, "-c", f"import train_v2 as t; t.OUT = __import__('pathlib').Path('{out}'); "
                    "import sys; sys.argv=['x','--horizons','1','2']; t.main()"], env=env, check=True)
    subprocess.run([py, "-c", f"import predict_v2 as p; p.MODELS = __import__('pathlib').Path('{out}'); p.main()"],
                   env=env, check=True)
    with connect() as conn:
        n, hs = conn.execute("SELECT count(*), array_agg(DISTINCT target_ts - made_at) FROM forecast").fetchone()
    assert n > 0 and len(hs) == 2, (n, hs)
    print(f"ok: DB end-to-end — {n} forecast rows for horizons {sorted(h // timedelta(minutes=30) for h in hs)}")


if __name__ == "__main__" and "--db" in __import__("sys").argv:
    test_db_end_to_end()
