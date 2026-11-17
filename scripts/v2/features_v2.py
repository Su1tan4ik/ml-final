"""
V2 features for (edge, direction) "units", built as a dense matrix S[unit, slot].

Why a matrix: ~87k units x 48 slots/day. Long format with groupby/shift would be tens of
millions of rows; S is float32 (NaN = no observation), and every feature is a vectorised
shift / window / sparse-matrix product over it. All features at forecast origin t use
only slots <= t (no leakage); calendar features describe the target slot t+h.

Main entry points:
  load_matrix(conn, start, end)   -> Panel (S, units, slots, static, weather)
  build_rows(panel, h, t_idx)     -> DataFrame of features (+ target) for origins t_idx
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

SLOT = timedelta(minutes=30)
PER_DAY = 48
SLOT_HIST_DAYS = 7  # window of slot_hist_mean; must fit in predict_v2.HISTORY_DAYS
GOOD_SLOT_SHARE = 0.8  # a slot counts for training/statistics if >= this share of tiles was fetched
FREE = 1.5
ALMATY_UTC_OFFSET = 5  # hours; Almaty is UTC+5 all year

HIGHWAY_CODE = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4,
                "unclassified": 5, "residential": 6, "living_street": 7, "service": 8}

FEATURES = [
    # recent history at origin t
    "cur", "lag_1", "lag_2", "lag_3", "diff_1", "roll_mean_6", "roll_std_6", "roll_max_6", "obs_6",
    # same time as the TARGET on previous days
    "day_ago", "week_ago", "slot_hist_mean", "slot_hist_n",
    # neighbours (edges sharing a node) at origin t
    "nb_mean", "nb_max", "nb_mean_lag_1",
    # calendar of the target slot (Almaty local time)
    "hour_sin", "hour_cos", "dow", "is_weekend", "minutes_ahead",
    # static road attributes
    "highway", "dir", "oneway", "lanes", "length_m", "lat", "lon",
    # weather at origin
    "temp_c", "precip_1h",
]


@dataclass
class Panel:
    S: np.ndarray            # float32 [n_units, n_slots]
    units: pd.DataFrame      # edge_id, dir + static columns, index = unit row
    slots: pd.DatetimeIndex  # UTC, regular 30-min grid
    weather: pd.DataFrame    # indexed by slot position
    adj: sp.csr_matrix       # unit x unit neighbour matrix (0/1)


def _sql(q, conn, params=None) -> pd.DataFrame:
    """psycopg3 query -> DataFrame (pd.read_sql warns on non-SQLAlchemy connections)."""
    with conn.cursor() as cur:
        cur.execute(q, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def _floor_slot(t: datetime) -> datetime:
    t = t.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return t - timedelta(minutes=t.minute % 30)


def good_slots_sql(min_ok_share: float) -> str:
    """Subquery of slot times whose scrape fetched at least min_ok_share of the tiles.
    A partial run (blocked / network errors) drops the tail of the tile list, i.e. the
    outskirts, so its slot is biased; runs without tile counters (tests) are kept."""
    return ("SELECT ts FROM scrape_run WHERE finished_at IS NOT NULL AND (coalesce(tiles_total, 0) = 0 "
            f"OR coalesce(tiles_ok, 0) + coalesce(tiles_empty, 0) >= {float(min_ok_share)} * tiles_total)")


def load_matrix(conn, start: datetime, end: datetime, min_ok_share: float | None = None) -> Panel:
    """Load traffic_obs in [start, end] into the dense matrix + static info.
    min_ok_share: skip slots whose scrape got fewer tiles (they stay all-NaN columns);
    used for training / statistics, not for live forecasting."""
    start, end = _floor_slot(start), _floor_slot(end)
    slots = pd.date_range(start, end, freq="30min", tz="UTC")
    good = "" if min_ok_share is None else f" AND ts IN ({good_slots_sql(min_ok_share)})"
    obs = _sql(
        "SELECT ts, edge_id, dir, score FROM traffic_obs WHERE ts BETWEEN %(a)s AND %(b)s" + good,
        conn, params={"a": start, "b": end})
    units = (obs[["edge_id", "dir"]].drop_duplicates()
             .sort_values(["edge_id", "dir"]).reset_index(drop=True))
    uidx = pd.Series(np.arange(len(units)), index=pd.MultiIndex.from_frame(units))
    S = np.full((len(units), len(slots)), np.nan, np.float32)
    if len(obs):
        r = uidx.loc[pd.MultiIndex.from_frame(obs[["edge_id", "dir"]])].values
        c = ((pd.to_datetime(obs["ts"], utc=True) - slots[0]) // pd.Timedelta(minutes=30)).values
        S[r, c] = obs["score"].values

    static = _sql(
        """SELECT edge_id, source, target, highway, oneway, length_m, lanes,
                  ST_Y(ST_Centroid(geom)) AS lat, ST_X(ST_Centroid(geom)) AS lon
           FROM road_edge WHERE edge_id = ANY(%(ids)s)""",
        conn, params={"ids": units["edge_id"].unique().tolist()})
    units = units.merge(static, on="edge_id", how="left")
    units["lanes"] = units["lanes"].astype(np.float32)
    units["highway"] = units["highway"].str.replace("_link", "").map(HIGHWAY_CODE).fillna(9).astype(np.int8)

    weather = _sql("SELECT ts, temp_c, precip_1h FROM weather_obs WHERE ts BETWEEN %(a)s AND %(b)s",
                          conn, params={"a": start, "b": end})
    w = pd.DataFrame(index=slots, columns=["temp_c", "precip_1h"], dtype=np.float32)
    if len(weather):
        weather["ts"] = pd.to_datetime(weather["ts"], utc=True)
        w.loc[weather["ts"], ["temp_c", "precip_1h"]] = weather[["temp_c", "precip_1h"]].values
    w = w.ffill(limit=2).reset_index(drop=True)

    return Panel(S, units, slots, w, _adjacency(units))


def _adjacency(units: pd.DataFrame) -> sp.csr_matrix:
    """Units are neighbours if their edges share a node (excluding the unit itself)."""
    n = len(units)
    node_unit = pd.concat([units[["source"]].rename(columns={"source": "node"}).assign(u=np.arange(n)),
                           units[["target"]].rename(columns={"target": "node"}).assign(u=np.arange(n))])
    pairs = node_unit.merge(node_unit, on="node")
    pairs = pairs[pairs.u_x != pairs.u_y].drop_duplicates(["u_x", "u_y"])
    return sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs.u_x, pairs.u_y)), shape=(n, n))


# ------------------------------------------------------------------ helpers
def _col(S, j):
    """Column j of S, all-NaN if out of range (works for vector j)."""
    j = np.asarray(j)
    ok = (j >= 0) & (j < S.shape[1])
    out = np.full((S.shape[0], j.size), np.nan, np.float32)
    out[:, ok] = S[:, j[ok]]
    return out


def _slot_hist(S: np.ndarray, slots: pd.DatetimeIndex, days: float | None = SLOT_HIST_DAYS):
    """
    Causal "typical value" per unit for each slot: mean of S over the same local
    time-of-day and same weekday/weekend type on PREVIOUS days only, limited to the
    last `days` days (None = all history). A fixed window keeps the feature identical
    in training (28 days of data) and in live forecasting (HISTORY_DAYS of data).
    Assumes `slots` is a regular 30-min grid (as built by load_matrix / extend).
    Returns (mean, count) arrays shaped like S.
    """
    local = slots + pd.Timedelta(hours=ALMATY_UTC_OFFSET)
    key = np.asarray((local.hour * 2 + local.minute // 30) * 2 + (local.dayofweek >= 5))
    win = None if days is None else int(round(days * PER_DAY))
    mean = np.full(S.shape, np.nan, np.float32)
    cnt = np.zeros(S.shape, np.float32)
    zero = np.zeros((S.shape[0], 1))
    for k in np.unique(key):
        cols = np.flatnonzero(key == k)
        X = S[:, cols]
        cv = np.hstack([zero, np.cumsum(np.nan_to_num(X), axis=1, dtype=np.float64)])
        cm = np.hstack([zero, np.cumsum(~np.isnan(X), axis=1, dtype=np.float64)])
        cur = np.arange(len(cols))
        # earlier occurrences inside the window: positions [lo, cur)
        lo = np.zeros_like(cur) if win is None else np.searchsorted(cols, cols - win, side="left")
        s, n = cv[:, cur] - cv[:, lo], cm[:, cur] - cm[:, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[:, cols] = np.where(n > 0, s / n, np.nan)
        cnt[:, cols] = n
    return mean, cnt


def extend(p: Panel, k: int) -> Panel:
    """Append k empty future slots. Live forecasting builds features at the last observed
    slot for targets up to k slots ahead; without the padding the target slot is outside
    the matrix and slot_hist_mean / slot_hist_n come out empty (unlike in training)."""
    if k <= 0:
        return p
    S = np.hstack([p.S, np.full((p.S.shape[0], k), np.nan, np.float32)])
    slots = p.slots.append(pd.date_range(p.slots[-1] + SLOT, periods=k, freq="30min"))
    return Panel(S, p.units, slots, p.weather, p.adj)


def _nb(adj: sp.csr_matrix, x: np.ndarray):
    """Neighbour mean and max of vector x (NaN-aware)."""
    m = ~np.isnan(x)
    s = adj @ np.where(m, x, 0)
    c = adj @ m.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / c, np.nan)
    # max via sparse: replace missing with -inf
    xx = np.where(m, x, -np.inf)
    A = adj.tocoo()
    mx = np.full(len(x), -np.inf, np.float32)
    np.maximum.at(mx, A.row, xx[A.col])
    mx[np.isinf(mx)] = np.nan
    return mean.astype(np.float32), mx


def build_rows(p: Panel, h: int, t_idx, with_target: bool = True,
               hist: tuple | None = None, unit_mask: np.ndarray | None = None,
               frac: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """
    Feature rows for every unit at each origin slot index in t_idx, forecasting t+h.
    If with_target, rows whose target is missing are dropped.
    hist = precomputed _slot_hist(S, slots) to avoid recomputation.
    unit_mask = only these units get rows (neighbour features still use all units).
    frac < 1 = random subsample of rows (memory cap for training).
    """
    rng = np.random.default_rng(seed)
    S, n = p.S, p.S.shape[0]
    hm, hn = hist if hist is not None else _slot_hist(S, p.slots)
    static = p.units
    frames = []
    for t in np.atleast_1d(t_idx):
        tgt = t + h
        y = _col(S, [tgt])[:, 0] if with_target else None
        keep = ~np.isnan(y) if with_target else np.ones(n, bool)
        if unit_mask is not None:
            keep &= unit_mask
        if frac < 1:
            keep &= rng.random(n) < frac
        if not keep.any():
            continue
        W = _col(S, np.arange(t - 5, t + 1))       # last 6 slots incl. t
        cur = W[:, -1]
        with np.errstate(invalid="ignore"):
            rm, rs, rx = np.nanmean(W, 1), np.nanstd(W, 1), np.nanmax(W, 1)
        nbm, nbx = _nb(p.adj, cur)
        nbm1, _ = _nb(p.adj, W[:, -2])
        ts_t = p.slots[0] + pd.Timedelta(minutes=30 * tgt)
        local = ts_t + pd.Timedelta(hours=ALMATY_UTC_OFFSET)
        hour = local.hour + local.minute / 60
        in_range = 0 <= tgt < S.shape[1]
        f = pd.DataFrame({
            "cur": cur, "lag_1": W[:, -2], "lag_2": W[:, -3], "lag_3": W[:, -4],
            "diff_1": cur - W[:, -2], "roll_mean_6": rm, "roll_std_6": rs, "roll_max_6": rx,
            "obs_6": (~np.isnan(W)).sum(1).astype(np.int8),
            "day_ago": _col(S, [tgt - PER_DAY])[:, 0], "week_ago": _col(S, [tgt - 7 * PER_DAY])[:, 0],
            "slot_hist_mean": hm[:, tgt] if in_range else np.nan,
            "slot_hist_n": hn[:, tgt] if in_range else 0,
            "nb_mean": nbm, "nb_max": nbx, "nb_mean_lag_1": nbm1,
            "hour_sin": np.float32(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": np.float32(np.cos(2 * np.pi * hour / 24)),
            "dow": np.int8(local.dayofweek), "is_weekend": np.int8(local.dayofweek >= 5),
            "minutes_ahead": np.int16(30 * h),
            "highway": static["highway"].values, "dir": static["dir"].values.astype(np.int8),
            "oneway": static["oneway"].values.astype(np.int8), "lanes": static["lanes"].values,
            "length_m": static["length_m"].values, "lat": static["lat"].values, "lon": static["lon"].values,
            "temp_c": p.weather["temp_c"].iloc[t] if t < len(p.weather) else np.nan,
            "precip_1h": p.weather["precip_1h"].iloc[t] if t < len(p.weather) else np.nan,
        })
        f["unit"] = np.arange(n)
        f["t"] = t
        if with_target:
            f["y"] = y
        frames.append(f[keep])
    if not frames:
        return pd.DataFrame(columns=FEATURES + ["unit", "t"] + (["y"] if with_target else []))
    out = pd.concat(frames, ignore_index=True)
    # slot_hist_mean is the typical value for the target slot; when there is no history
    # yet, leave NaN (XGBoost handles missing natively — no leaky fill)
    return out
