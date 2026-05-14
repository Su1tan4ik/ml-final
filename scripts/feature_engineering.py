"""
Feature engineering: data/staging.csv → append to data.csv → clear staging

Flow:
  1. Read data/staging.csv  (raw rows from latest scrape runs)
  2. Deduplicate against existing data.csv (by timestamp + segment_id)
  3. Compute all features, using tail of data.csv as lag context
  4. Append clean rows to data.csv
  5. Clear data/staging.csv

Run from repo root: python scripts/feature_engineering.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

STAGING_PATH   = os.environ.get("STAGING_PATH",   "data/staging.csv")
PROCESSED_PATH = os.environ.get("PROCESSED_PATH", "data.csv")
LAG_GAP_MINUTES  = 45
LAG_HISTORY_ROWS = 50   # rows per segment from data.csv used as lag context

ALMATY_CENTER = (43.2565, 76.9286)

MAJOR_STREETS = [
    "аль-фараби", "абая", "достык", "розыбакиева", "ауэзова",
    "байтурсынова", "жандосова", "толе би", "момышулы", "саина",
    "рыскулова", "суюнбая", "алтынсарина", "райымбека", "сейфуллина",
    "северное кольцо", "восточная объездная", "богенбай батыра",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

def is_major(street):
    if not isinstance(street, str):
        return 0
    s = street.lower()
    return int(any(m in s for m in MAJOR_STREETS))

def weather_severity(row):
    cond   = str(row.get("weather_condition", "")).lower()
    precip = float(row.get("weather_precip_1h", 0) or 0)
    vis    = float(row.get("weather_visibility_m", 10000) or 10000)
    if "thunderstorm" in cond or precip > 5 or vis < 500:            return 3
    if "rain" in cond or "snow" in cond or precip > 1 or vis < 2000: return 2
    if "drizzle" in cond or "mist" in cond or precip > 0:            return 1
    return 0

# ── feature blocks ────────────────────────────────────────────────────────────

def add_temporal(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    h_utc = df["timestamp"].dt.hour
    h_alm = (h_utc + 5) % 24
    df["hour_utc"]        = h_utc
    df["hour_almaty"]     = h_alm
    df["hour"]            = h_alm
    df["day_of_week"]     = df["timestamp"].dt.dayofweek
    df["day_of_month"]    = df["timestamp"].dt.day
    df["date_str"]        = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
    df["is_morning_rush"] = h_alm.isin([7, 8, 9]).astype(int)
    df["is_evening_rush"] = h_alm.isin([17, 18, 19]).astype(int)
    df["is_rush_hour"]    = ((df["is_morning_rush"] == 1) | (df["is_evening_rush"] == 1)).astype(int)
    df["is_night"]        = h_alm.isin([0, 1, 2, 3, 4, 5, 23]).astype(int)
    df["hour_sin"]        = np.sin(2 * np.pi * h_alm / 24)
    df["hour_cos"]        = np.cos(2 * np.pi * h_alm / 24)
    df["dow_sin"]         = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]         = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_holiday"]      = 0
    return df

def add_weather_features(df):
    df["weather_precip_1h"]    = pd.to_numeric(df.get("weather_precip_1h",    0),     errors="coerce").fillna(0)
    df["weather_visibility_m"] = pd.to_numeric(df.get("weather_visibility_m", 10000), errors="coerce").fillna(10000)
    df["weather_temp_c"]       = pd.to_numeric(df.get("weather_temp_c",       15),    errors="coerce").fillna(15)
    df["is_low_visibility"] = (df["weather_visibility_m"] < 1000).astype(int)
    df["is_cold"]           = (df["weather_temp_c"] < 0).astype(int)
    df["weather_severity"]  = df.apply(weather_severity, axis=1)
    df["rain_x_rush"]       = df["weather_precip_1h"] * df["is_rush_hour"]
    return df

def add_spatial_features(df, le=None):
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    street_col = df.get(
        "street_corrected",
        df.get("street_name", pd.Series([""] * len(df), index=df.index))
    )
    df["street_corrected"] = street_col.fillna("")
    df["major_street"]     = df["street_corrected"].apply(is_major)
    df["is_major_street"]  = df["major_street"]
    if le is None:
        le = LabelEncoder()
        df["street_encoded"] = le.fit_transform(df["street_corrected"].astype(str))
    else:
        mapping = {c: i for i, c in enumerate(le.classes_)}
        df["street_encoded"] = (
            df["street_corrected"].astype(str).map(mapping).fillna(0).astype(int)
        )
    df["dist_from_center"] = haversine_km(
        df["lat"], df["lon"], ALMATY_CENTER[0], ALMATY_CENTER[1]
    )
    return df, le

def add_lag_features(new_df, history_df=None):
    lag_gap = pd.Timedelta(minutes=LAG_GAP_MINUTES)

    if history_df is not None and len(history_df) > 0:
        ctx = history_df[["segment_id", "timestamp", "traffic_score"]].copy()
        ctx["_is_new"] = False
        wrk = pd.concat([ctx, new_df.assign(_is_new=True)], ignore_index=True)
    else:
        wrk = new_df.assign(_is_new=True)

    wrk["timestamp"]     = pd.to_datetime(wrk["timestamp"], utc=True)
    wrk["traffic_score"] = pd.to_numeric(wrk["traffic_score"], errors="coerce")
    wrk = wrk.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    n   = len(wrk)
    l1  = np.full(n, np.nan); l2  = np.full(n, np.nan); l48 = np.full(n, np.nan)
    rm6 = np.full(n, np.nan); rs6 = np.full(n, np.nan); rx6 = np.full(n, np.nan)

    for _, grp in wrk.groupby("segment_id"):
        idx    = grp.index.values
        scores = grp["traffic_score"].values
        times  = grp["timestamp"].values
        for ii, i in enumerate(idx):
            if ii >= 1 and (times[ii] - times[ii - 1]) <= lag_gap:
                l1[i] = scores[ii - 1]
            if ii >= 2 and (times[ii] - times[ii - 2]) <= lag_gap * 2:
                l2[i] = scores[ii - 2]
            if ii >= 48 and (times[ii] - times[ii - 48]) <= pd.Timedelta(hours=26):
                l48[i] = scores[ii - 48]
            window = [
                scores[jj] for jj in range(max(0, ii - 6), ii)
                if (times[ii] - times[jj]) <= lag_gap * 6 and not np.isnan(scores[jj])
            ]
            if window:
                rm6[i] = np.mean(window)
                rs6[i] = np.std(window) if len(window) > 1 else 0.0
                rx6[i] = np.max(window)

    wrk["lag_1"]          = l1;  wrk["lag_2"]  = l2;  wrk["lag_48"] = l48
    wrk["rolling_mean_6"] = rm6; wrk["rolling_std_6"] = rs6; wrk["rolling_max_6"] = rx6
    wrk["diff_1"]         = wrk["traffic_score"] - wrk["lag_1"]
    wrk["diff_2"]         = wrk["lag_1"] - wrk["lag_2"]
    wrk["lag_momentum"]   = wrk["lag_1"] - wrk["lag_2"]

    result = wrk[wrk["_is_new"] == True].drop(columns=["_is_new"]).copy()

    seg_avg = result.groupby("segment_id")["traffic_score"].transform("mean")
    result["segment_historical_avg"] = seg_avg

    for col in ["lag_1", "lag_2", "lag_48", "rolling_mean_6"]:
        result[col] = result[col].fillna(result["segment_historical_avg"])
    result["rolling_std_6"] = result["rolling_std_6"].fillna(0)
    result["rolling_max_6"] = result["rolling_max_6"].fillna(result["traffic_score"])
    result["diff_1"]        = result["diff_1"].fillna(0)
    result["diff_2"]        = result["diff_2"].fillna(0)
    result["lag_momentum"]  = result["lag_momentum"].fillna(0)
    return result

def clear_staging(path: Path):
    """Overwrite staging with just the header row — keeps schema, data gone."""
    header = pd.read_csv(path, nrows=0)
    header.to_csv(path, index=False)
    print(f"Staging cleared: {path}")

# ── main ──────────────────────────────────────────────────────────────────────

def run():
    staging_path = Path(STAGING_PATH)
    proc_path    = Path(PROCESSED_PATH)

    if not staging_path.exists():
        print(f"ERROR: staging file not found: {staging_path}")
        sys.exit(1)

    # ── 1. Read staging ───────────────────────────────────────────────
    staging = pd.read_csv(staging_path, low_memory=False)
    staging["timestamp"]     = pd.to_datetime(staging["timestamp"], utc=True)
    staging["traffic_score"] = pd.to_numeric(staging["traffic_score"], errors="coerce")
    staging = staging[staging["traffic_score"].notna() & (staging["traffic_score"] > 0)]

    print(f"Staging rows: {len(staging):,}")
    if len(staging) == 0:
        print("Staging is empty — nothing to do.")
        return

    # ── 2. Deduplicate against existing data.csv ──────────────────────
    history_df = None
    le         = None

    if proc_path.exists():
        existing = pd.read_csv(proc_path, low_memory=False)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        print(f"Existing data.csv: {len(existing):,} rows")

        # Only keep staging rows not already in data.csv
        existing_keys = set(
            zip(existing["timestamp"].astype(str), existing["segment_id"].astype(str))
        )
        staging["_key"] = list(zip(staging["timestamp"].astype(str), staging["segment_id"].astype(str)))
        new_rows = staging[~staging["_key"].isin(existing_keys)].drop(columns=["_key"])
        print(f"New unique rows (after dedup): {len(new_rows):,}")

        if len(new_rows) == 0:
            print("All staging rows already in data.csv — clearing staging.")
            clear_staging(staging_path)
            return

        # Lag context from existing data
        history_df = (
            existing
            .sort_values(["segment_id", "timestamp"])
            .groupby("segment_id")
            .tail(LAG_HISTORY_ROWS)
            [["segment_id", "timestamp", "traffic_score"]]
        )

        if "street_corrected" in existing.columns:
            le = LabelEncoder()
            le.classes_ = np.array(
                sorted(existing["street_corrected"].fillna("").astype(str).unique())
            )

        df = new_rows
    else:
        print("No existing data.csv — processing all staging data from scratch.")
        df = staging.copy()

    # Drop dead segments (always exactly 1.5) only on first run
    if not proc_path.exists():
        seg_std = df.groupby("segment_id")["traffic_score"].std()
        live    = seg_std[seg_std > 0.01].index
        df      = df[df["segment_id"].isin(live)].copy()

    # ── 3. Feature engineering ────────────────────────────────────────
    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    df = add_temporal(df)
    df = add_weather_features(df)
    df, _ = add_spatial_features(df, le)
    df = add_lag_features(df, history_df)

    # ── 4. Append to data.csv ─────────────────────────────────────────
    write_header = not proc_path.exists()
    if not write_header:
        existing_cols = pd.read_csv(proc_path, nrows=0).columns.tolist()
        for col in existing_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[existing_cols]
    df.to_csv(proc_path, mode="a", index=False, header=write_header)
    print(f"Appended {len(df):,} rows → {proc_path}")

    total = sum(1 for _ in open(proc_path)) - 1
    print(f"Total rows in data.csv: {total:,}")

    # ── 5. Clear staging (only after successful append) ───────────────
    clear_staging(staging_path)

if __name__ == "__main__":
    run()
