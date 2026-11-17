"""
V2 recommendations (V1's SHAP idea, now per street and direction-aware).

1. Hotspot streets: length-weighted mean score over the observed period (named streets only,
   >= MIN_STREET_M of observed road), top N_STREETS.
2. For their (edge, dir) units, XGBoost per-row contributions (= exact TreeSHAP,
   booster.predict(pred_contribs=True)) of the shortest-horizon regressor.
3. Mean |contribution| grouped into history / neighbours / time / road / weather.
   History always dominates (autoregressive model) — shown honestly; the TYPE is chosen
   from the structural part (everything except history and location), as in V1:
     weather >= 20%                     -> weather
     neighbours largest                 -> traffic_light  (jam spills over from adjacent links:
                                                           signal coordination / junction)
     time > road + 12 pp                -> peak           (time-of-day driven: peak-hour signal plans,
                                                           reversible lanes)
     road > time + 12 pp                -> lane_expansion (structural bottleneck)
     otherwise                          -> complex
Output: models_v2/recommendations.json

    venv/bin/python scripts/v2/recommend_v2.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from common import ROOT
from db import connect
from features_v2 import FEATURES, _slot_hist, _sql, build_rows, load_matrix

MODELS = ROOT / "models_v2"
DAYS = 7
N_STREETS = 40
MIN_STREET_M = 300
CONGESTED = 5.5
ALMATY_UTC_OFFSET = 5
MAX_ROWS_PER_STREET = 400

GROUPS = {
    "history": ["cur", "lag_1", "lag_2", "lag_3", "diff_1", "roll_mean_6", "roll_std_6", "roll_max_6",
                "obs_6", "day_ago", "week_ago", "slot_hist_mean", "slot_hist_n"],
    "neighbours": ["nb_mean", "nb_max", "nb_mean_lag_1"],
    "time": ["hour_sin", "hour_cos", "dow", "is_weekend", "minutes_ahead"],
    "road": ["highway", "dir", "oneway", "lanes", "length_m"],
    "location": ["lat", "lon"],   # shown, but not used to pick the type (place is not a cause)
    "weather": ["temp_c", "precip_1h"],
}
FEATURE_RU = {
    "cur": "балл сейчас", "lag_1": "балл 30 мин назад", "lag_2": "балл 1 ч назад", "lag_3": "балл 1.5 ч назад",
    "diff_1": "изменение за 30 мин", "roll_mean_6": "среднее за 3 ч", "roll_std_6": "колебания за 3 ч",
    "roll_max_6": "максимум за 3 ч", "obs_6": "наблюдений за 3 ч", "day_ago": "сутки назад",
    "week_ago": "неделю назад", "slot_hist_mean": "типично в это время", "slot_hist_n": "история слота",
    "nb_mean": "соседние участки (среднее)", "nb_max": "соседние участки (макс.)",
    "nb_mean_lag_1": "соседи 30 мин назад", "hour_sin": "время суток", "hour_cos": "время суток",
    "dow": "день недели", "is_weekend": "выходной", "minutes_ahead": "горизонт прогноза",
    "highway": "класс дороги", "dir": "направление", "oneway": "односторонняя", "lanes": "число полос",
    "length_m": "длина участка", "lat": "широта", "lon": "долгота", "temp_c": "температура",
    "precip_1h": "осадки",
}
TYPES = {
    "traffic_light": ("🚦", "Координация светофоров на перекрёстках",
                      "Пробка в основном приходит с соседних участков (эффект перекрёстков). "
                      "Имеет смысл проверить фазы светофоров и «зелёную волну» на этой улице."),
    "peak": ("⏰", "Регулирование в часы пик",
             "Загрузка сильно зависит от времени суток. Варианты: отдельный план светофоров "
             "на час пик, реверсивные полосы, ограничение левых поворотов в пик."),
    "lane_expansion": ("🛣", "Узкое место дороги",
                       "Загрузка объясняется параметрами самой дороги (класс, полосы, место). "
                       "Кандидат на расширение / перераспределение полос."),
    "weather": ("🌧", "Чувствительность к погоде",
                "Заметная часть прогноза объясняется погодой — приоритет для уборки/дренажа."),
    "complex": ("⚡", "Комплексная проблема",
                "Время суток и параметры дороги влияют примерно поровну — нужны и режим "
                "светофоров, и инфраструктурные меры."),
}


def classify(st: dict) -> str:
    if st["weather"] >= 0.20:
        return "weather"
    if st["neighbours"] >= max(st["time"], st["road"]):
        return "traffic_light"
    if st["time"] > st["road"] + 0.12:
        return "peak"
    if st["road"] > st["time"] + 0.12:
        return "lane_expansion"
    return "complex"


def main():
    t0 = time.time()
    meta = json.load(open(MODELS / "meta.json"))
    h = min(meta["horizons"])
    reg = xgb.XGBRegressor()
    reg.load_model(MODELS / f"xgb_reg_h{h}.json")
    booster = reg.get_booster()

    with connect() as conn:
        hi = conn.execute("SELECT max(ts) FROM traffic_obs").fetchone()[0]
        lo = max(conn.execute("SELECT min(ts) FROM traffic_obs").fetchone()[0], hi - timedelta(days=DAYS))
        p = load_matrix(conn, lo, hi)
        names = _sql("SELECT edge_id, name, lanes, highway AS hw FROM road_edge WHERE edge_id = ANY(%(ids)s)",
                     conn, params={"ids": p.units["edge_id"].unique().tolist()})
    u = p.units[["edge_id", "dir", "length_m", "lat", "lon"]].merge(names, on="edge_id", how="left")

    # ---- per-unit stats over the period
    S = p.S
    n_obs = (~np.isnan(S)).sum(1)
    with np.errstate(invalid="ignore"):
        u["mean_score"] = np.nanmean(S, 1)
        u["cong_share"] = np.nansum(S >= CONGESTED, 1) / np.maximum(n_obs, 1)
    u["n_obs"] = n_obs
    local_hour = np.asarray((p.slots + pd.Timedelta(hours=ALMATY_UTC_OFFSET)).hour)

    ok = u["name"].notna() & (u["n_obs"] > 0)
    g = u[ok].assign(w=lambda d: d["length_m"] * d["mean_score"], wc=lambda d: d["length_m"] * d["cong_share"])
    streets = g.groupby("name").agg(unit_len=("length_m", "sum"), w=("w", "sum"), wc=("wc", "sum"),
                                     max_score=("mean_score", "max"), units=("edge_id", "size"))
    # street length counts each edge once (units are per direction)
    streets["length_m"] = g.drop_duplicates("edge_id").groupby("name")["length_m"].sum()
    streets = streets[streets["length_m"] >= MIN_STREET_M]
    streets["mean_score"] = streets["w"] / streets["unit_len"]
    streets["cong_share"] = streets["wc"] / streets["unit_len"]
    top = streets.sort_values("mean_score", ascending=False).head(N_STREETS)

    # ---- contributions for the hotspot units
    act = pd.read_csv(MODELS / "active_units.csv")
    act_idx = pd.MultiIndex.from_frame(act.loc[act.active, ["edge_id", "dir"]])
    in_top = u["name"].isin(top.index).values
    mask = in_top & pd.MultiIndex.from_frame(u[["edge_id", "dir"]]).isin(act_idx)
    have = np.flatnonzero((~np.isnan(S)).any(0))
    origins = have
    hist = _slot_hist(S, p.slots)
    X = build_rows(p, h, origins.tolist(), with_target=False, hist=hist, unit_mask=mask)
    contrib = pd.DataFrame(np.zeros((0, len(FEATURES))), columns=FEATURES)
    if len(X):
        X = X.assign(name=u["name"].values[X["unit"].values])
        X = X.sample(frac=1.0, random_state=0)
        X = X[X.groupby("name").cumcount() < MAX_ROWS_PER_STREET]
        c = booster.predict(xgb.DMatrix(X[FEATURES]), pred_contribs=True)[:, :-1]
        contrib = pd.DataFrame(np.abs(c), columns=FEATURES).assign(name=X["name"].values)

    recs = []
    for rank, (name, s) in enumerate(top.iterrows(), 1):
        su = u[u["name"] == name]
        rows = S[su.index.values]
        with np.errstate(invalid="ignore"):
            by_hour = pd.Series(np.nanmean(rows, 0), index=local_hour).groupby(level=0).mean().dropna()
        worst = su.sort_values("mean_score", ascending=False).iloc[0]
        rec = {"rank": rank, "street": name, "mean_score": round(float(s.mean_score), 2),
               "max_score": round(float(s.max_score), 2), "cong_share": round(float(s.cong_share), 3),
               "length_km": round(float(s.length_m) / 1000, 2), "units": int(s.units),
               "peak_hour": int(by_hour.idxmax()) if len(by_hour) else None,
               "lanes": int(su["lanes"].dropna().median()) if su["lanes"].notna().any() else None,
               "highway": su["hw"].mode().iat[0] if su["hw"].notna().any() else None,
               "lat": round(float(worst.lat), 6), "lon": round(float(worst.lon), 6),
               "edge_ids": su.sort_values("mean_score", ascending=False)["edge_id"].head(60).astype(int).tolist()}
        cs = contrib[contrib["name"] == name]
        if len(cs):
            m = cs[FEATURES].mean()
            full = {k: float(m[v].sum()) for k, v in GROUPS.items()}
            tot = sum(full.values()) or 1.0
            struct = {k: v for k, v in full.items() if k not in ("history", "location")}
            stot = sum(struct.values()) or 1.0
            st = {k: v / stot for k, v in struct.items()}
            typ = classify(st)
            top_f = m.drop(GROUPS["history"] + GROUPS["location"]).idxmax()
            rec.update({"type": typ, "icon": TYPES[typ][0], "title": TYPES[typ][1], "description": TYPES[typ][2],
                        "shap_breakdown": {k: round(v / tot, 3) for k, v in full.items()},
                        "structural_breakdown": {k: round(v, 3) for k, v in st.items()},
                        "top_feature": top_f, "top_feature_ru": FEATURE_RU.get(top_f, top_f),
                        "top_feature_share": round(float(m[top_f] / stot), 3), "rows": int(len(cs))})
        else:
            rec.update({"type": None, "icon": "·", "title": "Нет данных модели для улицы",
                        "description": "Участки улицы не попали в обученную модель (мало наблюдений).",
                        "shap_breakdown": None, "structural_breakdown": None})
        rec["priority"] = ("высокий" if s.cong_share >= 0.25 or s.mean_score >= 5 else
                           "средний" if s.cong_share >= 0.08 or s.mean_score >= 3.5 else "низкий")
        recs.append(rec)

    out = {"made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data": {"from": lo.isoformat(), "to": hi.isoformat(), "slots_with_data": int(len(have))},
           "model_horizon": h, "method": "XGBoost pred_contribs (TreeSHAP), mean |SHAP| per group",
           "groups": GROUPS, "types": {k: {"icon": v[0], "title": v[1]} for k, v in TYPES.items()},
           "recommendations": recs}
    json.dump(out, open(MODELS / "recommendations.json", "w"), ensure_ascii=False, indent=1)
    cnt = pd.Series([r["type"] for r in recs]).value_counts().to_dict()
    print(f"recommendations: {len(recs)} streets {cnt} in {time.time() - t0:.0f}s -> {MODELS / 'recommendations.json'}")


if __name__ == "__main__":
    main()
