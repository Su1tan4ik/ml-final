"""
Almaty Traffic Prediction — FastAPI Backend (Self-Contained)
"""
import json, os, pickle
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Almaty Traffic Prediction API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data.csv"

FEATURES = json.load(open(MODEL_DIR / 'config.json'))['features']
CONFIG = json.load(open(MODEL_DIR / 'config.json'))
THRESHOLD = CONFIG['threshold']
CLS_PROBA_THRESHOLD = float(CONFIG.get('cls_proba_threshold', 0.5))  # tuned on validation in train.py

xgb_reg = pickle.load(open(MODEL_DIR / 'xgb_regressor.pkl', 'rb'))
xgb_cls = pickle.load(open(MODEL_DIR / 'xgb_classifier.pkl', 'rb'))
report  = json.load(open(MODEL_DIR / 'report.json'))
segments = pd.read_csv(MODEL_DIR / 'segments.csv')

streets_path = MODEL_DIR / 'streets.csv'
if streets_path.exists():
    streets_agg = pd.read_csv(streets_path)
else:
    streets_agg = pd.DataFrame()

hourly_lookup_path = MODEL_DIR / 'hourly_lookup.csv'
hourly_lookup = pd.read_csv(hourly_lookup_path) if hourly_lookup_path.exists() else None

df = pd.read_csv(DATA_PATH, low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
# keep only active segments (same selection as train.py → models/segments.csv)
df = df[df['segment_id'].isin(set(segments['segment_id']))].copy()
for col in FEATURES:
    if col not in df.columns:   # derived in train.py (e.g. slot_hist_mean)
        continue
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"Loaded: {len(df)} rows, {df['segment_id'].nunique()} segments")

# ── Per-segment profiles used to build model inputs for /api/predict ──
# Half-hour slot in Almaty time (0..47) and weekday/weekend group.
_ts_alm = df['timestamp'] + pd.Timedelta(hours=5)
df['_slot'] = _ts_alm.dt.hour * 2 + _ts_alm.dt.minute // 30
# slot_hist_mean — same causal definition as train.add_slot_history
df = df.sort_values('timestamp')
df['slot_hist_mean'] = (df.groupby(['segment_id', '_slot'])['traffic_score']
                          .transform(lambda s: s.shift(1).expanding().mean()))
df['slot_hist_mean'] = df['slot_hist_mean'].fillna(df['rolling_mean_6'])
SLOT_PROFILE = (df.groupby(['segment_id', 'is_weekend', '_slot'])['traffic_score']
                  .mean().to_dict())
SLOT_PROFILE_ALL = df.groupby(['segment_id', '_slot'])['traffic_score'].mean().to_dict()
SEG_STATIC = (df.groupby('segment_id')
                .agg(street_encoded=('street_encoded', 'first'),
                     is_major_street=('is_major_street', 'first'),
                     dist_from_center=('dist_from_center', 'first'),
                     seg_mean=('traffic_score', 'mean'))
                .to_dict('index'))
WEATHER_MEDIAN = {c: float(df[c].median()) for c in ['weather_humidity', 'weather_wind_ms']}

# ── SHAP setup ──
FEATURE_GROUPS = {
    'temporal': ['hour', 'hour_sin', 'hour_cos', 'is_rush_hour', 'is_night', 'day_of_week', 'is_weekend'],
    'weather':  ['weather_temp_c', 'weather_humidity', 'weather_wind_ms', 'weather_precip_1h',
                 'is_low_visibility', 'weather_severity', 'rain_x_rush'],
    'spatial':  ['lat', 'lon', 'is_major_street', 'dist_from_center', 'street_encoded'],
    'history':  ['lag_1', 'lag_2', 'rolling_mean_6', 'rolling_std_6', 'diff_1', 'diff_2', 'slot_hist_mean'],
}

FEAT_RU = {
    'hour': 'час суток', 'hour_sin': 'цикличность часа', 'hour_cos': 'цикличность часа',
    'is_rush_hour': 'час пик', 'is_night': 'ночное время', 'day_of_week': 'день недели',
    'is_weekend': 'выходной', 'weather_temp_c': 'температура', 'weather_humidity': 'влажность',
    'weather_wind_ms': 'ветер', 'weather_precip_1h': 'осадки', 'is_low_visibility': 'плохая видимость',
    'weather_severity': 'погодные условия', 'rain_x_rush': 'дождь в час пик',
    'lat': 'широта', 'lon': 'долгота', 'is_major_street': 'тип дороги',
    'dist_from_center': 'удалённость от центра', 'street_encoded': 'улица',
    'lag_1': 'трафик 30 мин назад', 'lag_2': 'трафик 1 час назад',
    'rolling_mean_6': 'среднее за 3 часа', 'rolling_std_6': 'волатильность трафика',
    'diff_1': 'изменение трафика', 'diff_2': 'тренд трафика (lag_1 − lag_2)',
    'slot_hist_mean': 'типичный трафик в это время',
}

GROUP_RU = {
    'temporal': 'Временные паттерны',
    'weather':  'Погодные условия',
    'spatial':  'Пространственные факторы',
    'history':  'Исторический трафик',
}

_shap_cache = None

def _compute_shap_recs():
    global _shap_cache
    if _shap_cache is not None:
        return _shap_cache

    print("Computing SHAP recommendations...")
    explainer = shap.TreeExplainer(xgb_reg)

    # Only analyse segments above the 60th percentile by mean_score
    threshold_seg = float(segments['mean_score'].quantile(0.60))
    candidates = segments[segments['mean_score'] >= threshold_seg]

    df_rush = df[df['is_rush_hour'] == 1].copy()

    feat_indices = {f: i for i, f in enumerate(FEATURES)}
    group_indices = {
        g: [feat_indices[f] for f in feats if f in feat_indices]
        for g, feats in FEATURE_GROUPS.items()
    }

    recs = []
    seen_streets = {}

    for _, seg in candidates.iterrows():
        seg_id  = seg['segment_id']
        mean_s  = float(seg['mean_score'])
        max_s   = float(seg['max_score'])
        street  = str(seg.get('street', ''))
        lat     = float(seg['lat'])
        lon     = float(seg['lon'])

        # prefer rush-hour rows, fall back to all
        sample_df = df_rush[df_rush['segment_id'] == seg_id]
        if len(sample_df) < 5:
            sample_df = df[df['segment_id'] == seg_id]
        if len(sample_df) == 0:
            continue

        X = sample_df[FEATURES].fillna(0).head(40).values
        shap_vals = explainer.shap_values(X)          # (n_samples, n_features)
        mean_abs  = np.abs(shap_vals).mean(axis=0)    # (n_features,)
        total     = mean_abs.sum() or 1.0

        # Full breakdown (shown in UI — honest: lag features will dominate)
        group_pct = {
            g: round(float(mean_abs[idxs].sum() / total * 100), 1)
            for g, idxs in group_indices.items()
        }

        # Structural driver = dominant among actionable groups (excluding history).
        # The model uses lag features for accuracy, but infrastructure decisions
        # must come from the structural signal in temporal/spatial/weather.
        actionable = {g: float(mean_abs[idxs].sum())
                      for g, idxs in group_indices.items() if g != 'history'}
        actionable_total = sum(actionable.values()) or 1.0
        actionable_pct = {g: round(v / actionable_total * 100, 1)
                          for g, v in actionable.items()}

        dominant = max(actionable_pct, key=actionable_pct.get)
        dom_pct  = actionable_pct[dominant]

        # top individual feature (with Russian label)
        top_idx  = int(np.argmax(mean_abs))
        top_feat = FEATURES[top_idx]
        top_ru   = FEAT_RU.get(top_feat, top_feat)
        top_pct  = round(float(mean_abs[top_idx] / total * 100), 1)

        # peak rush hours from raw data
        df_h = df[df['segment_id'] == seg_id].copy()
        df_h['hour_alm'] = (df_h['timestamp'].dt.hour + 5) % 24
        hourly = df_h.groupby('hour_alm')['traffic_score'].mean()
        rush_peaks = sorted([h for h in hourly[hourly > THRESHOLD].index
                             if 7 <= h <= 9 or 17 <= h <= 20])
        peak_str = ', '.join(f'{h}:00' for h in rush_peaks) if rush_peaks else '7:00–9:00, 17:00–19:00'

        # How much structural signal each actionable group contributes (renormalized)
        struct_temporal = actionable_pct.get('temporal', 0)
        struct_spatial  = actionable_pct.get('spatial', 0)
        struct_weather  = actionable_pct.get('weather', 0)

        # ── Classify into 4 types based on structural SHAP ──
        # weather: meaningful even if not #1 (≥20% is significant)
        # complex: temporal and spatial are close (gap ≤12%) and weather is minor
        # traffic_light: temporal clearly leads (gap >12%)
        # lane_expansion: spatial clearly leads (gap >12%)
        gap = struct_temporal - struct_spatial   # positive → temporal leads

        if struct_weather >= 20:
            rec_type = 'weather'
        elif abs(gap) <= 12:
            rec_type = 'complex'
        elif gap > 12:
            rec_type = 'traffic_light'
        else:
            rec_type = 'lane_expansion'

        key = (rec_type, street)
        if key in seen_streets:
            continue
        seen_streets[key] = True

        base = {
            'type': rec_type,
            'driver': dominant,
            'street': street, 'lat': round(lat, 4), 'lon': round(lon, 4),
            'shap_breakdown': group_pct,
            'actionable_pct': actionable_pct,
            'top_feature_ru': top_ru,
            'top_feature_pct': top_pct,
            'mean_score': round(mean_s, 2), 'max_score': round(max_s, 2),
        }

        if rec_type == 'traffic_light':
            extra = min(30, max(8, int(gap * 0.6)))
            recs.append({**base,
                'priority': 'High' if max_s >= 4.0 or struct_temporal >= 45 else 'Medium',
                'title': f'Оптимизация светофора — {street}',
                'description': (
                    f'Структурный SHAP: временной фактор явно доминирует — {struct_temporal}% '
                    f'против пространства {struct_spatial}% и погоды {struct_weather}%. '
                    f'Топ-фича: «{top_ru}». '
                    f'Рекомендуется увеличить зелёную фазу на {extra} сек в часы пик ({peak_str}).'
                ),
                'impact': f'Устранение временного bottleneck ({struct_temporal:.0f}% структурной причины)',
            })

        elif rec_type == 'lane_expansion':
            lanes = 2 if mean_s >= 2.5 else 1
            recs.append({**base,
                'priority': 'High' if mean_s >= 2.5 or struct_spatial >= 45 else 'Medium',
                'title': f'Расширение дороги — {street}',
                'description': (
                    f'Структурный SHAP: пространственный фактор явно доминирует — {struct_spatial}% '
                    f'против времени {struct_temporal}% и погоды {struct_weather}%. '
                    f'Топ-фича: «{top_ru}». '
                    f'Перегрузка геометрическая — рекомендуется добавить {lanes} полос{"ы" if lanes>1 else "у"}.'
                ),
                'impact': f'Устранение геометрического bottleneck ({struct_spatial:.0f}% структурной причины)',
            })

        elif rec_type == 'complex':
            extra = min(20, max(8, int(abs(gap) * 0.5 + 8)))
            lanes = 1
            recs.append({**base,
                'priority': 'High' if max_s >= 4.0 else 'Medium',
                'title': f'Комплексное решение — {street}',
                'description': (
                    f'Структурный SHAP: временной ({struct_temporal}%) и пространственный ({struct_spatial}%) '
                    f'факторы примерно равны — одной меры недостаточно. '
                    f'Топ-фича: «{top_ru}». '
                    f'Рекомендуется: (1) светофор +{extra} сек в пики ({peak_str}), '
                    f'(2) добавить {lanes} полосу движения.'
                ),
                'impact': f'Комбинация мер устраняет оба источника перегрузки',
            })

        else:  # weather
            recs.append({**base,
                'priority': 'Low',
                'title': f'Погодный фактор — {street}',
                'description': (
                    f'Структурный SHAP: погодный фактор значим — {struct_weather}% '
                    f'(время: {struct_temporal}%, пространство: {struct_spatial}%). '
                    f'Топ-фича: «{top_ru}». '
                    'Инфраструктурные изменения малоэффективны. '
                    'Рекомендуется: переменные ограничения скорости при осадках.'
                ),
                'impact': 'Динамические дорожные знаки снизят аварийность при плохой погоде',
            })

    order = {'High': 0, 'Medium': 1, 'Low': 2}
    recs.sort(key=lambda x: (order.get(x['priority'], 2), -x['max_score']))
    _shap_cache = recs
    print(f"SHAP recommendations computed: {len(recs)} total")
    return recs

# ── HTML (embedded) ──
HTML_CONTENT = open(BASE_DIR / "templates" / "index.html", encoding="utf-8").read()
print(f"HTML loaded: {len(HTML_CONTENT)} chars, grid: {'L.rectangle' in HTML_CONTENT}")

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        content=HTML_CONTENT,
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}
    )

@app.get("/api/summary")
async def api_summary():
    return {
        "rows": len(df), "segments": int(df['segment_id'].nunique()),
        "date_start": str(df['timestamp'].min().date()),
        "date_end": str(df['timestamp'].max().date()),
        "mean_score": round(float(df['traffic_score'].mean()), 3),
        "max_score": round(float(df['traffic_score'].max()), 2),
        "congested_pct": round(float((df['traffic_score'] > THRESHOLD).mean() * 100), 1),
    }

@app.get("/api/segments")
async def api_segments():
    return [{"id":r["segment_id"],"lat":round(float(r["lat"]),4),"lon":round(float(r["lon"]),4),
        "street":str(r.get("street","")),"mean":round(float(r["mean_score"]),2),
        "max":round(float(r["max_score"]),2),"cluster":int(r.get("cluster",0))}
        for _,r in segments.iterrows()]

@app.get("/api/hotspots")
async def api_hotspots(top: int = Query(15)):
    if len(streets_agg) > 0:
        data = streets_agg.nlargest(top, 'impact_score' if 'impact_score' in streets_agg.columns else 'max_score')
        return [{"street":str(r.get("street",r.get("clean_street",""))),"lat":round(float(r["lat"]),4),
            "lon":round(float(r["lon"]),4),"mean":round(float(r["mean_score"]),2),
            "max":round(float(r["max_score"]),2),"zones":int(r.get("segments",1))}
            for _,r in data.iterrows()]
    return []

@app.get("/api/hourly")
async def api_hourly():
    df['h'] = (df['timestamp'].dt.hour + 5) % 24
    h = df.groupby('h')['traffic_score'].agg(['mean','max','count']).reset_index()
    return [{"hour":int(r['h']),"mean":round(float(r['mean']),3),"max":round(float(r['max']),2)}
        for _,r in h.iterrows()]

@app.get("/api/daily")
async def api_daily():
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    d = df.groupby('day_of_week')['traffic_score'].mean().reset_index()
    return [{"day":days[int(r['day_of_week'])],"mean":round(float(r['traffic_score']),3)}
        for _,r in d.iterrows()]

@app.get("/api/report")
async def api_report():
    return report

@app.get("/api/features")
async def api_features():
    return report.get('feature_importance', [])

def _typical_score(seg_id, weekend, slot):
    """Typical historical score of a segment in a given half-hour slot."""
    slot %= 48
    v = SLOT_PROFILE.get((seg_id, weekend, slot))
    if v is None:
        v = SLOT_PROFILE_ALL.get((seg_id, slot))
    if v is None:
        v = SEG_STATIC[seg_id]['seg_mean']
    return float(v)

def _build_features(seg, hour, day, temp, precip):
    """Model input for (segment, hour, day, weather). The autoregressive
    features (lags) are not known for a future moment, so they are taken
    from the segment's typical history for the preceding half-hours."""
    seg_id  = seg['segment_id']
    weekend = int(day >= 5)
    slot    = hour * 2
    prev    = [_typical_score(seg_id, weekend, slot - k) for k in range(1, 7)]  # 30 min … 3 h ago
    is_rush = int(hour in (7, 8, 9, 17, 18, 19))
    severity = 3 if precip > 5 else 2 if precip > 1 else 1 if precip > 0 else 0
    st = SEG_STATIC[seg_id]
    row = {
        'hour': hour, 'day_of_week': day, 'is_weekend': weekend,
        'is_rush_hour': is_rush, 'is_night': int(hour in (0, 1, 2, 3, 4, 5, 23)),
        'hour_sin': np.sin(2 * np.pi * hour / 24), 'hour_cos': np.cos(2 * np.pi * hour / 24),
        'weather_temp_c': temp, 'weather_humidity': WEATHER_MEDIAN['weather_humidity'],
        'weather_wind_ms': WEATHER_MEDIAN['weather_wind_ms'], 'weather_precip_1h': precip,
        'is_low_visibility': int(precip > 5), 'weather_severity': severity,
        'rain_x_rush': precip * is_rush,
        'lat': float(seg['lat']), 'lon': float(seg['lon']),
        'street_encoded': st['street_encoded'], 'is_major_street': st['is_major_street'],
        'dist_from_center': st['dist_from_center'],
        'lag_1': prev[0], 'lag_2': prev[1],
        'rolling_mean_6': float(np.mean(prev)), 'rolling_std_6': float(np.std(prev)),
        'diff_1': prev[0] - prev[1], 'diff_2': prev[0] - prev[1],
        'slot_hist_mean': float(SLOT_PROFILE_ALL.get((seg_id, slot % 48), st['seg_mean'])),
    }
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

@app.get("/api/predict")
def api_predict(lat:float=Query(43.265),lon:float=Query(76.945),
    hour:int=Query(18, ge=0, le=23),day:int=Query(4, ge=0, le=6),
    temp:float=Query(15),precip:float=Query(0, ge=0)):
    dists = np.sqrt((segments['lat']-lat)**2+(segments['lon']-lon)**2)
    nearest = segments.loc[dists.idxmin()]
    if nearest['segment_id'] not in SEG_STATIC:
        return {"error": "No data"}

    X = _build_features(nearest, hour, day, temp, precip)
    predicted_score = float(xgb_reg.predict(X)[0])
    congestion_prob = float(xgb_cls.predict_proba(X)[0, 1]) * 100

    # Rain effect measured by the model itself: same input with precip = 0
    rain_effect = "none"
    if precip > 0:
        dry = float(xgb_reg.predict(_build_features(nearest, hour, day, temp, 0.0))[0])
        rain_effect = f"{(predicted_score - dry) / dry * 100:+.1f}%"

    level = 'Low' if predicted_score<=2 else ('Medium' if predicted_score<=3 else 'High')
    speed = round(65-(min(max(predicted_score,1),10)-1)*(65-3)/9, 1)
    th = CLS_PROBA_THRESHOLD * 100
    risk = 'HIGH RISK' if congestion_prob >= th else ('MODERATE RISK' if congestion_prob >= th / 2 else 'LOW RISK')
    return {"predicted_score":round(predicted_score,2),
        "predicted_max":round(float(nearest['max_score']),2),
        "congestion_probability":round(congestion_prob,1),"risk":risk,"level":level,
        "speed_estimate_kmh":speed,
        "zone_stats":{"mean":round(float(nearest['mean_score']),2),"max":round(float(nearest['max_score']),2)},
        "rain_effect":rain_effect,
        "nearest_street":str(nearest.get('street','')),
        "model":{"regressor":"XGBoost","classifier":"XGBoost","decision_threshold":round(th,1),
                 "lags":"typical history of this zone for the preceding 3 h"},
        "input":{"lat":lat,"lon":lon,"hour":hour,"day":day,"temp":temp,"precip":precip}}

@app.get("/api/recommendations")
async def api_recommendations():
    return _compute_shap_recs()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
