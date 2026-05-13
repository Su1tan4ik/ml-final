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
THRESHOLD = json.load(open(MODEL_DIR / 'config.json'))['threshold']

xgb_reg = pickle.load(open(MODEL_DIR / 'xgb_regressor.pkl', 'rb'))
rf_cls  = pickle.load(open(MODEL_DIR / 'xgb_classifier.pkl', 'rb'))
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
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"Loaded: {len(df)} rows, {df['segment_id'].nunique()} segments")

# ── SHAP setup ──
FEATURE_GROUPS = {
    'temporal': ['hour', 'hour_sin', 'hour_cos', 'is_rush_hour', 'is_night', 'day_of_week', 'is_weekend'],
    'weather':  ['weather_temp_c', 'weather_humidity', 'weather_wind_ms', 'weather_precip_1h',
                 'is_low_visibility', 'weather_severity', 'rain_x_rush'],
    'spatial':  ['lat', 'lon', 'is_major_street', 'dist_from_center', 'street_encoded'],
    'history':  ['lag_1', 'lag_2', 'rolling_mean_6', 'rolling_std_6', 'diff_1'],
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
    'diff_1': 'изменение трафика',
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

@app.get("/api/predict")
async def api_predict(lat:float=Query(43.265),lon:float=Query(76.945),
    hour:int=Query(18),day:int=Query(4),temp:float=Query(15),precip:float=Query(0)):
    dists = np.sqrt((segments['lat']-lat)**2+(segments['lon']-lon)**2)
    nearest = segments.iloc[dists.idxmin()]
    nearest_id = nearest['segment_id']
    seg_data = df[df['segment_id']==nearest_id].copy()
    if len(seg_data)==0:
        return {"error":"No data"}
    seg_data['hour_alm'] = pd.to_numeric(seg_data.get('hour',0), errors='coerce')
    available_hours = seg_data['hour_alm'].unique()
    closest_hour = int(min(available_hours, key=lambda h: min(abs(h-hour),24-abs(h-hour))))
    hour_data = seg_data[seg_data['hour_alm'].between(closest_hour-1,closest_hour+1)]
    if len(hour_data)==0: hour_data = seg_data
    mean_score = float(hour_data['traffic_score'].mean())
    max_score = float(hour_data['traffic_score'].max())
    congestion_prob = float((hour_data['traffic_score']>THRESHOLD).mean())*100
    rain_factor = 1.08 if precip > 0 else 1.0
    predicted_score = round(mean_score*rain_factor, 2)
    predicted_max = round(max_score*rain_factor, 2)
    level = 'Low' if predicted_score<=2 else ('Medium' if predicted_score<=3 else 'High')
    speed = round(65-(min(max(predicted_score,1),10)-1)*(65-3)/9, 1)
    risk = 'HIGH RISK' if congestion_prob>30 else ('MODERATE RISK' if congestion_prob>10 else 'LOW RISK')
    return {"predicted_score":predicted_score,"predicted_max":predicted_max,
        "congestion_probability":round(congestion_prob,1),"risk":risk,"level":level,
        "speed_estimate_kmh":speed,"zone_stats":{"mean":round(mean_score,2),"max":round(float(nearest['max_score']),2)},
        "rain_effect":f"+{round((rain_factor-1)*100)}%" if precip>0 else "none",
        "nearest_street":str(nearest.get('street','')),"observations":len(hour_data),
        "input":{"lat":lat,"lon":lon,"hour":hour,"day":day,"temp":temp,"precip":precip}}

@app.get("/api/recommendations")
async def api_recommendations():
    return _compute_shap_recs()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
