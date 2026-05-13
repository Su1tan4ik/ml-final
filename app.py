"""
Almaty Traffic Prediction — FastAPI Backend (Self-Contained)
"""
import json, os, pickle
import numpy as np
import pandas as pd
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
