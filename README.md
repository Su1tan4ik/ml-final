---
title: Almaty Traffic Prediction
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Almaty Traffic Congestion Prediction System

**AI-Powered Traffic Congestion Prediction and Infrastructure Optimization**

Students: Demessinov Rakhymzhan (23B031273), Kuantayev Sultan (23B031521)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates plots + saves models)
python train.py

# 3. Run web application
python app.py
# Open http://localhost:8000
```

## Project Structure

```
├── train.py              # ML pipeline: train + evaluate + generate plots
├── app.py                # FastAPI web server
├── data.csv              # Preprocessed traffic dataset (24K rows, 24 features)
├── requirements.txt      # Python dependencies
├── models/               # Trained models + evaluation report
│   ├── xgb_regressor.pkl
│   ├── rf_regressor.pkl
│   ├── xgb_classifier.pkl
│   ├── report.json
│   └── segments.csv
├── static/plots/         # 12 visualization plots
│   ├── 01_eda_overview.png
│   ├── 02_temporal.png
│   ├── ...
│   └── 12_weather.png
├── templates/
│   └── index.html        # Frontend dashboard
├── scraper/              # Data collection scripts
│   ├── scraper.py
│   ├── config.py
│   ├── tiles.py
│   ├── weather.py
│   └── geocoder.py
└── README.md
```

## System Architecture

```
Data Collection → Preprocessing → Feature Engineering → ML Models → API → Dashboard
(Yandex tiles)    (HSV colors)     (24 features)        (XGBoost)    (FastAPI) (Leaflet+Chart.js)
```

## Results

| Model | MAE | R² | MAPE |
|-------|-----|-----|------|
| Random Forest | 0.0066 | 0.990 | 0.30% |
| XGBoost | 0.0075 | 0.991 | 0.36% |
| MLP Neural Net | 0.0214 | 0.994 | 1.34% |

**Classification (Congestion Detection):** XGBoost Accuracy=99.4%, F1=0.975

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/summary` | Dataset statistics |
| `GET /api/segments` | All traffic zones for map |
| `GET /api/hourly` | Hourly traffic patterns |
| `GET /api/predict` | Real-time prediction |
| `GET /api/report` | Model evaluation metrics |
