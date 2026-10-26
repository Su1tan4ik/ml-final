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
|---|---|---|---|
| Naive baseline (last value) | 0.063 | 0.724 | — |
| Random Forest | 0.0623 | 0.778 | 3.40% |
| XGBoost | 0.0627 | 0.779 | 3.41% |
| MLP Neural Net | 0.0811 | 0.543 | 4.55% |

> An earlier version reported R² ≈ 0.99 because the `diff_1` feature included the current target value (data leakage). It was replaced with `diff_2` (difference between the two previous values), and models are now compared with a naive last-value baseline.
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
