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
| Naive baseline (last value) | 0.0666 | 0.707 | 3.58% |
| Random Forest | 0.0655 | 0.762 | 3.57% |
| XGBoost | 0.0648 | 0.766 | 3.53% |
| MLP Neural Net | 0.0788 | 0.678 | 4.39% |

> An earlier version reported R² ≈ 0.99 because the `diff_1` feature included the current target value (data leakage). It was replaced with `diff_2` (difference between the two previous values). A second leak was fixed later: missing lags were filled with the segment mean computed *including the current row*; they are now filled with the mean of the segment's **past** values only (`scripts/feature_engineering.py --rebuild-lags`). Models are compared with a naive last-value baseline: XGBoost improves R² from 0.707 to 0.766 and MAE by ~3%.

**Classification (congestion detection, score > 2.0, ~10% positive class):**

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| XGBoost | 90.8% | 0.60 | 0.45 | 0.91 |
| Random Forest | 92.9% | 0.65 | 0.52 | 0.86 |

Accuracy is inflated by class imbalance; F1/precision are the meaningful metrics. `/api/predict` uses the XGBoost regressor + classifier.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/summary` | Dataset statistics |
| `GET /api/segments` | All traffic zones for map |
| `GET /api/hourly` | Hourly traffic patterns |
| `GET /api/predict` | Real-time prediction |
| `GET /api/report` | Model evaluation metrics |
