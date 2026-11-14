"""
===========================================================================
Almaty Traffic Congestion Prediction — Train & Evaluate Pipeline
===========================================================================
Students: Demessinov Rakhymzhan (23B031273), Kuantayev Sultan (23B031521)

This script:
  1. Loads preprocessed traffic data
  2. Trains 3 regression models (+ naive baseline) + 2 classification models
  3. Evaluates all models with proper metrics
  4. Runs SHAP explainability analysis
  5. Performs K-Means clustering
  6. Runs counterfactual simulations
  7. Generates 12 publication-quality plots
  8. Saves trained models for the API server
===========================================================================
"""

import warnings; warnings.filterwarnings('ignore')
import os, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier
)
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    silhouette_score
)
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH   = os.environ.get('DATA_PATH', 'data.csv')
MODEL_DIR   = os.environ.get('MODEL_DIR', 'models')
PLOT_DIR    = os.environ.get('PLOT_DIR', 'static/plots')
CONGESTION_THRESHOLD = 2.0

FEATURES = [
    'hour','day_of_week','is_weekend','is_rush_hour','is_night',
    'hour_sin','hour_cos',
    'weather_temp_c','weather_humidity','weather_wind_ms',
    'weather_precip_1h','is_low_visibility','weather_severity','rain_x_rush',
    'lat','lon','street_encoded','is_major_street','dist_from_center',
    'lag_1','lag_2','rolling_mean_6','rolling_std_6','diff_2',
    'slot_hist_mean',
]
FREE_FLOW = 1.5
INACTIVE_FREE_SHARE = 0.99   # segment is "inactive" if ≥99% of its train observations are free-flow
MIN_SEGMENT_OBS     = 50
TARGET = 'traffic_score'

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.facecolor': 'white',
})

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    d = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(np.where(d == 0, 0, 2 * np.abs(y_true - y_pred) / d)) * 100
    return {
        'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4),
        'MAPE': round(mape, 2), 'sMAPE': round(smape, 2),
    }

def save_model(obj, name):
    path = os.path.join(MODEL_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'    saved → {path}')


# =====================================================================
#  1. LOAD & SPLIT
# =====================================================================
def add_slot_history(df):
    """slot_hist_mean: mean score of the same segment in the same half-hour slot
    (Almaty time) on PREVIOUS observations only (expanding mean, shifted by one)
    → causal, no target leakage."""
    ts = df['timestamp'] + pd.Timedelta(hours=5)
    df['slot'] = ts.dt.hour * 2 + ts.dt.minute // 30
    df = df.sort_values('timestamp')
    df['slot_hist_mean'] = (df.groupby(['segment_id', 'slot'])['traffic_score']
                              .transform(lambda s: s.shift(1).expanding().mean()))
    df['slot_hist_mean'] = df['slot_hist_mean'].fillna(df['rolling_mean_6'])
    return df


def load_and_split(path, test_ratio=0.2):
    print('═'*60)
    print('  1. LOADING DATA')
    print('═'*60)

    df = pd.read_csv(path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['traffic_score'] = pd.to_numeric(df['traffic_score'], errors='coerce')

    df = df.sort_values('timestamp').reset_index(drop=True)
    cut_time = df['timestamp'].iloc[int(len(df) * (1 - test_ratio))]

    # Drop inactive segments (parks, mountains, fringe tiles): decided on the
    # TRAIN period only so the test set does not influence the selection.
    tr = df[df['timestamp'] < cut_time]
    free_share = tr.groupby('segment_id')['traffic_score'].apply(lambda s: (s <= FREE_FLOW + 1e-4).mean())
    n_obs = tr.groupby('segment_id').size()
    active = free_share[(free_share < INACTIVE_FREE_SHARE) & (n_obs >= MIN_SEGMENT_OBS)].index
    n_all = df['segment_id'].nunique()
    df = df[df['segment_id'].isin(active)].copy()
    print(f'  Segments   : {len(active)} active of {n_all} '
          f'(dropped: ≥{INACTIVE_FREE_SHARE:.0%} free-flow or <{MIN_SEGMENT_OBS} obs in train period)')

    df = add_slot_history(df).sort_values('timestamp').reset_index(drop=True)
    train = df[df['timestamp'] < cut_time].copy()
    test  = df[df['timestamp'] >= cut_time].copy()

    print(f'  Total rows : {len(df):,}')
    print(f'  Features   : {len(FEATURES)}')
    print(f'  Train      : {len(train):,}  ({train["timestamp"].min().date()} → {train["timestamp"].max().date()})')
    print(f'  Test       : {len(test):,}  ({test["timestamp"].min().date()} → {test["timestamp"].max().date()})')
    return df, train, test, {'active': len(active), 'total': n_all}


# =====================================================================
#  2. TRAIN REGRESSION MODELS
# =====================================================================
def train_regression(train, test):
    print('\n' + '═'*60)
    print('  2. TRAINING REGRESSION MODELS')
    print('═'*60)

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES],  test[TARGET]

    models  = {}
    results = {}
    preds   = {}

    # 2.0  Naive baseline: "traffic in 30 min = traffic now" (lag_1)
    baseline = regression_metrics(y_te.values, test['lag_1'].values)
    print(f'\n  ▸ Naive baseline (last value): {baseline}')

    # 2a  Random Forest
    print('\n  ▸ Random Forest')
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p = rf.predict(X_te)
    results['RandomForest'] = regression_metrics(y_te.values, p)
    models['RandomForest'] = rf; preds['RandomForest'] = p
    print(f'    {results["RandomForest"]}')
    save_model(rf, 'rf_regressor')

    # 2b  XGBoost (or GradientBoosting)
    if HAS_XGB:
        print('\n  ▸ XGBoost')
        xgb_m = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, n_jobs=-1)
        xgb_m.fit(X_tr, y_tr)
        p = xgb_m.predict(X_te)
        results['XGBoost'] = regression_metrics(y_te.values, p)
        models['XGBoost'] = xgb_m; preds['XGBoost'] = p
        print(f'    {results["XGBoost"]}')
        save_model(xgb_m, 'xgb_regressor')
    else:
        print('\n  ▸ GradientBoosting (XGBoost unavailable)')
        gb = GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42)
        gb.fit(X_tr, y_tr)
        p = gb.predict(X_te)
        results['GradientBoosting'] = regression_metrics(y_te.values, p)
        models['GradientBoosting'] = gb; preds['GradientBoosting'] = p
        print(f'    {results["GradientBoosting"]}')
        save_model(gb, 'gb_regressor')

    # 2c  MLP Neural Network
    print('\n  ▸ MLP Neural Network')
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', max_iter=500, early_stopping=True,
        validation_fraction=0.15, learning_rate='adaptive',
        random_state=42)
    mlp.fit(X_tr_s, y_tr)
    p = mlp.predict(X_te_s)
    results['MLP'] = regression_metrics(y_te.values, p)
    models['MLP'] = mlp; preds['MLP'] = p
    print(f'    {results["MLP"]}')
    save_model(mlp, 'mlp_regressor')
    save_model(scaler, 'scaler')

    best = min(results, key=lambda k: results[k]['MAE'])
    print(f'\n  ★ Best regression model: {best}  (MAE = {results[best]["MAE"]})')

    return models, results, preds, y_te, baseline


# =====================================================================
#  3. TRAIN CLASSIFICATION MODELS
# =====================================================================
def _cls_metrics(y, p, threshold=None):
    d = {
        'accuracy': round(accuracy_score(y, p), 4),
        'f1': round(f1_score(y, p), 4),
        'precision': round(precision_score(y, p, zero_division=0), 4),
        'recall': round(recall_score(y, p, zero_division=0), 4),
    }
    if threshold is not None:
        d['threshold'] = round(float(threshold), 3)
    return d


def _tune_threshold(make_model, X_tr, y_tr, val_ratio=0.15):
    """Pick the probability threshold that maximises F1 on the last 15% of the
    TRAIN period (time-ordered validation), then refit on the full train set."""
    cut = int(len(X_tr) * (1 - val_ratio))
    m = make_model().fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
    pv = m.predict_proba(X_tr.iloc[cut:])[:, 1]
    ths = np.arange(0.10, 0.951, 0.025)
    f1s = [f1_score(y_tr.iloc[cut:], pv >= t) for t in ths]
    th = float(ths[int(np.argmax(f1s))])
    return make_model().fit(X_tr, y_tr), th


def train_classification(train, test):
    print('\n' + '═'*60)
    print('  3. TRAINING CLASSIFICATION MODELS')
    print('═'*60)
    print(f'  Threshold: score > {CONGESTION_THRESHOLD} → Congested')

    X_tr, X_te = train[FEATURES], test[FEATURES]
    y_tr = (train[TARGET] > CONGESTION_THRESHOLD).astype(int)
    y_te = (test[TARGET]  > CONGESTION_THRESHOLD).astype(int)

    print(f'  Train congested: {y_tr.sum():,} ({y_tr.mean()*100:.1f}%)')
    print(f'  Test  congested: {y_te.sum():,} ({y_te.mean()*100:.1f}%)')

    cls_models  = {}
    cls_results = {}
    cls_preds   = {}

    # Naive baseline: "congested in 30 min if congested now"
    naive = _cls_metrics(y_te, (test['lag_1'] > CONGESTION_THRESHOLD).astype(int))
    print(f'\n  ▸ Naive baseline (lag_1 > {CONGESTION_THRESHOLD}): {naive}')

    # 3a  XGBoost Classifier — no class re-weighting (it inflated recall at the
    #     cost of precision); decision threshold tuned on validation instead.
    if HAS_XGB:
        print('\n  ▸ XGBoost Classifier')
        xgb_c, th = _tune_threshold(
            lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                      random_state=42, n_jobs=-1),
            X_tr, y_tr)
        p = (xgb_c.predict_proba(X_te)[:, 1] >= th).astype(int)
        cls_results['XGBoost'] = _cls_metrics(y_te, p, th)
        cls_models['XGBoost'] = xgb_c; cls_preds['XGBoost'] = p
        print(f'    {cls_results["XGBoost"]}')
        save_model(xgb_c, 'xgb_classifier')
        print(classification_report(y_te, p, target_names=['Free','Congested'], zero_division=0))

    # 3b  Random Forest Classifier
    print('  ▸ Random Forest Classifier')
    rf_c, th = _tune_threshold(
        lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
                                       random_state=42, n_jobs=-1),
        X_tr, y_tr)
    p = (rf_c.predict_proba(X_te)[:, 1] >= th).astype(int)
    cls_results['RF'] = _cls_metrics(y_te, p, th)
    cls_models['RF'] = rf_c; cls_preds['RF'] = p
    print(f'    {cls_results["RF"]}')
    save_model(rf_c, 'rf_classifier')

    return cls_models, cls_results, cls_preds, y_te, naive


# =====================================================================
#  4. EXPLAINABILITY (SHAP + Feature Importance)
# =====================================================================
def explain_models(models, X_test):
    print('\n' + '═'*60)
    print('  4. MODEL EXPLAINABILITY')
    print('═'*60)

    # Feature importance from tree models
    best_key = 'XGBoost' if 'XGBoost' in models else 'GradientBoosting'
    if best_key not in models:
        best_key = 'RandomForest'
    best = models[best_key]

    fi = pd.DataFrame({
        'feature': FEATURES,
        'importance': best.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f'\n  Top 10 features ({best_key}):')
    for _, r in fi.head(10).iterrows():
        bar = '█' * int(r['importance'] * 80)
        print(f'    {r["feature"]:<22s} {r["importance"]:.4f}  {bar}')

    # SHAP
    shap_values = None
    X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    if HAS_SHAP:
        print('\n  Running SHAP analysis...')
        try:
            # Try XGBoost first
            explainer = shap.TreeExplainer(best)
            shap_values = explainer.shap_values(X_sample)
            print('  SHAP values computed (XGBoost).')
        except (ValueError, TypeError, Exception) as e:
            print(f'  SHAP+XGBoost failed ({e.__class__.__name__}), trying RandomForest...')
            try:
                rf_model = models.get('RandomForest')
                if rf_model is not None:
                    explainer = shap.TreeExplainer(rf_model)
                    shap_values = explainer.shap_values(X_sample)
                    print('  SHAP values computed (RandomForest).')
                else:
                    raise RuntimeError('No RF model')
            except Exception as e2:
                print(f'  SHAP failed completely: {e2}')
                print('  Using permutation importance as fallback.')
                from sklearn.inspection import permutation_importance
                pi = permutation_importance(best, X_sample, X_sample.iloc[:, 0],
                                            n_repeats=5, random_state=42, n_jobs=-1)
                # Create pseudo-SHAP from permutation importance
                shap_values = None

    return fi, shap_values, X_sample


# =====================================================================
#  5. CLUSTERING
# =====================================================================
def run_clustering(df):
    print('\n' + '═'*60)
    print('  5. TRAFFIC PATTERN CLUSTERING')
    print('═'*60)

    agg = df.groupby('segment_id').agg(
        lat=('lat','first'), lon=('lon','first'),
        street=('street_corrected','first') if 'street_corrected' in df.columns else ('street_name','first'),
        mean_score=('traffic_score','mean'),
        max_score=('traffic_score','max'),
        std_score=('traffic_score','std'),
    ).reset_index()
    agg['std_score'] = agg['std_score'].fillna(0)

    cl_features = ['mean_score','std_score','max_score','lat','lon']
    X = StandardScaler().fit_transform(agg[cl_features])

    sils = {}
    for k in range(2, 7):
        labels = KMeans(k, random_state=42, n_init=10).fit_predict(X)
        sils[k] = round(silhouette_score(X, labels), 3)
        print(f'  K={k}: silhouette = {sils[k]}')

    best_k = max(sils, key=sils.get)
    km = KMeans(best_k, random_state=42, n_init=10)
    agg['cluster'] = km.fit_predict(X)

    print(f'\n  ★ Best K = {best_k}  (silhouette = {sils[best_k]})')
    for c in sorted(agg['cluster'].unique()):
        sub = agg[agg['cluster'] == c]
        print(f'    Cluster {c}: {len(sub)} segments, '
              f'mean={sub["mean_score"].mean():.2f}, max={sub["max_score"].mean():.2f}')

    return agg, sils, best_k


# =====================================================================
#  6. COUNTERFACTUAL SIMULATIONS
# =====================================================================
def _set_hour(X, hour):
    X['hour'] = hour
    X['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    X['is_rush_hour'] = int(hour in (7, 8, 9, 17, 18, 19))
    X['is_night'] = int(hour in (0, 1, 2, 3, 4, 5, 23))
    X['rain_x_rush'] = X['weather_precip_1h'] * X['is_rush_hour']


def run_counterfactuals(model, test_df):
    """What-if analysis on the trained model (associational, not causal).
    Only exogenous inputs are changed (time, weather); the current traffic
    state (lags) is kept fixed, so the result is the model's estimate of the
    *direct* effect of that factor. A separate 'persistence' scenario
    measures how much of a congestion shock carries over to the next 30 min."""
    print('\n' + '═'*60)
    print('  6. COUNTERFACTUAL SIMULATIONS')
    print('═'*60)

    busy = test_df[test_df[TARGET] > 1.8]
    rush = busy[busy['is_rush_hour'] == 1]

    def scenario_rush_to_offpeak(X):  _set_hour(X, 14)
    def scenario_clear(X):
        X['weather_precip_1h'] = 0; X['is_low_visibility'] = 0
        X['weather_severity'] = 0;  X['rain_x_rush'] = 0
    def scenario_heavy_rain(X):
        X['weather_precip_1h'] = 5.0; X['weather_severity'] = 2
        X['rain_x_rush'] = 5.0 * X['is_rush_hour']
    def scenario_shock(X):
        for c in ['lag_1', 'lag_2', 'rolling_mean_6']:
            X[c] = X[c] * 1.2

    scenarios = [
        ('Peak spreading: rush trips at 14:00', 'Rush-hour observations re-timed to 14:00 (flexible hours)', rush, scenario_rush_to_offpeak),
        ('Clear weather', 'Precipitation and low visibility removed', busy, scenario_clear),
        ('Heavy rain (5 mm/h)', 'Same moments under heavy rain', busy, scenario_heavy_rain),
        ('Persistence: +20% congestion now', 'Share of a congestion shock that persists 30 min later', busy, scenario_shock),
    ]

    results = []
    for name, desc, subset, fn in scenarios:
        X = subset[FEATURES].fillna(0).copy()
        if len(X) == 0:
            continue
        base = model.predict(X).mean()
        X_mod = X.copy(); fn(X_mod)
        mod = model.predict(X_mod).mean()
        change = (mod - base) / base * 100
        results.append({
            'name': name, 'description': desc, 'rows': int(len(X)),
            'baseline_score': round(float(base), 3),
            'modified_score': round(float(mod), 3),
            'change_pct': round(float(change), 2),
        })
        print(f'  {name}: {change:+.2f}%  (n={len(X):,})')

    return results


def build_streets(df):
    """Hotspot table: one row per street, impact = total excess congestion
    (sum over the street's zones of mean score above free-flow)."""
    seg = df.groupby('segment_id').agg(
        street=('street_corrected', 'first'), lat=('lat', 'first'), lon=('lon', 'first'),
        mean_score=('traffic_score', 'mean'), max_score=('traffic_score', 'max')).reset_index()
    seg = seg[seg['street'].fillna('').str.strip() != '']
    # Nominatim sometimes returns a district instead of a street — not a hotspot street
    seg = seg[~seg['street'].str.contains('район', case=False)]
    seg['excess'] = (seg['mean_score'] - FREE_FLOW).clip(lower=0)
    st = seg.groupby('street').agg(
        lat=('lat', 'mean'), lon=('lon', 'mean'),
        mean_score=('mean_score', 'mean'), max_score=('max_score', 'max'),
        segments=('segment_id', 'count'), impact_score=('excess', 'sum')).reset_index()
    st['impact_score'] = (st['impact_score'] * 10).round(3)
    return st.sort_values('impact_score', ascending=False)


def _tile_bounds(seg_id):
    """Exact lat/lon bounds of a slippy-map tile id like 'z15_x23376_y11999'."""
    z, x, y = (int(p[1:]) for p in seg_id.split('_'))
    n = 2 ** z
    lon1, lon2 = x / n * 360 - 180, (x + 1) / n * 360 - 180
    lat = lambda yy: np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * yy / n))))
    return [[round(lat(y + 1), 5), round(lon1, 5)], [round(lat(y), 5), round(lon2, 5)]], z


def build_map_data(df):
    """static/map_data.json for the Leaflet map: one rectangle per active
    segment (its real tile), mean score per Almaty hour."""
    df = df.copy()
    df['hour_alm'] = (df['timestamp'].dt.hour + 5) % 24
    seg = df.groupby('segment_id').agg(lat=('lat', 'first'), lon=('lon', 'first'),
                                       mean=('traffic_score', 'mean'), max=('traffic_score', 'max'))
    hourly = df.groupby(['segment_id', 'hour_alm'])['traffic_score'].mean().round(2)
    grid, hourly_out = [], {}
    for sid, r in seg.iterrows():
        bounds, z = _tile_bounds(sid)
        grid.append({'id': sid, 'bounds': bounds, 'zoom': z,
                     'lat': round(float(r['lat']), 4), 'lon': round(float(r['lon']), 4),
                     'mean': round(float(r['mean']), 2), 'max': round(float(r['max']), 2),
                     'has_data': True})
        hourly_out[sid] = {str(int(h)): float(v) for h, v in hourly.loc[sid].items()}
    grid.sort(key=lambda c: c['zoom'])   # draw coarse z14 tiles first, finer z15 on top
    return {
        'grid': grid, 'hourly': hourly_out,
        'available_hours': sorted(int(h) for h in df['hour_alm'].unique()),
        'data_info': {'total_rows': int(len(df)), 'segments': int(len(seg)),
                      'date_start': str(df['timestamp'].min().date()),
                      'date_end': str(df['timestamp'].max().date())},
    }


# =====================================================================
#  7. GENERATE ALL PLOTS
# =====================================================================
def generate_plots(reg_results, reg_preds, y_test_reg,
                   cls_preds, y_test_cls,
                   fi, shap_values, X_shap_sample,
                   cluster_agg, cluster_sils, best_k,
                   cf_results, df):
    print('\n' + '═'*60)
    print('  7. GENERATING PLOTS')
    print('═'*60)

    # ── Plot 1: EDA Overview ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    y_all = df['traffic_score']
    axes[0].hist(y_all, bins=60, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[0].axvline(y_all.mean(), color='red', ls='--', label=f'Mean={y_all.mean():.2f}')
    axes[0].axvline(CONGESTION_THRESHOLD, color='orange', ls='--', label=f'Threshold={CONGESTION_THRESHOLD}')
    axes[0].set_title('Traffic Score Distribution'); axes[0].set_xlabel('Score'); axes[0].legend()

    lv = df['congestion_level'].value_counts()
    c_map = {'Low':'#4CAF50','Medium':'#FF9800','High':'#F44336','Unknown':'#9E9E9E'}
    axes[1].pie(lv, labels=lv.index, colors=[c_map.get(l,'#9E9E9E') for l in lv.index],
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Congestion Level Distribution')

    if 'street_corrected' in df.columns:
        top = df.groupby('street_corrected')['traffic_score'].mean().nlargest(10)
    else:
        top = df.groupby('street_name')['traffic_score'].mean().nlargest(10)
    axes[2].barh(top.index, top.values, color='#FF5722', alpha=0.8)
    axes[2].set_title('Top 10 Zones by Mean Score')
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/01_eda_overview.png', bbox_inches='tight'); plt.close()
    print('  ✓ 01_eda_overview.png')

    # ── Plot 2: Temporal Patterns ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    df['hour_alm'] = (df['timestamp'].dt.hour + 5) % 24
    hourly = df.groupby('hour_alm')['traffic_score'].agg(['mean','std']).reset_index()
    axes[0,0].plot(hourly['hour_alm'], hourly['mean'], 'o-', color='#2196F3', lw=2)
    axes[0,0].fill_between(hourly['hour_alm'], hourly['mean']-hourly['std'],
                            hourly['mean']+hourly['std'], alpha=0.2, color='#2196F3')
    axes[0,0].axvspan(7,9,alpha=0.1,color='red',label='Morning rush')
    axes[0,0].axvspan(17,20,alpha=0.1,color='orange',label='Evening rush')
    axes[0,0].set_title('Traffic Score by Hour (Almaty)'); axes[0,0].legend()

    dow = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    daily = df.groupby('day_of_week')['traffic_score'].mean()
    axes[0,1].bar(range(7), [daily.get(i,0) for i in range(7)],
                  color=['#F44336' if i<5 else '#4CAF50' for i in range(7)])
    axes[0,1].set_xticks(range(7)); axes[0,1].set_xticklabels(dow)
    axes[0,1].set_title('Score by Day of Week')

    pivot = df.pivot_table(values='traffic_score', index='day_of_week',
                           columns='hour_alm', aggfunc='mean')
    sns.heatmap(pivot, ax=axes[1,0], cmap='YlOrRd', yticklabels=dow[:len(pivot)])
    axes[1,0].set_title('Heatmap: Day × Hour')

    ts_agg = df.groupby(df['timestamp'].dt.floor('h'))['traffic_score'].mean()
    axes[1,1].plot(ts_agg.index, ts_agg.values, lw=0.8, color='#2196F3')
    axes[1,1].set_title('City Average Over Time'); axes[1,1].tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/02_temporal.png', bbox_inches='tight'); plt.close()
    print('  ✓ 02_temporal.png')

    # ── Plot 3: Spatial Map ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    seg = df.groupby('segment_id').agg(
        lat=('lat','first'), lon=('lon','first'),
        mean_s=('traffic_score','mean'), max_s=('traffic_score','max')).reset_index()
    sc = axes[0].scatter(seg['lon'], seg['lat'], c=seg['mean_s'], cmap='YlOrRd', s=30, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label='Mean Score'); axes[0].set_title('Traffic Map (Mean)')
    sc2 = axes[1].scatter(seg['lon'], seg['lat'], c=seg['max_s'], cmap='YlOrRd', s=30, alpha=0.7)
    plt.colorbar(sc2, ax=axes[1], label='Max Score'); axes[1].set_title('Traffic Map (Max)')
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/03_spatial.png', bbox_inches='tight'); plt.close()
    print('  ✓ 03_spatial.png')

    # ── Plot 4: Model Comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(reg_results.keys())
    x = np.arange(len(names)); w = 0.22
    for i, met in enumerate(['MAE','RMSE','sMAPE']):
        axes[0].bar(x+i*w, [reg_results[m][met] for m in names], w, label=met, alpha=0.8)
    axes[0].set_xticks(x+w); axes[0].set_xticklabels(names, rotation=10)
    axes[0].set_title('Regression Model Comparison'); axes[0].legend(); axes[0].set_ylabel('Error')

    r2_vals = [reg_results[m]['R2'] for m in names]
    bars = axes[1].bar(names, r2_vals, color=['#2196F3','#FF5722','#4CAF50'][:len(names)], alpha=0.8)
    axes[1].set_title('R² Score (higher = better)'); axes[1].set_ylabel('R²')
    for b, v in zip(bars, r2_vals):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.4f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/04_model_comparison.png', bbox_inches='tight'); plt.close()
    print('  ✓ 04_model_comparison.png')

    # ── Plot 5: Predictions vs Actual ──
    fig, axes = plt.subplots(1, len(reg_preds), figsize=(6*len(reg_preds), 5))
    if len(reg_preds) == 1: axes = [axes]
    for ax, (name, pred) in zip(axes, reg_preds.items()):
        ax.scatter(y_test_reg, pred, alpha=0.15, s=8, color='#2196F3')
        lims = [min(y_test_reg.min(), pred.min()), max(y_test_reg.max(), pred.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        ax.set_title(f'{name} (R²={reg_results[name]["R2"]:.4f})')
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.legend()
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/05_pred_vs_actual.png', bbox_inches='tight'); plt.close()
    print('  ✓ 05_pred_vs_actual.png')

    # ── Plot 6: Error Distribution ──
    fig, axes = plt.subplots(1, len(reg_preds), figsize=(6*len(reg_preds), 5))
    if len(reg_preds) == 1: axes = [axes]
    for ax, (name, pred) in zip(axes, reg_preds.items()):
        err = y_test_reg.values - pred
        ax.hist(err, bins=50, color='#FF5722', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', lw=1)
        ax.set_title(f'{name}  (μ={err.mean():.4f}, σ={err.std():.4f})')
        ax.set_xlabel('Error')
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/06_error_dist.png', bbox_inches='tight'); plt.close()
    print('  ✓ 06_error_dist.png')

    # ── Plot 7: Confusion Matrix ──
    fig, axes = plt.subplots(1, len(cls_preds), figsize=(6*len(cls_preds), 5))
    if len(cls_preds) == 1: axes = [axes]
    for ax, (name, pred) in zip(axes, cls_preds.items()):
        cm = confusion_matrix(y_test_cls, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Free','Congested'], yticklabels=['Free','Congested'])
        ax.set_title(f'{name} Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/07_confusion.png', bbox_inches='tight'); plt.close()
    print('  ✓ 07_confusion.png')

    # ── Plot 8: Feature Importance ──
    fig, ax = plt.subplots(figsize=(10, 7))
    t = fi.head(15)
    ax.barh(t['feature'][::-1], t['importance'][::-1], color='#FF5722', alpha=0.8)
    ax.set_title('Feature Importance (Tree-based)')
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/08_feature_importance.png', bbox_inches='tight'); plt.close()
    print('  ✓ 08_feature_importance.png')

    # ── Plot 9: SHAP ──
    if shap_values is not None and HAS_SHAP:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap_sample, show=False, max_display=15)
        plt.title('SHAP Feature Impact on Traffic Score')
        plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/09_shap.png', bbox_inches='tight', dpi=150); plt.close()
        print('  ✓ 09_shap.png')
    else:
        print('  ⊘ 09_shap.png  (SHAP not available)')

    # ── Plot 10: Cluster Map ──
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = ['#4CAF50','#FF9800','#F44336','#2196F3','#9C27B0','#607D8B']
    for c in sorted(cluster_agg['cluster'].unique()):
        sub = cluster_agg[cluster_agg['cluster'] == c]
        ax.scatter(sub['lon'], sub['lat'], s=50, alpha=0.7,
                   color=palette[c % len(palette)],
                   label=f'Cluster {c} ({len(sub)} seg, μ={sub["mean_score"].mean():.2f})')
    ax.set_title(f'K-Means Clusters (K={best_k}, silhouette={cluster_sils[best_k]})')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/10_clusters.png', bbox_inches='tight'); plt.close()
    print('  ✓ 10_clusters.png')

    # ── Plot 11: Counterfactual ──
    fig, ax = plt.subplots(figsize=(12, 5))
    cf_names = [r['name'] for r in cf_results]
    cf_vals  = [r['change_pct'] for r in cf_results]
    colors_cf = ['#4CAF50' if v < 0 else '#F44336' for v in cf_vals]
    bars = ax.barh(cf_names, cf_vals, color=colors_cf, alpha=0.8)
    ax.set_title('What-if: change in predicted traffic score (model-based, not causal)')
    ax.set_xlabel('Change in predicted score (%)  — green = less congestion'); ax.axvline(0, color='black', lw=0.8)
    for b, v in zip(bars, cf_vals):
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height()/2,
                f'{v:+.2f}%', va='center', fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/11_counterfactual.png', bbox_inches='tight'); plt.close()
    print('  ✓ 11_counterfactual.png')

    # ── Plot 12: Weather Impact ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    wt = df[df['weather_temp_c'] > -10]
    axes[0].scatter(wt['weather_temp_c'], wt['traffic_score'], alpha=0.03, s=5, color='#2196F3')
    axes[0].set_title('Temperature vs Traffic'); axes[0].set_xlabel('°C'); axes[0].set_ylabel('Score')
    rain = df[df['weather_precip_1h'] > 0]['traffic_score'].mean()
    no_rain = df[df['weather_precip_1h'] == 0]['traffic_score'].mean()
    axes[1].bar(['No Rain','Rain'], [no_rain, rain], color=['#4CAF50','#F44336'], alpha=0.8)
    axes[1].set_title(f'Rain Impact ({((rain-no_rain)/no_rain*100):+.1f}%)'); axes[1].set_ylabel('Mean Score')
    plt.tight_layout(); plt.savefig(f'{PLOT_DIR}/12_weather.png', bbox_inches='tight'); plt.close()
    print('  ✓ 12_weather.png')


# =====================================================================
#  MAIN
# =====================================================================
def main():
    t0 = time.time()
    print('\n' + '╔' + '═'*58 + '╗')
    print('║  ALMATY TRAFFIC PREDICTION — TRAIN & EVALUATE PIPELINE  ║')
    print('╚' + '═'*58 + '╝\n')

    # 1. Load
    df, train, test, seg_info = load_and_split(DATA_PATH)

    # 2. Regression
    reg_models, reg_results, reg_preds, y_test_reg, baseline = train_regression(train, test)

    # 3. Classification
    cls_models, cls_results, cls_preds, y_test_cls, cls_naive = train_classification(train, test)

    # 4. Explainability
    best_key = 'XGBoost' if 'XGBoost' in reg_models else list(reg_models.keys())[0]
    fi, shap_values, X_shap = explain_models(reg_models, test[FEATURES])

    # 5. Clustering
    cluster_agg, sils, best_k = run_clustering(df)

    # 6. Counterfactual
    best_model = reg_models[best_key]
    cf_results = run_counterfactuals(best_model, test)

    # 7. Plots
    generate_plots(reg_results, reg_preds, y_test_reg,
                   cls_preds, y_test_cls,
                   fi, shap_values, X_shap,
                   cluster_agg, sils, best_k,
                   cf_results, df)

    # 8. Save report
    report = {
        'regression': reg_results,
        'baseline': {'name': 'Naive (last value)', **baseline},
        'classification_baseline': {'name': 'Naive (congested now)', **cls_naive},
        'classification': cls_results,
        'clustering': {'k': best_k, 'silhouette': sils[best_k]},
        'counterfactual': cf_results,
        'data': {
            'total_rows': len(df),
            'train_rows': len(train),
            'test_rows': len(test),
            'segments': int(df['segment_id'].nunique()),
            'segments_total': seg_info['total'],
            'features': len(FEATURES),
            'date_range': f'{df["timestamp"].min().date()} → {df["timestamp"].max().date()}',
        },
        'feature_importance': fi.head(10)[['feature','importance']].to_dict('records'),
    }
    # Convert numpy types for JSON
    def convert(o):
        if hasattr(o, 'item'): return o.item()
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list): return [convert(v) for v in o]
        return o
    report = convert(report)
    with open(os.path.join(MODEL_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Save segments data for API
    seg_data = df.groupby('segment_id').agg(
        lat=('lat','first'), lon=('lon','first'),
        street=('street_corrected','first') if 'street_corrected' in df.columns else ('street_name','first'),
        mean_score=('traffic_score','mean'),
        max_score=('traffic_score','max'),
    ).reset_index()
    seg_data = seg_data.merge(cluster_agg[['segment_id','cluster']], on='segment_id', how='left')
    seg_data.to_csv(os.path.join(MODEL_DIR, 'segments.csv'), index=False)

    # Hotspots table + map grid (regenerated on every retrain)
    build_streets(df).to_csv(os.path.join(MODEL_DIR, 'streets.csv'), index=False)
    os.makedirs('static', exist_ok=True)
    with open(os.path.join('static', 'map_data.json'), 'w') as f:
        json.dump(build_map_data(df), f, ensure_ascii=False)

    # Save feature list
    json.dump({'features': FEATURES, 'target': TARGET, 'threshold': CONGESTION_THRESHOLD,
               'cls_proba_threshold': cls_results.get('XGBoost', {}).get('threshold', 0.5)},
              open(os.path.join(MODEL_DIR, 'config.json'), 'w'), indent=2)

    elapsed = time.time() - t0
    print('\n' + '╔' + '═'*58 + '╗')
    print('║                    RESULTS SUMMARY                     ║')
    print('╚' + '═'*58 + '╝')
    print(f'\n  ⏱  Elapsed: {elapsed:.1f}s')
    print(f'\n  📊 Regression:')
    for name, m in reg_results.items():
        print(f'     {name:<18s}  MAE={m["MAE"]}  R²={m["R2"]}  sMAPE={m["sMAPE"]}%')
    print(f'\n  🎯 Classification:')
    for name, m in cls_results.items():
        print(f'     {name:<18s}  Acc={m["accuracy"]}  F1={m["f1"]}  Prec={m["precision"]}  Rec={m["recall"]}')
    print(f'\n  🗺  Clustering: K={best_k}, silhouette={sils[best_k]}')
    print(f'\n  📁 Saved to: {MODEL_DIR}/')
    print(f'  📈 Plots to: {PLOT_DIR}/')
    print(f'\n  ✅ Pipeline complete!')

    return report


if __name__ == '__main__':
    main()
