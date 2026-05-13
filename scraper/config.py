"""
Configuration for the Almaty traffic scraper (IMPROVED v2).

Changes from v1:
  1. HSV saturation threshold for "free" lowered: 25 → 10
     (Yandex draws pale green overlays with S=10-25 that v1 missed)
  2. Added BACKGROUND detection (white/gray pixels to ignore)
  3. Wider HSV ranges verified against real tile diagnostics

Get free weather key: https://openweathermap.org/api
"""

import os

# ---------------------------------------------------------------------------
# Almaty city bounding box
# ---------------------------------------------------------------------------
ALMATY_BOUNDS = {
    "lat_min": 43.17,
    "lat_max": 43.38,
    "lon_min": 76.82,
    "lon_max": 77.07,
}

ZOOM = 15

# ---------------------------------------------------------------------------
# Yandex Maps traffic tile endpoint
# ---------------------------------------------------------------------------
TILE_URL = (
    "https://core-jams-rdr-cache.maps.yandex.net/1.1/tiles"
    "?trf&l=trf&x={x}&y={y}&z={z}&scale=1&lang=ru_RU"
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://yandex.kz/maps/",
    "Accept": "image/png,image/*,*/*;q=0.8",
}

# ---------------------------------------------------------------------------
# HSV color rules (verified against real Yandex tile data)
#
# From diagnostic output we see:
#   - Pure green traffic: HSV(110-115, 40-65, 78-84) → S > 25
#   - Pale green (blended with map): HSV(105-120, 8-26, 84-100) → S = 8-26
#   - White background: HSV(0-120, 0-5, 96-100) → S < 5
#
# Key fix: "free" S_min lowered from 25 to 10 to catch pale greens.
# Pixels with S < 5 AND V > 90 are treated as background (white/map).
#
# (label, score_min, score_max, h_min, h_max, s_min, v_min, v_max)
# ---------------------------------------------------------------------------
HSV_RULES = [
    ("free",      1, 2,    80, 160,  10,  25, 100),   # green (S lowered: 25→10)
    ("slow",      3, 4,    35,  80,  25,  40, 100),   # yellow
    ("moderate",  5, 6,    10,  35,  25,  40, 100),   # orange
    ("heavy",     7, 8,   350,  10,  25,  35, 100),   # red (wraps around 0°)
    ("jam",       9, 10,  345,  15,  25,  15,  50),   # dark red / maroon
]

# Fallback RGB rules (wider ranges)
COLOR_RULES = [
    # (label, score_min, score_max, R_min, R_max, G_min, G_max, B_min, B_max)
    ("free",      1, 2,    40, 210,  140, 255,   20, 210),  # green (much wider for pale greens)
    ("slow",      3, 4,   180, 255,  150, 255,    0,  80),  # yellow
    ("moderate",  5, 6,   190, 255,   60, 180,    0,  60),  # orange
    ("heavy",     7, 8,   180, 255,    0, 100,    0,  60),  # red
    ("jam",       9, 10,   80, 220,    0,  50,    0,  50),  # dark red
]

# Pixels with S < 5 and V > 90 are background (white/gray map tiles)
# These should NOT be counted as traffic pixels at all
BACKGROUND_S_MAX = 5
BACKGROUND_V_MIN = 90

ALPHA_THRESHOLD = 30
MIN_TRAFFIC_PIXEL_RATIO = 0.001

# ---------------------------------------------------------------------------
# HTTP settings
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 15
MAX_CONCURRENT_REQUESTS = 16
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR    = "../data"
OUTPUT_FILE   = "staging.csv"    # cleared after feature engineering processes it
LATEST_FILE   = "staging.csv"    # same file — latest snapshot = staging buffer

# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
GEOCODE_CACHE_FILE = "../data/geocode_cache.json"
NOMINATIM_RATE_LIMIT = 1.1

# ---------------------------------------------------------------------------
# Weather — paste your key here or set env var OWM_API_KEY
# Get free key: https://openweathermap.org/api
# ---------------------------------------------------------------------------
OWM_API_KEY = os.environ.get("OWM_API_KEY", "")
OWM_CITY = "Almaty,KZ"

# ---------------------------------------------------------------------------
# Speed estimate
# ---------------------------------------------------------------------------
SPEED_FREE_KMH = 65.0
SPEED_JAM_KMH  = 3.0


def score_to_speed(score: float) -> float:
    if score <= 0:
        return 0.0
    s = max(1.0, min(10.0, score))
    return round(SPEED_FREE_KMH - (s - 1) * (SPEED_FREE_KMH - SPEED_JAM_KMH) / 9.0, 1)


def score_to_label(score: float) -> str:
    if score <= 0:
        return "Unknown"
    if score <= 3:
        return "Low"
    if score <= 6:
        return "Medium"
    if score <= 8:
        return "High"
    return "Critical"
