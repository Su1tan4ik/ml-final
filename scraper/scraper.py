"""
Almaty traffic scraper — IMPROVED VERSION.

Key improvements over original:
  1. HSV-based pixel classification (catches orange/red/dark-red that RGB missed)
  2. Lower alpha threshold (30 vs 60) — detects more traffic pixels
  3. Debug mode to log color distribution per run
  4. Saves color diagnostics to help tune COLOR_RULES if needed

Run once:  python scraper.py
Cron:      */30 * * * * /path/to/run_scraper.sh
Output:    data/traffic_data.csv   ← append-only log (for ML)
           data/traffic_latest.csv ← current city snapshot
"""

import asyncio
import csv
import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from PIL import Image

from config import (
    ALMATY_BOUNDS,
    ALPHA_THRESHOLD,
    BACKGROUND_S_MAX,
    BACKGROUND_V_MIN,
    COLOR_RULES,
    HSV_RULES,
    LATEST_FILE,
    MAX_CONCURRENT_REQUESTS,
    MIN_TRAFFIC_PIXEL_RATIO,
    OUTPUT_DIR,
    OUTPUT_FILE,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF,
    TILE_URL,
    ZOOM,
    score_to_label,
    score_to_speed,
)
from geocoder import build_geocode_cache, load_cache
from tiles import Tile, tiles_for_bbox
from weather import fetch_weather

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV schema (same as original — backward compatible)
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "timestamp", "segment_id", "lat", "lon", "street_name",
    "traffic_score", "congestion_level", "speed_avg",
    "pixels_free", "pixels_slow", "pixels_moderate",
    "pixels_heavy", "pixels_jam", "total_traffic_pixels",
    "pct_free", "pct_slow", "pct_moderate", "pct_heavy", "pct_jam",
    "weather_temp_c", "weather_feels_c", "weather_humidity",
    "weather_wind_ms", "weather_precip_1h", "weather_visibility_m",
    "weather_condition", "weather_description",
    "tile_x", "tile_y", "tile_z",
]

# ---------------------------------------------------------------------------
# HSV conversion
# ---------------------------------------------------------------------------

def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """
    Convert RGB (0-255) to HSV (H: 0-360, S: 0-100, V: 0-100).
    """
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r_, g_, b_), min(r_, g_, b_)
    diff = mx - mn

    v = mx * 100
    s = (diff / mx * 100) if mx > 0 else 0

    if diff == 0:
        h = 0
    elif mx == r_:
        h = (60 * ((g_ - b_) / diff) + 360) % 360
    elif mx == g_:
        h = (60 * ((b_ - r_) / diff) + 120) % 360
    else:
        h = (60 * ((r_ - g_) / diff) + 240) % 360

    return h, s, v


# ---------------------------------------------------------------------------
# IMPROVED pixel classification — HSV primary, RGB fallback
# ---------------------------------------------------------------------------

def classify_pixel_hsv(r: int, g: int, b: int) -> str | None:
    """
    Classify a pixel using HSV color space (primary method).
    More robust than RGB for detecting traffic overlay colors.
    """
    h, s, v = rgb_to_hsv(r, g, b)

    # Skip background pixels (white/gray map tiles)
    if s <= BACKGROUND_S_MAX and v >= BACKGROUND_V_MIN:
        return None

    # Skip very dark pixels (not traffic overlay)
    if v < 15:
        return None

    for label, _s_min, _s_max, h_min, h_max, s_min, v_min, v_max in HSV_RULES:
        # Handle hue wrap-around for red (350-10 degrees)
        if h_min > h_max:
            h_match = (h >= h_min or h <= h_max)
        else:
            h_match = (h_min <= h <= h_max)

        if h_match and s >= s_min and v_min <= v <= v_max:
            return label

    return None


def classify_pixel_rgb(r: int, g: int, b: int) -> str | None:
    """Fallback: classify using RGB ranges."""
    for label, _s_min, _s_max, r_min, r_max, g_min, g_max, b_min, b_max in COLOR_RULES:
        if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
            return label
    return None


def classify_pixel(r: int, g: int, b: int) -> str | None:
    """
    Classify a pixel: try HSV first, then RGB fallback.
    """
    result = classify_pixel_hsv(r, g, b)
    if result is not None:
        return result
    return classify_pixel_rgb(r, g, b)


# ---------------------------------------------------------------------------
# Tile image analysis
# ---------------------------------------------------------------------------

def analyse_tile_image(image_bytes: bytes) -> dict:
    """
    Parse a 256×256 RGBA PNG tile.
    Returns raw pixel counts, percentages, and traffic_score.
    """
    labels = ["free", "slow", "moderate", "heavy", "jam"]
    counts = {label: 0 for label in labels}

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    pixels = img.load()
    width, height = img.size

    total_pixels = 0
    for py in range(height):
        for px in range(width):
            r, g, b, a = pixels[px, py]
            if a < ALPHA_THRESHOLD:  # <-- IMPROVED: lower threshold
                continue
            total_pixels += 1
            label = classify_pixel(r, g, b)
            if label and label in counts:
                counts[label] += 1

    total_traffic = sum(counts.values())

    # Weighted average score using HSV_RULES score ranges
    score_weights = {"free": 1.5, "slow": 3.5, "moderate": 5.5, "heavy": 7.5, "jam": 9.5}
    if total_traffic == 0:
        traffic_score = 0.0
    else:
        weighted = sum(counts[label] * score_weights[label] for label in labels)
        traffic_score = round(weighted / total_traffic, 2)

    # Reject near-empty tiles
    if total_pixels > 0 and total_traffic / total_pixels < MIN_TRAFFIC_PIXEL_RATIO:
        traffic_score = 0.0

    def pct(n: int) -> float:
        return round(n / total_traffic * 100, 2) if total_traffic > 0 else 0.0

    return {
        "traffic_score":    traffic_score,
        "pixels_free":      counts["free"],
        "pixels_slow":      counts["slow"],
        "pixels_moderate":  counts["moderate"],
        "pixels_heavy":     counts["heavy"],
        "pixels_jam":       counts["jam"],
        "total_traffic_pixels": total_traffic,
        "pct_free":      pct(counts["free"]),
        "pct_slow":      pct(counts["slow"]),
        "pct_moderate":  pct(counts["moderate"]),
        "pct_heavy":     pct(counts["heavy"]),
        "pct_jam":       pct(counts["jam"]),
    }


# ---------------------------------------------------------------------------
# Fetch one tile
# ---------------------------------------------------------------------------

async def fetch_tile(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    tile: Tile,
    timestamp: str,
    street_name: str,
    weather: dict,
) -> dict | None:
    url = TILE_URL.format(x=tile.x, y=tile.y, z=tile.z)
    lat, lon = tile.center_latlon()

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with sem:
                async with session.get(
                    url,
                    headers=REQUEST_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 204:
                        return None
                    if resp.status == 429:
                        await asyncio.sleep(RETRY_BACKOFF * attempt)
                        continue
                    if resp.status != 200:
                        log.debug("HTTP %s for tile %s", resp.status, tile.segment_id)
                        break
                    image_bytes = await resp.read()

            analysis = analyse_tile_image(image_bytes)
            score = analysis["traffic_score"]

            return {
                "timestamp":        timestamp,
                "segment_id":       tile.segment_id,
                "lat":              lat,
                "lon":              lon,
                "street_name":      street_name,
                "traffic_score":    score,
                "congestion_level": score_to_label(score),
                "speed_avg":        score_to_speed(score),
                **{k: analysis[k] for k in analysis if k != "traffic_score"},
                **weather,
                "tile_x": tile.x,
                "tile_y": tile.y,
                "tile_z": tile.z,
            }

        except asyncio.TimeoutError:
            log.warning("Timeout — %s (attempt %d/%d)", tile.segment_id, attempt, RETRY_ATTEMPTS)
        except Exception as exc:
            log.warning("Error — %s: %s (attempt %d/%d)", tile.segment_id, exc, attempt, RETRY_ATTEMPTS)

        if attempt < RETRY_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF * attempt)

    return {
        "timestamp":        timestamp,
        "segment_id":       tile.segment_id,
        "lat":              lat,
        "lon":              lon,
        "street_name":      street_name,
        "traffic_score":    None,
        "congestion_level": "Unknown",
        "speed_avg":        None,
        "pixels_free": 0, "pixels_slow": 0, "pixels_moderate": 0,
        "pixels_heavy": 0, "pixels_jam": 0, "total_traffic_pixels": 0,
        "pct_free": 0.0, "pct_slow": 0.0, "pct_moderate": 0.0,
        "pct_heavy": 0.0, "pct_jam": 0.0,
        **weather,
        "tile_x": tile.x, "tile_y": tile.y, "tile_z": tile.z,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def append_to_csv(rows: list[dict]):
    path = Path(OUTPUT_DIR) / OUTPUT_FILE
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    log.info("Appended %d rows → %s", len(rows), path)


def write_latest_csv(rows: list[dict]):
    path = Path(OUTPUT_DIR) / LATEST_FILE
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Latest snapshot → %s", path)


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------

async def run_collection():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.info("=== Collection started at %s ===", timestamp)

    all_tiles = tiles_for_bbox(
        ALMATY_BOUNDS["lat_min"],
        ALMATY_BOUNDS["lat_max"],
        ALMATY_BOUNDS["lon_min"],
        ALMATY_BOUNDS["lon_max"],
        ZOOM,
    )
    log.info("Tiles to process: %d (zoom=%d)", len(all_tiles), ZOOM)

    geocode_cache = load_cache()
    if not geocode_cache:
        log.warning("Geocode cache is empty! Run: python build_geocache.py")
        log.warning("Continuing without street names...")

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)

    rows: list[dict] = []
    skipped = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        weather = await fetch_weather(session)
        if weather.get("weather_condition"):
            log.info("Weather: %s %.1f°C", weather["weather_condition"], weather["weather_temp_c"])
        else:
            log.info("Weather: not configured (set OWM_API_KEY)")

        tasks = [
            fetch_tile(
                session, sem, tile, timestamp,
                geocode_cache.get(tile.segment_id, ""),
                weather,
            )
            for tile in all_tiles
        ]

        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            results = await asyncio.gather(*tasks[i : i + chunk_size])
            for r in results:
                if r is None:
                    skipped += 1
                else:
                    rows.append(r)
            done = min(i + chunk_size, len(tasks))
            log.info("Progress: %d / %d tiles  (%d rows collected)", done, len(tasks), len(rows))

    ensure_output_dir()
    append_to_csv(rows)
    write_latest_csv(rows)

    # --- IMPROVED: Log congestion summary ---
    success = sum(1 for r in rows if r["traffic_score"] is not None)
    scores = [r["traffic_score"] for r in rows if r["traffic_score"] is not None and r["traffic_score"] > 0]
    if scores:
        import statistics
        log.info("Score stats: mean=%.2f, max=%.2f, std=%.3f",
                 statistics.mean(scores), max(scores), statistics.stdev(scores) if len(scores) > 1 else 0)
        level_counts = {}
        for r in rows:
            lvl = r.get("congestion_level", "Unknown")
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        log.info("Congestion levels: %s", level_counts)

        # Log pixel category totals
        for cat in ["moderate", "heavy", "jam"]:
            total = sum(r.get(f"pixels_{cat}", 0) for r in rows)
            if total > 0:
                log.info("  pixels_%s: %d (detected!)", cat, total)
            else:
                log.warning("  pixels_%s: 0 — still not detecting %s traffic!", cat, cat)

    log.info(
        "=== Done: %d rows saved | %d empty tiles skipped | %s ===",
        success, skipped, timestamp,
    )
    return rows


def main():
    t0 = time.time()
    asyncio.run(run_collection())
    log.info("Total elapsed: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
