"""
Reverse geocoding via Nominatim (OpenStreetMap) — free, no API key.

Cache strategy:
- Local dev:  JSON file at data/geocode_cache.json
- Render/prod: Supabase geocode_cache table (persistent across cron runs)

Tiles are geocoded once and never re-queried.
"""

import asyncio
import json
import logging
from pathlib import Path

import aiohttp

from config import GEOCODE_CACHE_FILE, NOMINATIM_RATE_LIMIT

log = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {
    "User-Agent": "AlmatyTrafficResearch/1.0 (student project)",
    "Accept-Language": "ru,en",
}


# ---------------------------------------------------------------------------
# Cache: load / save  (auto-detects local vs Supabase)
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    """Load geocode cache from local JSON file."""
    path = Path(GEOCODE_CACHE_FILE)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    """Save geocode cache to local JSON file."""
    path = Path(GEOCODE_CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Reverse geocode one point
# ---------------------------------------------------------------------------

async def _fetch_street_name(session: aiohttp.ClientSession, lat: float, lon: float) -> str:
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 16, "addressdetails": 1}
    try:
        async with session.get(
            NOMINATIM_URL,
            params=params,
            headers=NOMINATIM_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json(content_type=None)
            addr = data.get("address", {})
            name = (
                addr.get("road")
                or addr.get("pedestrian")
                or addr.get("footway")
                or addr.get("cycleway")
                or addr.get("path")
                or addr.get("neighbourhood")
                or addr.get("quarter")
                or addr.get("city_district")
                or addr.get("suburb")
                or addr.get("county")
                or ""
            )
            return name[:120]
    except Exception as exc:
        log.debug("Nominatim error (%.4f, %.4f): %s", lat, lon, exc)
        return ""


# ---------------------------------------------------------------------------
# Build / update cache for a list of tiles
# ---------------------------------------------------------------------------

async def build_geocode_cache(tiles: list) -> dict:
    """
    Geocode all tiles not yet in cache.
    Saves checkpoints every 20 tiles so it's safe to interrupt.
    Returns full cache dict {segment_id: street_name}.
    """
    cache = load_cache()
    missing = [t for t in tiles if t.segment_id not in cache]

    if not missing:
        log.info("Geocode cache: all %d tiles already cached.", len(tiles))
        return cache

    log.info(
        "Geocode cache: %d new tiles to geocode (~%ds at 1 req/sec).",
        len(missing),
        int(len(missing) * NOMINATIM_RATE_LIMIT),
    )

    async with aiohttp.ClientSession() as session:
        for i, tile in enumerate(missing, 1):
            lat, lon = tile.center_latlon()
            cache[tile.segment_id] = await _fetch_street_name(session, lat, lon)

            if i % 20 == 0 or i == len(missing):
                log.info("  Geocoded %d / %d", i, len(missing))
                save_cache(cache)

            if i < len(missing):
                await asyncio.sleep(NOMINATIM_RATE_LIMIT)

    return cache
