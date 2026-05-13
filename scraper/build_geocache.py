"""
One-time script: builds the reverse geocoding cache for all Almaty tiles.
Run this ONCE before starting regular scraping.

  python build_geocache.py

Takes ~3 minutes (Nominatim rate limit: 1 req/sec).
After this, scraper.py never waits for geocoding again.
"""

import asyncio
import logging

from config import ALMATY_BOUNDS, ZOOM
from geocoder import build_geocode_cache
from tiles import tiles_for_bbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

async def main():
    tiles = tiles_for_bbox(
        ALMATY_BOUNDS["lat_min"], ALMATY_BOUNDS["lat_max"],
        ALMATY_BOUNDS["lon_min"], ALMATY_BOUNDS["lon_max"],
        ZOOM,
    )
    print(f"Building geocode cache for {len(tiles)} tiles...")
    cache = await build_geocode_cache(tiles)
    filled = sum(1 for v in cache.values() if v)
    print(f"\nDone: {filled}/{len(cache)} tiles have street names.")

if __name__ == "__main__":
    asyncio.run(main())
