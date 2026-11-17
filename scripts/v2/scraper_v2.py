"""
V2 scraper: Yandex z16 traffic tiles -> per (road edge, direction) scores -> PostgreSQL.

One run = one 30-min slot (UTC, rounded down). Steps:
  1. register the run in scrape_run (a slot already done is skipped unless --force)
  2. download tiles from data/v2/tiles_z16.csv (EPSG:3395 grid), limited concurrency,
     retry/backoff on 429/5xx, circuit breaker on repeated 403/429 (= being blocked)
  3. classify pixels (numpy) and aggregate per (edge, dir) via data/v2/pixmap_z16.npz
  4. COPY rows into traffic_obs, weather into weather_obs, counters into scrape_run

    venv/bin/python scripts/v2/scraper_v2.py --from-cache data/v2_experiment/tiles   # offline test
    venv/bin/python scripts/v2/scraper_v2.py --limit 20                              # small live run
    venv/bin/python scripts/v2/scraper_v2.py                                         # full run (~2.2k tiles)
Exit code 1 if fewer than MIN_OK_SHARE of the tiles succeeded (for cron alerting).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import pandas as pd
from PIL import Image

from common import REQUEST_HEADERS, TILE_URL, V2DATA, ZOOM
from db import connect
from pixel_map import load_map, score_tiles

CONCURRENCY = 4
JITTER_S = (0.05, 0.25)       # random pause before each request
RETRIES = 3
BLOCK_LIMIT = 25              # this many 403/429 in one run -> stop (probably blocked)
MIN_OK_SHARE = 0.8
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"


def slot_now() -> datetime:
    t = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return t - timedelta(minutes=t.minute % 30)


class Stats:
    ok = empty = err = blocked = 0


async def fetch_one(session, sem, x, y, st: Stats):
    url = TILE_URL.format(x=x, y=y, z=ZOOM)
    for attempt in range(1, RETRIES + 1):
        if st.blocked >= BLOCK_LIMIT:
            return None
        async with sem:
            await asyncio.sleep(random.uniform(*JITTER_S))
            try:
                async with session.get(url, headers=REQUEST_HEADERS,
                                       timeout=aiohttp.ClientTimeout(total=15)) as r:
                    if r.status == 200:
                        st.ok += 1
                        return Image.open(io.BytesIO(await r.read()))
                    if r.status == 204:
                        st.empty += 1
                        return None
                    if r.status in (403, 429):
                        st.blocked += 1
                    elif r.status < 500:
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
        await asyncio.sleep(2 ** attempt)
    st.err += 1
    return None


async def fetch_all(tiles, concurrency):
    st = Stats()
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as s:
        imgs = await asyncio.gather(*(fetch_one(s, sem, x, y, st) for x, y in tiles))
    return {k: im for k, im in zip(tiles, imgs) if im is not None}, st


def fetch_weather() -> dict | None:
    key = os.getenv("OWM_API_KEY")
    if not key:
        return None
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode({"q": "Almaty,KZ", "appid": key, "units": "metric"})
    try:
        with urllib.request.urlopen(f"{OWM_URL}?{q}", timeout=10) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        print("weather failed:", e)
        return None
    precip = d.get("rain", {}).get("1h", 0) + d.get("snow", {}).get("1h", 0)
    return {"temp_c": d["main"].get("temp"), "humidity": d["main"].get("humidity"),
            "wind_ms": d.get("wind", {}).get("speed"), "precip_1h": precip,
            "visibility": d.get("visibility"),
            "description": (d.get("weather") or [{}])[0].get("description")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", type=Path, help="score cached PNGs {z}_{x}_{y}.png, no network")
    ap.add_argument("--limit", type=int, help="only the first N tiles (busiest first)")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--force", action="store_true", help="overwrite an already scraped slot")
    a = ap.parse_args()

    t0 = time.time()
    ts = slot_now()
    tiles_df = pd.read_csv(V2DATA / f"tiles_z{ZOOM}.csv").sort_values("n_pixels", ascending=False)
    if a.limit:
        tiles_df = tiles_df.head(a.limit)
    tiles = list(zip(tiles_df.tile_x, tiles_df.tile_y))
    note = None

    if a.from_cache:
        imgs = {}
        for f in sorted(a.from_cache.glob(f"{ZOOM}_*.png")):
            _, x, y = f.stem.split("_")
            imgs[(int(x), int(y))] = Image.open(f)
        st = Stats()
        st.ok = len(imgs)
        tiles = list(imgs)
        note = f"from cache {a.from_cache.name}"
    else:
        with connect() as conn:
            done = conn.execute("SELECT 1 FROM scrape_run WHERE ts=%s AND zoom=%s AND finished_at IS NOT NULL",
                                (ts, ZOOM)).fetchone()
        if done and not a.force:
            print(f"slot {ts:%Y-%m-%d %H:%M} UTC already scraped — skip")
            return
        imgs, st = asyncio.run(fetch_all(tiles, a.concurrency))
        if st.blocked >= BLOCK_LIMIT:
            note = f"STOPPED: {st.blocked} responses 403/429 (blocked?)"

    pmap = load_map()
    res = score_tiles(imgs, pmap)
    weather = None if a.from_cache else fetch_weather()

    with connect() as conn, conn.cursor() as cur:
        # started_at = when downloading began (t0); finished_at = clock_timestamp() (now() is frozen per transaction)
        cur.execute("""INSERT INTO scrape_run (ts, zoom, tiles_total, note, started_at)
                       VALUES (%s, %s, %s, %s, to_timestamp(%s))
                       ON CONFLICT (ts, zoom) DO UPDATE SET started_at = EXCLUDED.started_at,
                         tiles_total = EXCLUDED.tiles_total, note = EXCLUDED.note
                       RETURNING run_id""", (ts, ZOOM, len(tiles), note, t0))
        run_id = cur.fetchone()[0]
        cur.execute("DELETE FROM traffic_obs WHERE ts = %s", (ts,))
        if len(res):
            with cur.copy("COPY traffic_obs (ts, edge_id, dir, score, n_px, n_free, n_slow, "
                          "n_moderate, n_heavy, n_jam) FROM STDIN") as cp:
                for r in res.itertuples(index=False):
                    cp.write_row((ts, int(r.edge_id), int(r.dir), round(float(r.score), 3),
                                  min(int(r.n_px), 32767), *(min(int(v), 32767) for v in
                                  (r.n_free, r.n_slow, r.n_moderate, r.n_heavy, r.n_jam))))
        if weather:
            cur.execute("""INSERT INTO weather_obs (ts, temp_c, humidity, wind_ms, precip_1h, visibility, description)
                           VALUES (%(ts)s, %(temp_c)s, %(humidity)s, %(wind_ms)s, %(precip_1h)s, %(visibility)s, %(description)s)
                           ON CONFLICT (ts) DO NOTHING""", {"ts": ts, **weather})
        cur.execute("""UPDATE scrape_run SET finished_at = clock_timestamp(), tiles_ok = %s, tiles_empty = %s,
                       tiles_err = %s, rows_written = %s WHERE run_id = %s""",
                    (st.ok, st.empty, st.err + st.blocked, len(res), run_id))

    cov = res.attrs.get("coverage", 0) if len(res) else 0
    print(f"slot {ts:%Y-%m-%d %H:%M} UTC | tiles {len(tiles)}: ok {st.ok}, empty {st.empty}, "
          f"err {st.err}, blocked {st.blocked} | rows {len(res):,} | pixel match {100 * cov:.1f}% | "
          f"weather {'yes' if weather else 'no'} | {time.time() - t0:.1f}s" + (f" | {note}" if note else ""))
    if (st.ok + st.empty) < MIN_OK_SHARE * len(tiles):
        sys.exit(1)


if __name__ == "__main__":
    main()
