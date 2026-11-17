"""
V2 Stage 1 — zoom experiment.

Question: at which zoom do Yandex traffic tiles colour small streets, and
which tile projection (Web Mercator EPSG:3857 vs Yandex EPSG:3395) lines up
with OSM roads?

Downloads the tiles covering ONE z15 tile (~890 m) in central Almaty at
z15..z18 (1 + 4 + 16 + 64 = 85 requests, sequential, with a pause; cached),
then for every OSM road in that area samples points every ~5 m and checks
whether a coloured traffic pixel lies within NEAR_M metres (Yandex draws each
direction offset from the road centreline). Writes summary + overlay PNGs.

    venv/bin/python scripts/v2/zoom_experiment.py
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scraper.config import TILE_URL, REQUEST_HEADERS  # noqa: E402

OUT = ROOT / "data" / "v2_experiment"
ROADS = ROOT / "data" / "osm" / "almaty_roads.geojson"

# Abay ave x Dostyk ave, central Almaty
CENTER = (43.2400, 76.9500)
BASE_Z = 15
ZOOMS = [15, 16, 17, 18]
NEAR_M = 12.0  # tolerance around the OSM centreline
STEP_M = 5.0

E = 0.0818191908426  # WGS84 eccentricity


# ---------------------------------------------------------------- projection
def merc_y(lat: float, ellipsoid: bool) -> float:
    """Normalised mercator y in [0,1] (0 = north)."""
    phi = math.radians(lat)
    y = math.log(math.tan(math.pi / 4 + phi / 2))
    if ellipsoid:  # Yandex / EPSG:3395
        es = E * math.sin(phi)
        y -= E / 2 * math.log((1 + es) / (1 - es))
    return (1 - y / math.pi) / 2


def inv_merc(v: float) -> float:
    """Inverse of merc_y(lat, ellipsoid=True) (iterative)."""
    t = math.exp(-(1 - 2 * v) * math.pi)
    lat = math.pi / 2 - 2 * math.atan(t)
    for _ in range(10):
        es = E * math.sin(lat)
        lat = math.pi / 2 - 2 * math.atan(t * ((1 - es) / (1 + es)) ** (E / 2))
    return math.degrees(lat)


def to_px(lat: float, lon: float, z: int, ellipsoid: bool) -> tuple[float, float]:
    """Global pixel coords at zoom z (256-px tiles)."""
    n = 256 * 2 ** z
    return (lon + 180) / 360 * n, merc_y(lat, ellipsoid) * n


# ---------------------------------------------------------------- pixels
def traffic_mask(img: Image.Image) -> np.ndarray:
    """Vectorised traffic-colour mask -> int8 class (-1 none, 0..3 green..red)."""
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    m = d > 0
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = (60 * (g[rm] - b[rm]) / d[rm] + 360) % 360
    h[gm] = 60 * (b[gm] - r[gm]) / d[gm] + 120
    h[bm] = 60 * (r[bm] - g[bm]) / d[bm] + 240
    ok = (al >= 30) & (s > 0.25) & (mx > 40)
    cls = np.full(mx.shape, -1, np.int8)
    cls[ok & (h >= 80) & (h <= 160)] = 0          # green
    cls[ok & (h >= 35) & (h < 80)] = 1            # yellow
    cls[ok & (h >= 10) & (h < 35)] = 2            # orange
    cls[ok & ((h < 10) | (h >= 340))] = 3         # red
    return cls


def fetch(x: int, y: int, z: int) -> Image.Image | None:
    cache = OUT / "tiles" / f"{z}_{x}_{y}.png"
    if cache.exists():
        return Image.open(cache)
    req = urllib.request.Request(TILE_URL.format(x=x, y=y, z=z), headers=REQUEST_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read() if r.status == 200 else b""
    except Exception as e:  # noqa: BLE001
        print(f"  ! z{z} {x},{y}: {e}")
        return None
    finally:
        time.sleep(0.4)
    if not data:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return Image.open(io.BytesIO(data))


def mosaic(z: int, ellipsoid: bool, bx0: float, by0: float, bx1: float, by1: float):
    """Fetch tiles covering global-pixel box at zoom z; return (mask, rgba, ox, oy)."""
    tx0, ty0 = int(bx0 // 256), int(by0 // 256)
    tx1, ty1 = int((bx1 - 1) // 256), int((by1 - 1) // 256)
    W, H = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            im = fetch(tx, ty, z)
            if im is not None:
                canvas.paste(im.convert("RGBA"), ((tx - tx0) * 256, (ty - ty0) * 256))
    return traffic_mask(canvas), canvas, tx0 * 256, ty0 * 256


# ---------------------------------------------------------------- roads
def haversine(a, b) -> float:
    (lon1, lat1), (lon2, lat2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def sample_line(coords, step=STEP_M):
    pts = []
    for a, b in zip(coords, coords[1:]):
        n = max(1, int(haversine(a, b) // step))
        for i in range(n):
            t = i / n
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    pts.append(coords[-1])
    return pts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # area = one BASE_Z tile in Yandex (EPSG:3395) tile space
    n = 256 * 2 ** BASE_Z
    px, py = to_px(*CENTER, BASE_Z, True)
    tx, ty = int(px // 256), int(py // 256)
    lon0, lon1 = tx * 256 / n * 360 - 180, (tx + 1) * 256 / n * 360 - 180
    lat1, lat0 = inv_merc((ty * 256) / n), inv_merc(((ty + 1) * 256) / n)
    print(f"Area: lat {lat0:.5f}..{lat1:.5f}, lon {lon0:.5f}..{lon1:.5f} "
          f"(~{haversine((lon0, lat0), (lon1, lat0)):.0f} m wide)")

    roads = []
    for f in json.load(open(ROADS))["features"]:
        c = f["geometry"]["coordinates"]
        if any(lat0 <= la <= lat1 and lon0 <= lo <= lon1 for lo, la in c):
            pts = [(lo, la) for lo, la in sample_line(c) if lat0 <= la <= lat1 and lon0 <= lo <= lon1]
            if pts:
                roads.append((f["properties"], pts))
    print(f"OSM ways in area: {len(roads)}")

    results = {}
    for ellipsoid in (True,):  # EPSG:3857 indices return 204 (empty) — verified
        proj = "EPSG3395" if ellipsoid else "EPSG3857"
        for z in ZOOMS:
            x0, y0 = to_px(lat1, lon0, z, ellipsoid)
            x1, y1 = to_px(lat0, lon1, z, ellipsoid)
            mask, rgba, ox, oy = mosaic(z, ellipsoid, x0, y0, x1, y1)
            colored = mask >= 0
            # dilate by NEAR_PX
            m_per_px = 40075016.7 * math.cos(math.radians(CENTER[0])) / (256 * 2 ** z)
            k = max(1, math.ceil(NEAR_M / m_per_px))
            near = colored.copy()
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    if dx * dx + dy * dy > k * k:
                        continue
                    near |= np.roll(np.roll(colored, dy, 0), dx, 1)
            by_cls = defaultdict(lambda: [0, 0, 0.0])  # hits, total, length_m
            draw = ImageDraw.Draw(rgba)
            for props, pts in roads:
                hw = props["highway"]
                hit = 0
                xy = []
                for lo, la in pts:
                    px, py = to_px(la, lo, z, ellipsoid)
                    ix, iy = int(px - ox), int(py - oy)
                    xy.append((ix, iy))
                    if 0 <= iy < near.shape[0] and 0 <= ix < near.shape[1] and near[iy, ix]:
                        hit += 1
                by_cls[hw][0] += hit
                by_cls[hw][1] += len(pts)
                if len(xy) > 1:
                    draw.line(xy, fill=(0, 120, 255, 160), width=1)
            road_px = colored.sum()
            # fraction of coloured pixels that lie near some OSM road (precision)
            results[(proj, z)] = {"cls": dict(by_cls), "colored_px": int(road_px), "mpp": m_per_px}
            crop = rgba.crop((int(x0 - ox), int(y0 - oy), int(x1 - ox), int(y1 - oy)))
            crop.save(OUT / f"overlay_{proj}_z{z}.png")

    groups = [("primary/trunk", {"primary", "trunk", "motorway", "primary_link", "trunk_link", "motorway_link"}),
              ("secondary", {"secondary", "secondary_link"}),
              ("tertiary", {"tertiary", "tertiary_link"}),
              ("residential/unclass.", {"residential", "unclassified", "living_street"}),
              ("service", {"service"})]
    lines = ["| projection | zoom | " + " | ".join(g for g, _ in groups) + " | coloured px | m/px |",
             "|---|---|" + "---|" * (len(groups) + 2)]
    for (proj, z), r in results.items():
        cells = []
        for _, hs in groups:
            hit = sum(v[0] for k, v in r["cls"].items() if k in hs)
            tot = sum(v[1] for k, v in r["cls"].items() if k in hs)
            cells.append(f"{100 * hit / tot:.0f}%" if tot else "—")
        lines.append(f"| {proj} | z{z} | " + " | ".join(cells) + f" | {r['colored_px']} | {r['mpp']:.2f} |")
    table = "\n".join(lines)
    print("\nShare of OSM road length with a coloured traffic pixel within "
          f"{NEAR_M:.0f} m:\n{table}")
    (OUT / "summary.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
