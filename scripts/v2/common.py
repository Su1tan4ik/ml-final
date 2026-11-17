"""Shared V2 helpers: Yandex tile projection (EPSG:3395) and traffic pixel classes."""
from __future__ import annotations

import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scraper.config import TILE_URL, REQUEST_HEADERS  # noqa: E402

OSM = ROOT / "data" / "osm"
V2DATA = ROOT / "data" / "v2"
E = 0.0818191908426  # WGS84 eccentricity
ZOOM = 16

# pixel classes (index into CLASS_NAMES / CLASS_SCORE); -1 = no traffic colour
CLASS_NAMES = ["free", "slow", "moderate", "heavy", "jam"]
CLASS_SCORE = np.array([1.5, 3.5, 5.5, 7.5, 9.5])  # same weights as V1


# ------------------------------------------------------------- projection
def merc_y(lat, ellipsoid: bool = True):
    """Normalised mercator y in [0,1] (0 = north). Yandex uses ellipsoid=True (EPSG:3395)."""
    phi = np.radians(lat)
    y = np.log(np.tan(np.pi / 4 + phi / 2))
    if ellipsoid:
        es = E * np.sin(phi)
        y = y - E / 2 * np.log((1 + es) / (1 - es))
    return (1 - y / np.pi) / 2


def inv_merc(v: float) -> float:
    """Inverse of merc_y(lat, ellipsoid=True)."""
    t = math.exp(-(1 - 2 * v) * math.pi)
    lat = math.pi / 2 - 2 * math.atan(t)
    for _ in range(10):
        es = E * math.sin(lat)
        lat = math.pi / 2 - 2 * math.atan(t * ((1 - es) / (1 + es)) ** (E / 2))
    return math.degrees(lat)


def to_px(lat, lon, z: int = ZOOM):
    """Global pixel coords (Yandex grid, 256-px tiles). Works on scalars or arrays."""
    n = 256 * 2 ** z
    return (np.asarray(lon) + 180) / 360 * n, merc_y(lat) * n


def m_per_px(lat: float, z: int = ZOOM) -> float:
    return 40075016.7 * math.cos(math.radians(lat)) / (256 * 2 ** z)


# ------------------------------------------------------------- pixels
def traffic_classes(img: Image.Image) -> np.ndarray:
    """Vectorised pixel classification -> int8 (256x256), -1 = not traffic."""
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    mx = np.maximum(np.maximum(r, g), b)
    d = mx - np.minimum(np.minimum(r, g), b)
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    m = d > 0
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = (60 * (g[rm] - b[rm]) / d[rm] + 360) % 360
    h[gm] = 60 * (b[gm] - r[gm]) / d[gm] + 120
    h[bm] = 60 * (r[bm] - g[bm]) / d[bm] + 240
    ok = (al >= 30) & (s > 0.25) & (mx > 40)  # drops grey "closed / no data" lines
    red = ok & ((h < 10) | (h >= 340))
    cls = np.full(mx.shape, -1, np.int8)
    cls[ok & (h >= 80) & (h <= 160)] = 0
    cls[ok & (h >= 35) & (h < 80)] = 1
    cls[ok & (h >= 10) & (h < 35)] = 2
    cls[red & (mx >= 150)] = 3
    cls[red & (mx < 150)] = 4  # dark red = jam
    return cls


def fetch_tile(x: int, y: int, z: int = ZOOM, cache: Path | None = None,
               pause: float = 0.4) -> Image.Image | None:
    """Download one Yandex traffic tile (x, y in the EPSG:3395 grid). None if empty (204)."""
    if cache is not None:
        f = cache / f"{z}_{x}_{y}.png"
        if f.exists():
            return Image.open(f)
    req = urllib.request.Request(TILE_URL.format(x=x, y=y, z=z), headers=REQUEST_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read() if r.status == 200 else b""
    finally:
        time.sleep(pause)
    if not data:
        return None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{z}_{x}_{y}.png").write_bytes(data)
    import io
    return Image.open(io.BytesIO(data))
