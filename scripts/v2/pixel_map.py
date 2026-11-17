"""
V2 Stage 3 — pixel -> road-edge map for Yandex z16 tiles, per travel direction.

For every z16 tile that contains roads, precompute which pixels belong to
which graph edge (scripts/v2/build_graph.py output) and to which direction:
  dir = +1  travel source->target  (Yandex draws a direction on its RIGHT side)
  dir = -1  travel target->source
  dir =  0  one-way edge, or pixel too close to the centreline to tell
Pixels further than NEAR_M from any eligible edge are not mapped.

Output: data/v2/pixmap_z16.npz with flat arrays
  tile_x, tile_y, pix (y*256+x, uint16), edge (int32), dir (int8)
  + tiles.csv (tile_x, tile_y, n_pixels, n_edges) = the tile list to scrape.

    venv/bin/python scripts/v2/pixel_map.py            # build map for the whole bbox
    venv/bin/python scripts/v2/pixel_map.py --test     # score cached centre tiles + overlay PNG
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import (CLASS_NAMES, CLASS_SCORE, OSM, ROOT, V2DATA, ZOOM, fetch_tile,
                    m_per_px, to_px, traffic_classes)

NEAR_M = 12.0       # max distance pixel <-> edge centreline
CENTRE_M = 1.0      # |offset| below this -> direction unknown (dir 0)
# Yandex does not colour these (stage 1); mapping them would steal pixels from real streets
SKIP_HIGHWAY = {"service", "living_street"}
MIN_PX = 3          # (edge, dir) with fewer coloured pixels = crossing-street noise, dropped
LAT_REF = 43.25


def load_edges():
    fc = json.load(open(OSM / "graph_edges.geojson"))
    edges = []
    for f in fc["features"]:
        p = f["properties"]
        if p["highway"] in SKIP_HIGHWAY:
            continue
        c = np.array(f["geometry"]["coordinates"])
        x, y = to_px(c[:, 1], c[:, 0])
        edges.append((p["edge_id"], p["oneway"], x, y))
    return edges


def build_map():
    t0 = time.time()
    R = NEAR_M / m_per_px(LAT_REF)          # radius in px (~6.9 at z16)
    C = CENTRE_M / m_per_px(LAT_REF)
    edges = load_edges()

    # bucket every segment piece into the tiles its (bbox + R) touches
    buckets = defaultdict(list)
    for eid, ow, xs, ys in edges:
        for i in range(len(xs) - 1):
            ax, ay, bx, by = xs[i], ys[i], xs[i + 1], ys[i + 1]
            x0, x1 = min(ax, bx) - R, max(ax, bx) + R
            y0, y1 = min(ay, by) - R, max(ay, by) + R
            for tx in range(int(x0 // 256), int(x1 // 256) + 1):
                for ty in range(int(y0 // 256), int(y1 // 256) + 1):
                    buckets[(tx, ty)].append((eid, ow, ax, ay, bx, by))

    out = {k: [] for k in ("tile_x", "tile_y", "pix", "edge", "dir")}
    rows = []
    for (tx, ty), pieces in buckets.items():
        best_d = np.full((256, 256), np.inf, np.float32)
        best_e = np.full((256, 256), -1, np.int32)
        best_s = np.zeros((256, 256), np.int8)
        ox, oy = tx * 256, ty * 256
        for eid, ow, ax, ay, bx, by in pieces:
            ax, ay, bx, by = ax - ox, ay - oy, bx - ox, by - oy
            i0, i1 = max(0, int(min(ax, bx) - R)), min(256, int(max(ax, bx) + R) + 1)
            j0, j1 = max(0, int(min(ay, by) - R)), min(256, int(max(ay, by) + R) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            px, py = np.meshgrid(np.arange(i0, i1) + 0.5, np.arange(j0, j1) + 0.5)
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) if L2 > 0 else 0
            d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
            # screen coords (y down): cross > 0 => pixel on the RIGHT of travel a->b
            cross = dx * (py - ay) - dy * (px - ax)
            side = np.where(np.abs(cross) / max(np.sqrt(L2), 1e-9) < C, 0, np.sign(cross)).astype(np.int8)
            if ow:
                side[:] = 0
            win = best_d[j0:j1, i0:i1]
            upd = (d < win) & (d <= R)
            win[upd] = d[upd]
            best_e[j0:j1, i0:i1][upd] = eid
            best_s[j0:j1, i0:i1][upd] = side[upd]
        pix = np.flatnonzero(best_e >= 0)
        if not len(pix):
            continue
        out["tile_x"].append(np.full(len(pix), tx, np.int32))
        out["tile_y"].append(np.full(len(pix), ty, np.int32))
        out["pix"].append(pix.astype(np.uint16))
        out["edge"].append(best_e.ravel()[pix])
        out["dir"].append(best_s.ravel()[pix])
        rows.append((tx, ty, len(pix), len(np.unique(best_e.ravel()[pix]))))

    V2DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(V2DATA / f"pixmap_z{ZOOM}.npz", **{k: np.concatenate(v) for k, v in out.items()})
    tiles = pd.DataFrame(rows, columns=["tile_x", "tile_y", "n_pixels", "n_edges"])
    tiles.to_csv(V2DATA / f"tiles_z{ZOOM}.csv", index=False)
    print(f"edges eligible {len(edges):,}; tiles with roads {len(tiles):,}; "
          f"mapped pixels {tiles.n_pixels.sum():,}; {time.time() - t0:.0f}s")
    return tiles


def load_map():
    with np.load(V2DATA / f"pixmap_z{ZOOM}.npz") as z:
        m = {k: z[k] for k in z.files}  # decompress once (NpzFile re-reads on every access)
    by_tile = {}
    keys = m["tile_x"].astype(np.int64) * 1_000_000 + m["tile_y"]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(keys)]):
        idx = order[s:e]
        k = int(keys[s])
        by_tile[(k // 1_000_000, k % 1_000_000)] = (m["pix"][idx], m["edge"][idx], m["dir"][idx])
    return by_tile


def score_tiles(images: dict, pmap: dict) -> pd.DataFrame:
    """images: {(tx,ty): PIL image}. Returns per (edge_id, dir) pixel counts + score."""
    parts = []
    unmatched = total = 0
    for key, img in images.items():
        cls = traffic_classes(img).ravel()
        coloured = cls >= 0
        total += int(coloured.sum())
        if key not in pmap:
            unmatched += int(coloured.sum())
            continue
        pix, edge, dr = pmap[key]
        c = cls[pix]
        hit = c >= 0
        mapped = np.zeros(65536, bool)
        mapped[pix[hit]] = True
        unmatched += int((coloured & ~mapped).sum())
        parts.append(pd.DataFrame({"edge_id": edge[hit], "dir": dr[hit], "cls": c[hit]}))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    cnt = pd.crosstab([df.edge_id, df.dir], df.cls).reindex(columns=range(5), fill_value=0)
    cnt.columns = [f"n_{n}" for n in CLASS_NAMES]
    cnt["n_px"] = cnt.sum(axis=1)
    cnt["score"] = (cnt[[f"n_{n}" for n in CLASS_NAMES]].values @ CLASS_SCORE) / cnt["n_px"]
    cov = 1 - unmatched / max(total, 1)
    cnt = cnt[cnt["n_px"] >= MIN_PX].reset_index()
    cnt.attrs["coverage"] = cov
    return cnt


def test():
    """Score the z16 tiles cached by the zoom experiment and draw a side-by-side check."""
    cache = ROOT / "data" / "v2_experiment" / "tiles"
    pmap = load_map()
    imgs = {}
    for f in sorted(cache.glob(f"{ZOOM}_*.png")):
        _, x, y = f.stem.split("_")
        imgs[(int(x), int(y))] = Image.open(f)
    res = score_tiles(imgs, pmap)
    print(f"tiles {len(imgs)}; coloured pixels matched to an edge: {100 * res.attrs['coverage']:.1f}%")
    print(f"(edge, dir) units with data: {len(res)}; by dir: {res['dir'].value_counts().to_dict()}")
    print(res.sort_values("score", ascending=False).head(8).to_string(index=False))

    # render: left = original Yandex, right = our per-edge/direction scores
    fc = json.load(open(OSM / "graph_edges.geojson"))
    geom = {f["properties"]["edge_id"]: np.array(f["geometry"]["coordinates"]) for f in fc["features"]}
    xs = sorted({k[0] for k in imgs}); ys = sorted({k[1] for k in imgs})
    W, H = len(xs) * 256, len(ys) * 256
    left = Image.new("RGBA", (W, H), "white"); right = Image.new("RGBA", (W, H), "white")
    for (x, y), im in imgs.items():
        left.alpha_composite(im.convert("RGBA"), ((x - xs[0]) * 256, (y - ys[0]) * 256))
    d = ImageDraw.Draw(right)
    pal = lambda s: (60, 180, 60) if s < 2.5 else (240, 200, 40) if s < 4.5 else (240, 130, 30) if s < 6.5 else (220, 40, 40) if s < 8.5 else (120, 0, 0)  # noqa: E731
    off = 4  # px to the right of travel, to mimic Yandex
    for r in res.itertuples():
        c = geom[r.edge_id]
        px, py = to_px(c[:, 1], c[:, 0])
        px = px - xs[0] * 256; py = py - ys[0] * 256
        if r.dir == -1:
            px, py = px[::-1], py[::-1]
        if r.dir != 0:
            dx, dy = np.gradient(px), np.gradient(py)
            n = np.hypot(dx, dy) + 1e-9
            px, py = px - dy / n * off, py + dx / n * off
        d.line(list(zip(px, py)), fill=pal(r.score), width=3)
    both = Image.new("RGBA", (W * 2 + 10, H), "white")
    both.paste(left, (0, 0)); both.paste(right, (W + 10, 0))
    out = V2DATA / "pixmap_test.png"
    both.save(out)
    print("overlay:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        test()
    else:
        build_map()
