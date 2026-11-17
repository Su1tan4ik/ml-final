"""
Download Almaty drivable roads from OSM Overpass -> GeoJSON (same format as
data/osm/almaty_roads.geojson: osm_id, highway, name, oneway, lanes, maxspeed, length_m, nodes).

The whole bbox times out (504), so it is fetched in NxN quadrants and merged by way id.

⚠️ A fresh download gives DIFFERENT edge ids after build_graph.py (OSM changes). Never rebuild the
graph of a DB that already has traffic_obs — when moving to a server, copy data/osm + data/v2 instead.

    venv/bin/python scripts/v2/download_osm_roads.py --out data/osm/almaty_roads_new.geojson
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

from common import ROOT

BBOX = (43.17, 76.82, 43.38, 77.07)   # = ALMATY_BOUNDS in scraper/config.py (lat_min, lon_min, lat_max, lon_max)
HIGHWAY = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|"
           "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|road")
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]


def hav(a, b):
    (lo1, la1), (lo2, la2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def fetch(bbox):
    s, w, n, e = bbox
    q = f'[out:json][timeout:180];way["highway"~"^({HIGHWAY})$"]({s},{w},{n},{e});out body geom;'
    data = urllib.parse.urlencode({"data": q}).encode()
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "almaty-traffic-research/2"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    return json.load(r)["elements"]
            except Exception as ex:  # noqa: BLE001
                print(f"  {url.split('/')[2]}: {ex}")
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"Overpass failed for {bbox}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "osm" / "almaty_roads_new.geojson")
    ap.add_argument("--split", type=int, default=2, help="NxN quadrants")
    ap.add_argument("--bbox", type=float, nargs=4, default=BBOX, metavar=("S", "W", "N", "E"))
    a = ap.parse_args()
    s, w, n, e = a.bbox
    ways = {}
    for i in range(a.split):
        for j in range(a.split):
            q = (s + (n - s) * i / a.split, w + (e - w) * j / a.split,
                 s + (n - s) * (i + 1) / a.split, w + (e - w) * (j + 1) / a.split)
            els = fetch(q)
            for el in els:
                if el.get("type") == "way" and "geometry" in el:
                    ways[el["id"]] = el
            print(f"quadrant {i},{j}: {len(els)} ways (total {len(ways)})")
    feats = []
    for wid in sorted(ways):
        el = ways[wid]
        t = el.get("tags", {})
        coords = [[g["lon"], g["lat"]] for g in el["geometry"]]
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                      "properties": {"osm_id": wid, "highway": t.get("highway"), "name": t.get("name"),
                                     "oneway": t.get("oneway"), "lanes": t.get("lanes"),
                                     "maxspeed": t.get("maxspeed"),
                                     "length_m": round(sum(hav(p, q) for p, q in zip(coords, coords[1:])), 1),
                                     "nodes": el["nodes"]}})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats}, open(a.out, "w"))
    print(f"saved {len(feats)} ways, {sum(f['properties']['length_m'] for f in feats) / 1000:.0f} km -> {a.out}")


if __name__ == "__main__":
    main()
