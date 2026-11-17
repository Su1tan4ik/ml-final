"""
V2 Stage 2 — road graph from OSM ways + A* smoke test.

Splits every OSM way at intersections (OSM nodes shared by >= 2 ways, plus
way endpoints) into edges, then cuts edges longer than MAX_EDGE_M into ~equal
pieces (synthetic nodes get negative ids). Output (data/osm/, not committed):
  graph_nodes.csv   node_id, lat, lon
  graph_edges.geojson  LineString per edge: edge_id, source, target, osm_id,
                       highway, name, oneway (1 forward-only, 0 both), length_m
Later this is what gets imported into PostGIS/pgRouting (road_segment/road_node).

    venv/bin/python scripts/v2/build_graph.py            # build + A* test
"""
from __future__ import annotations

import csv
import heapq
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OSM = ROOT / "data" / "osm"
ROADS = OSM / "almaty_roads.geojson"

# rough free-flow speeds (km/h) — used only for the time-based A* demo
SPEED = {"motorway": 90, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 40,
         "unclassified": 30, "residential": 25, "living_street": 10, "service": 15}


def hav(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371000 * math.asin(math.sqrt(h))


MAX_EDGE_M = 150.0  # long edges are cut so that a traffic score is local (stage 3a TODO)


def split_long(seg, coord, new_id):
    """Cut a node chain into ~equal pieces <= MAX_EDGE_M; inserts synthetic nodes (negative ids)."""
    dense = [seg[0]]
    for a, b in zip(seg, seg[1:]):
        d = hav(*coord[a], *coord[b])
        k = math.ceil(d / (MAX_EDGE_M / 3))
        for j in range(1, k):
            nid = new_id()
            (la1, lo1), (la2, lo2) = coord[a], coord[b]
            coord[nid] = (la1 + (la2 - la1) * j / k, lo1 + (lo2 - lo1) * j / k)
            dense.append(nid)
        dense.append(b)
    cum = [0.0]
    for a, b in zip(dense, dense[1:]):
        cum.append(cum[-1] + hav(*coord[a], *coord[b]))
    n = math.ceil(cum[-1] / MAX_EDGE_M)
    if n <= 1:
        return [seg]
    cuts = [0]
    for k in range(1, n):
        target = cum[-1] * k / n
        i = min(range(cuts[-1] + 1, len(dense) - 1), key=lambda j: abs(cum[j] - target), default=None)
        if i is not None and i > cuts[-1]:
            cuts.append(i)
    cuts.append(len(dense) - 1)
    return [dense[i:j + 1] for i, j in zip(cuts, cuts[1:])]


def parse_int(v):
    """OSM '2', '2;3', '60 mph' -> first integer or None."""
    import re
    m = re.match(r"\s*(\d+)", str(v or ""))
    return int(m.group(1)) if m else None


def oneway_dir(v) -> int:
    v = str(v or "").lower()
    if v in ("yes", "1", "true"):
        return 1
    if v == "-1":
        return -1
    return 0


def build():
    feats = json.load(open(ROADS))["features"]
    use = Counter()
    coord = {}
    for f in feats:
        nodes = f["properties"]["nodes"]
        for nid, (lo, la) in zip(nodes, f["geometry"]["coordinates"]):
            coord[nid] = (la, lo)
        for nid in set(nodes):
            use[nid] += 1
        use[nodes[0]] += 1   # endpoints are always graph vertices
        use[nodes[-1]] += 1

    edges = []
    counter = [0]

    def new_id():
        counter[0] -= 1
        return counter[0]
    for f in feats:
        p = f["properties"]
        nodes = p["nodes"]
        ow = oneway_dir(p.get("oneway"))
        if ow == -1:
            nodes = nodes[::-1]
            ow = 1
        start = 0
        for i in range(1, len(nodes)):
            if use[nodes[i]] >= 2 or i == len(nodes) - 1:
                for seg in split_long(nodes[start:i + 1], coord, new_id):
                    length = sum(hav(*coord[a], *coord[b]) for a, b in zip(seg, seg[1:]))
                    if length > 0:
                        edges.append({"source": seg[0], "target": seg[-1], "osm_id": p["osm_id"],
                                      "highway": p["highway"], "name": p.get("name"),
                                      "oneway": ow, "length_m": round(length, 1),
                                      "lanes": parse_int(p.get("lanes")),
                                      "maxspeed": parse_int(p.get("maxspeed")),
                                      "coords": [[coord[n][1], coord[n][0]] for n in seg]})
                start = i
    used = {e["source"] for e in edges} | {e["target"] for e in edges}

    with open(OSM / "graph_nodes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "lat", "lon"])
        for nid in sorted(used):
            w.writerow([nid, *coord[nid]])
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": e["coords"]},
         "properties": {"edge_id": i, **{k: v for k, v in e.items() if k != "coords"}}}
        for i, e in enumerate(edges)]}
    json.dump(fc, open(OSM / "graph_edges.geojson", "w"))
    return edges, {n: coord[n] for n in used}


def components(edges, nodes):
    """Size of largest weakly connected component."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in edges:
        a, b = find(e["source"]), find(e["target"])
        if a != b:
            parent[a] = b
    sizes = Counter(find(n) for n in nodes)
    return sizes.most_common(1)[0][1], len(sizes)


def astar(adj, coord, src, dst, cost="length"):
    """A* on length (m) or free-flow time (s). Heuristic = straight-line distance (/ max speed)."""
    vmax = max(SPEED.values()) / 3.6
    tla, tlo = coord[dst]

    def h(n):
        d = hav(*coord[n], tla, tlo)
        return d if cost == "length" else d / vmax
    g = {src: 0.0}
    prev = {}
    pq = [(h(src), src)]
    seen = 0
    while pq:
        _, u = heapq.heappop(pq)
        seen += 1
        if u == dst:
            path = [u]
            while u in prev:
                u = prev[u][0]
                path.append(u)
            return g[dst], path[::-1], seen
        for v, e in adj[u]:
            w = e["length_m"] if cost == "length" else e["length_m"] / (SPEED.get(e["highway"].replace("_link", ""), 20) / 3.6)
            if g[u] + w < g.get(v, math.inf):
                g[v] = g[u] + w
                prev[v] = (u, e)
                heapq.heappush(pq, (g[v] + h(v), v))
    return math.inf, [], seen


def nearest(coord, lat, lon, ids):
    return min(ids, key=lambda n: hav(*coord[n], lat, lon))


def main():
    t = time.time()
    edges, coord = build()
    big, ncomp = components(edges, coord)
    print(f"nodes {len(coord):,}  edges {len(edges):,}  build {time.time() - t:.1f}s")
    print(f"weak components {ncomp:,}; largest {big:,} nodes ({100 * big / len(coord):.1f}%)")
    print("oneway edges:", sum(e["oneway"] for e in edges))

    adj = defaultdict(list)
    for e in edges:
        adj[e["source"]].append((e["target"], e))
        if not e["oneway"]:
            adj[e["target"]].append((e["source"], e))
    ids = [n for n in coord if adj[n]]
    # A: Abay x Dostyk area; B: Al-Farabi x Rozybakiev area
    A, B = (43.2400, 76.9570), (43.2050, 76.8960)
    a, b = nearest(coord, *A, ids), nearest(coord, *B, ids)
    for cost in ("length", "time"):
        t = time.time()
        c, path, seen = astar(adj, coord, a, b, cost)
        km = sum(hav(*coord[x], *coord[y]) for x, y in zip(path, path[1:])) / 1000
        cs = f"{c / 1000:.2f} km" if cost == "length" else f"{c / 60:.1f} min free-flow"
        print(f"A* by {cost}: {cs}, route {km:.2f} km, {len(path)} nodes, "
              f"expanded {seen:,}, {1000 * (time.time() - t):.0f} ms")
        json.dump({"type": "Feature", "properties": {"cost": cost},
                   "geometry": {"type": "LineString",
                                "coordinates": [[coord[n][1], coord[n][0]] for n in path]}},
                  open(OSM / f"route_test_{cost}.geojson", "w"))
    print("straight line A-B:", round(hav(*A, *B) / 1000, 2), "km")


if __name__ == "__main__":
    main()
