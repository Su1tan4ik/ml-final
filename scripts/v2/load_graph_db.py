"""
Load the road graph (build_graph.py output) into PostGIS: road_node, road_edge.
Replaces existing graph rows ONLY if traffic_obs is empty (edge ids must stay stable once
observations reference them).

    venv/bin/python scripts/v2/load_graph_db.py
"""
from __future__ import annotations

import csv
import json
import time

from common import OSM
from db import connect


def main():
    t0 = time.time()
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM traffic_obs")
        if cur.fetchone()[0]:
            raise SystemExit("traffic_obs is not empty — refusing to replace the graph")
        cur.execute("TRUNCATE road_edge, road_node CASCADE")

        with cur.copy("COPY road_node (node_id, geom) FROM STDIN") as cp:
            for r in csv.DictReader(open(OSM / "graph_nodes.csv")):
                cp.write_row((int(r["node_id"]), f"SRID=4326;POINT({r['lon']} {r['lat']})"))

        feats = json.load(open(OSM / "graph_edges.geojson"))["features"]
        with cur.copy("COPY road_edge (edge_id, source, target, osm_id, highway, name, oneway, "
                      "length_m, lanes, maxspeed, cost, reverse_cost, x1, y1, x2, y2, geom) FROM STDIN") as cp:
            for f in feats:
                p = f["properties"]
                c = f["geometry"]["coordinates"]
                wkt = "SRID=4326;LINESTRING(" + ",".join(f"{x} {y}" for x, y in c) + ")"
                L = p["length_m"]
                cp.write_row((p["edge_id"], p["source"], p["target"], p["osm_id"], p["highway"],
                              p["name"], p["oneway"], L, p.get("lanes"), p.get("maxspeed"), L, -1 if p["oneway"] else L,
                              c[0][0], c[0][1], c[-1][0], c[-1][1], wkt))
        cur.execute("ANALYZE road_node; ANALYZE road_edge;")
        cur.execute("SELECT (SELECT count(*) FROM road_node), (SELECT count(*) FROM road_edge), "
                    "(SELECT round(sum(length_m)/1000) FROM road_edge)")
        n, e, km = cur.fetchone()
    print(f"loaded {n:,} nodes, {e:,} edges, {km:,} km in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
