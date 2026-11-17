"""Migration 001 (2026-11-17): add road_edge.lanes / maxspeed to an existing DB without touching edge ids."""
import json

from common import OSM
from db import connect

with connect() as conn, conn.cursor() as cur:
    cur.execute("ALTER TABLE road_edge ADD COLUMN IF NOT EXISTS lanes smallint, "
                "ADD COLUMN IF NOT EXISTS maxspeed smallint")
    feats = json.load(open(OSM / "graph_edges.geojson"))["features"]
    cur.execute("CREATE TEMP TABLE tmp_l (edge_id int, source bigint, target bigint, lanes smallint, maxspeed smallint)")
    with cur.copy("COPY tmp_l FROM STDIN") as cp:
        for f in feats:
            p = f["properties"]
            cp.write_row((p["edge_id"], p["source"], p["target"], p.get("lanes"), p.get("maxspeed")))
    cur.execute("SELECT count(*) FROM tmp_l t JOIN road_edge e USING (edge_id) "
                "WHERE e.source <> t.source OR e.target <> t.target")
    if cur.fetchone()[0]:
        raise SystemExit("edge ids do not match the DB graph — aborting")
    cur.execute("UPDATE road_edge e SET lanes = t.lanes, maxspeed = t.maxspeed FROM tmp_l t WHERE t.edge_id = e.edge_id")
    print("updated", cur.rowcount, "edges")
