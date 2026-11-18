"""
V2 API + map (MapLibre). Run from scripts/v2:
    ../../venv/bin/uvicorn api:app --port 8002
Endpoints:
  GET /                      map page (web/index.html)
  GET /api/slots             available time slots (latest first)
  GET /api/traffic?ts=&h=   GeoJSON of (edge, direction) scores for a slot (default latest);
                             h>0 = model forecast h*30 min ahead made at that slot
  GET /api/route?a_lat&a_lon&b_lat&b_lon[&ts][&h]
                             shortest-by-length vs fastest-with-traffic routes (pgRouting Dijkstra)
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from common import ROOT
from db import connect
from features_v2 import GOOD_SLOT_SHARE, good_slots_sql

WEB = Path(__file__).resolve().parent / "web"
MODELS = Path(os.environ.get("MODELS_DIR", ROOT / "models_v2"))
STATS_DAYS = 7                # dashboard aggregates use the last N days
CONGESTED = 5.5
LOCAL = "interval '5 hours'"  # Almaty = UTC+5 (same convention as features_v2)
app = FastAPI(title="Almaty traffic V2")


def _slot(cur, ts: datetime | None, h: int = 0):
    """Default slot: the latest observation (h=0) or, for a forecast (h>0), the latest
    slot that already HAS a forecast. predict_v2 runs a few minutes after the scrape (longer
    after a retrain), so the newest observed slot may not be forecast yet; if predict fails,
    the map keeps the previous forecast instead of going empty."""
    if ts is None and h > 0:
        cur.execute("SELECT max(made_at) FROM forecast")
        ts = cur.fetchone()[0]
    if ts is None:
        cur.execute("SELECT max(ts) FROM traffic_obs")
        ts = cur.fetchone()[0]
    return ts


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/slots")
def slots(limit: int = Query(96, le=2000)):
    with connect() as c:
        rows = c.execute("""SELECT r.ts, r.tiles_ok, r.tiles_total, r.rows_written, r.note
                            FROM scrape_run r WHERE r.finished_at IS NOT NULL
                            ORDER BY r.ts DESC LIMIT %s""", (limit,)).fetchall()
    return [{"ts": r[0].isoformat(), "tiles_ok": r[1], "tiles_total": r[2], "rows": r[3], "note": r[4]}
            for r in rows]


def _scores_sql(ts: datetime, h: int) -> str:
    """Subquery (edge_id, dir, score): observations (h=0) or forecast made at ts for ts+h*30min.
    Values are server-side (ts from DB, h validated int) — not user strings."""
    if h == 0:
        return f"SELECT edge_id, dir, score FROM traffic_obs WHERE ts = '{ts.isoformat()}'"
    return (f"SELECT edge_id, dir, score FROM forecast WHERE made_at = '{ts.isoformat()}' "
            f"AND target_ts = '{ts.isoformat()}'::timestamptz + interval '{30 * int(h)} minutes'")


@app.get("/api/horizons")
def horizons():
    with connect() as c:
        ts = c.execute("SELECT max(made_at) FROM forecast").fetchone()[0]
        if ts is None:
            return {"made_at": None, "horizons": [0]}
        hs = c.execute("SELECT DISTINCT (extract(epoch FROM target_ts - made_at) / 1800)::int "
                       "FROM forecast WHERE made_at = %s ORDER BY 1", (ts,)).fetchall()
    return {"made_at": ts.isoformat(), "horizons": [0] + [r[0] for r in hs]}


@app.get("/api/traffic")
def traffic(ts: datetime | None = None, h: int = Query(0, ge=0, le=12)):
    with connect() as c, c.cursor() as cur:
        ts = _slot(cur, ts, h)
        if ts is None:
            return {"type": "FeatureCollection", "features": [], "ts": None}
        src = _scores_sql(ts, h)
        # dir -1 geometry is reversed so the map can always draw the line to the RIGHT of travel
        cur.execute("""
            SELECT json_build_object('type','FeatureCollection','ts',%s::text,'h',%s,'features',
                   coalesce(json_agg(json_build_object(
                     'type','Feature',
                     'geometry', ST_AsGeoJSON(CASE WHEN o.dir = -1 THEN ST_Reverse(e.geom) ELSE e.geom END, 6)::json,
                     'properties', json_build_object('edge_id', e.edge_id, 'dir', o.dir,
                                   'score', round(o.score::numeric, 2),
                                   'name', e.name, 'highway', e.highway))), '[]'::json))
            FROM (""" + src + """) o JOIN road_edge e USING (edge_id)""", (ts, h))
        return cur.fetchone()[0]


ROUTE_EDGES_SQL = """
SELECT e.edge_id AS id, e.source, e.target,
  CASE WHEN '{mode}' = 'length' THEN e.cost
       ELSE e.length_m / (free_speed_ms(e.highway) * congestion_factor(coalesce(f.score, b.score))) END AS cost,
  CASE WHEN e.reverse_cost < 0 THEN -1
       WHEN '{mode}' = 'length' THEN e.reverse_cost
       ELSE e.length_m / (free_speed_ms(e.highway) * congestion_factor(coalesce(r.score, b.score))) END AS reverse_cost
FROM road_edge e
LEFT JOIN ({src}) f ON f.edge_id = e.edge_id AND f.dir = 1
LEFT JOIN ({src}) r ON r.edge_id = e.edge_id AND r.dir = -1
LEFT JOIN ({src}) b ON b.edge_id = e.edge_id AND b.dir = 0
"""

NEAREST_SQL = """SELECT node_id FROM road_node
                 WHERE node_id IN (SELECT source FROM road_edge UNION SELECT target FROM road_edge)
                 ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326) LIMIT 1"""


def _route(cur, a, b, ts, mode, h=0):
    src = _scores_sql(ts, h) if ts else "SELECT 0 AS edge_id, 0 AS dir, 1.5::real AS score WHERE false"
    edges_sql = ROUTE_EDGES_SQL.format(mode=mode, src=src)
    cur.execute("""
        WITH p AS (SELECT * FROM pgr_dijkstra(%s, %s::bigint, %s::bigint, directed => true)),
        seg AS (
          SELECT p.seq, e.length_m, e.highway,
                 -- travel time with traffic on the direction actually used
                 e.length_m / (free_speed_ms(e.highway) * congestion_factor(coalesce(
                     CASE WHEN p.node = e.source THEN f.score ELSE r.score END, bb.score))) AS t_s,
                 e.length_m / free_speed_ms(e.highway) AS t_free,
                 CASE WHEN p.node = e.source THEN e.geom ELSE ST_Reverse(e.geom) END AS g
          FROM p JOIN road_edge e ON e.edge_id = p.edge
          LEFT JOIN (""" + src + """) f  ON f.edge_id = e.edge_id AND f.dir = 1
          LEFT JOIN (""" + src + """) r  ON r.edge_id = e.edge_id AND r.dir = -1
          LEFT JOIN (""" + src + """) bb ON bb.edge_id = e.edge_id AND bb.dir = 0)
        SELECT ST_AsGeoJSON(ST_LineMerge(ST_Collect(g ORDER BY seq)), 6), sum(length_m), sum(t_s), sum(t_free), count(*)
        FROM seg""", (edges_sql, a, b))
    g, length, t, t_free, n = cur.fetchone()
    if not g:
        return None
    return {"geometry": json.loads(g), "length_km": round(length / 1000, 2),
            "time_min": round(t / 60, 1), "time_free_min": round(t_free / 60, 1), "edges": n}


@app.get("/api/route")
def route(a_lat: float, a_lon: float, b_lat: float, b_lon: float, ts: datetime | None = None,
          h: int = Query(0, ge=0, le=12)):
    with connect() as c, c.cursor() as cur:
        ts = _slot(cur, ts, h)
        cur.execute(NEAREST_SQL, (a_lon, a_lat)); a = cur.fetchone()[0]
        cur.execute(NEAREST_SQL, (b_lon, b_lat)); b = cur.fetchone()[0]
        out = {"ts": ts.isoformat() if ts else None, "h": h,
               "shortest": _route(cur, a, b, ts, "length", h),
               "fastest": _route(cur, a, b, ts, "time", h)}
    if not out["shortest"]:
        raise HTTPException(404, "no route between these points")
    return out


# ---------------------------------------------------------------- dashboard tabs
_cache: dict = {}
_lock = threading.Lock()


def _cached(key: str, fn):
    """Recompute only when a new scrape slot appears (aggregates over millions of rows)."""
    with connect() as c:
        ver = c.execute("SELECT max(ts) FROM scrape_run WHERE finished_at IS NOT NULL").fetchone()[0]
    hit = _cache.get(key)
    if hit and hit[0] == ver:
        return hit[1]
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == ver:
            return hit[1]
        val = fn()
        _cache[key] = (ver, val)
        return val


def _q(sql, params=None):
    with connect() as c, c.cursor() as cur:
        cur.execute(sql, params)          # params=None -> no %-placeholder parsing (LIKE 'x%')
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _json_file(name):
    f = MODELS / name
    return json.load(open(f)) if f.exists() else None


# last STATS_DAYS days, only slots whose scrape fetched >= GOOD_SLOT_SHARE of tiles
# (a partial run misses the outskirts and would make city-wide averages jump)
WINDOW = (f"o.ts > (SELECT max(ts) FROM traffic_obs) - interval '{STATS_DAYS} days' "
          f"AND o.ts IN ({good_slots_sql(GOOD_SLOT_SHARE)})")
ROAD_CLASS = """CASE WHEN e.highway LIKE 'motorway%' OR e.highway LIKE 'trunk%' OR e.highway LIKE 'primary%' THEN 'магистрали'
                     WHEN e.highway LIKE 'secondary%' THEN 'районные'
                     WHEN e.highway LIKE 'tertiary%' OR e.highway = 'unclassified' THEN 'местные'
                     ELSE 'жилые улицы' END"""


@app.get("/api/summary")
def summary():
    def f():
        r = _q(f"""SELECT count(*) AS slots, min(ts) AS first, max(ts) AS last, sum(rows_written) AS obs,
                          round(sum(tiles_ok + tiles_empty)::numeric / nullif(sum(tiles_total), 0) * 100, 1) AS tile_ok_pct
                   FROM scrape_run WHERE finished_at IS NOT NULL""")[0]
        now = _q(f"""SELECT count(*) AS units, count(DISTINCT o.edge_id) AS edges,
                            round((SELECT sum(length_m) FROM road_edge WHERE edge_id IN
                                   (SELECT edge_id FROM traffic_obs WHERE ts = (SELECT max(ts) FROM traffic_obs)))::numeric / 1000) AS km,
                            round((sum(e.length_m * o.score) / sum(e.length_m))::numeric, 2) AS mean_score,
                            round((sum(e.length_m * (o.score >= {CONGESTED})::int) / sum(e.length_m) * 100)::numeric, 1) AS congested_pct
                     FROM traffic_obs o JOIN road_edge e USING (edge_id)
                     WHERE o.ts = (SELECT max(ts) FROM traffic_obs)""")[0]
        g = _q("SELECT count(*) AS edges, round(sum(length_m)::numeric / 1000) AS km FROM road_edge")[0]
        w = _q("SELECT count(*) AS n FROM weather_obs")[0]
        rep = _json_file("report.json")
        return {"slots": r["slots"], "first": r["first"], "last": r["last"], "obs_total": r["obs"],
                "tile_ok_pct": r["tile_ok_pct"], "hours_covered": r["slots"] / 2,
                "graph": g, "now": now, "weather_obs": w["n"],
                "model_trained_at": rep and rep.get("trained_at"),
                "model_slots": rep and rep["data"].get("slots_with_data")}
    return _cached("summary", f)


@app.get("/api/timeline")
def timeline():
    """City-wide length-weighted score and congested share per slot (all classes + by class)."""
    def f():
        return _q(f"""SELECT o.ts, {ROAD_CLASS} AS cls,
                             round((sum(e.length_m * o.score) / sum(e.length_m))::numeric, 3) AS mean_score,
                             round((sum(e.length_m * (o.score >= {CONGESTED})::int) / sum(e.length_m) * 100)::numeric, 2) AS congested_pct
                      FROM traffic_obs o JOIN road_edge e USING (edge_id) WHERE {WINDOW}
                      GROUP BY GROUPING SETS ((o.ts), (o.ts, cls)) ORDER BY o.ts""")
    return _cached("timeline", f)


@app.get("/api/hourly")
def hourly():
    """Almaty hour-of-day and day-of-week profiles over the last STATS_DAYS days."""
    def f():
        by_hour = _q(f"""SELECT extract(hour FROM o.ts + {LOCAL})::int AS hour,
                                round((sum(e.length_m * o.score) / sum(e.length_m))::numeric, 3) AS mean_score,
                                round((sum(e.length_m * (o.score >= {CONGESTED})::int) / sum(e.length_m) * 100)::numeric, 2) AS congested_pct,
                                count(DISTINCT o.ts) AS slots
                         FROM traffic_obs o JOIN road_edge e USING (edge_id) WHERE {WINDOW}
                         GROUP BY 1 ORDER BY 1""")
        by_dow = _q(f"""SELECT extract(isodow FROM o.ts + {LOCAL})::int AS dow,
                               round((sum(e.length_m * o.score) / sum(e.length_m))::numeric, 3) AS mean_score,
                               round((sum(e.length_m * (o.score >= {CONGESTED})::int) / sum(e.length_m) * 100)::numeric, 2) AS congested_pct,
                               count(DISTINCT o.ts) AS slots
                        FROM traffic_obs o JOIN road_edge e USING (edge_id) WHERE {WINDOW}
                        GROUP BY 1 ORDER BY 1""")
        return {"by_hour": by_hour, "by_dow": by_dow}
    return _cached("hourly", f)


@app.get("/api/hotspots")
def hotspots(limit: int = Query(20, le=100)):
    """Most congested named streets (length-weighted, >= 300 m observed) + score in the latest slot."""
    def f():
        return _q(f"""WITH s AS (
                        SELECT e.name, sum(e.length_m * o.score) / sum(e.length_m) AS mean_score,
                               sum(e.length_m * (o.score >= {CONGESTED})::int) / sum(e.length_m) AS cong,
                               max(o.score) AS max_score, count(DISTINCT o.edge_id) AS edges,
                               sum(e.length_m) / count(DISTINCT o.ts) AS len_per_slot
                        FROM traffic_obs o JOIN road_edge e USING (edge_id)
                        WHERE {WINDOW} AND e.name IS NOT NULL GROUP BY e.name),
                      n AS (
                        SELECT e.name, sum(e.length_m * o.score) / sum(e.length_m) AS now_score
                        FROM traffic_obs o JOIN road_edge e USING (edge_id)
                        WHERE o.ts = (SELECT max(ts) FROM traffic_obs) AND e.name IS NOT NULL GROUP BY e.name)
                      SELECT s.name, round(s.mean_score::numeric, 2) AS mean_score,
                             round((s.cong * 100)::numeric, 1) AS congested_pct, round(s.max_score::numeric, 1) AS max_score,
                             s.edges, round((s.len_per_slot / 1000)::numeric, 2) AS km, round(n.now_score::numeric, 2) AS now_score
                      FROM s LEFT JOIN n USING (name) WHERE s.len_per_slot >= 300
                      ORDER BY s.mean_score DESC LIMIT 100""")
    return _cached("hotspots", f)[:limit]


@app.get("/api/collection")
def collection():
    return _q("""SELECT ts, tiles_total, tiles_ok, tiles_empty, tiles_err, rows_written, note,
                        round(extract(epoch FROM finished_at - started_at)::numeric) AS seconds
                 FROM scrape_run ORDER BY ts DESC LIMIT 2000""")


@app.get("/api/report")
def report():
    return {"report": _json_file("report.json"), "meta": _json_file("meta.json"),
            "serving": _json_file("serving.json")}   # per horizon: model or naive forecast on the map


@app.get("/api/recommendations")
def recommendations():
    r = _json_file("recommendations.json")
    if r is None:
        raise HTTPException(404, "recommendations.json not built yet (recommend_v2.py)")
    return r


@app.get("/api/edge")
def edge(edge_id: int, dir: int | None = None):
    """Street segment card: static info, observed history (last 7 days) and latest forecasts."""
    info = _q("""SELECT edge_id, name, highway, oneway, round(length_m::numeric) AS length_m, lanes, maxspeed
                 FROM road_edge WHERE edge_id = %s""", (edge_id,))
    if not info:
        raise HTTPException(404, "no such edge")
    d = "" if dir is None else "AND dir = %s"
    p = (edge_id,) if dir is None else (edge_id, dir)
    hist = _q(f"""SELECT ts, dir, round(score::numeric, 2) AS score FROM traffic_obs
                  WHERE edge_id = %s {d} AND ts > (SELECT max(ts) FROM traffic_obs) - interval '{STATS_DAYS} days'
                  ORDER BY ts""", p)
    fc = _q(f"""SELECT made_at, target_ts, dir, round(score::numeric, 2) AS score,
                       round(p_congested::numeric, 3) AS p_congested FROM forecast
                WHERE edge_id = %s {d} AND made_at = (SELECT max(made_at) FROM forecast)
                ORDER BY target_ts""", p)
    return {"edge": info[0], "history": hist, "forecast": fc}


@app.get("/api/street")
def street(name: str):
    """Whole street: length-weighted observed score per slot + latest model forecast per horizon."""
    hist = _q(f"""SELECT o.ts, round((sum(e.length_m * o.score) / sum(e.length_m))::numeric, 2) AS score,
                         round((sum(e.length_m * (o.score >= {CONGESTED})::int) / sum(e.length_m) * 100)::numeric, 1) AS congested_pct
                  FROM traffic_obs o JOIN road_edge e USING (edge_id)
                  WHERE e.name = %s AND o.ts > (SELECT max(ts) FROM traffic_obs) - interval '{STATS_DAYS} days'
                  GROUP BY o.ts ORDER BY o.ts""", (name,))
    if not hist:
        raise HTTPException(404, "no observations for this street")
    fc = _q("""SELECT f.made_at, f.target_ts, round((sum(e.length_m * f.score) / sum(e.length_m))::numeric, 2) AS score,
                      round(avg(f.p_congested)::numeric, 3) AS p_congested, count(*) AS units
               FROM forecast f JOIN road_edge e USING (edge_id)
               WHERE e.name = %s AND f.made_at = (SELECT max(made_at) FROM forecast)
               GROUP BY f.made_at, f.target_ts ORDER BY f.target_ts""", (name,))
    info = _q("""SELECT count(*) AS edges, round(sum(length_m)::numeric) AS length_m,
                        mode() WITHIN GROUP (ORDER BY highway) AS highway,
                        percentile_disc(0.5) WITHIN GROUP (ORDER BY lanes) AS lanes,
                        round(avg(ST_Y(ST_Centroid(geom)))::numeric, 6) AS lat,
                        round(avg(ST_X(ST_Centroid(geom)))::numeric, 6) AS lon
                 FROM road_edge WHERE name = %s""", (name,))[0]
    return {"name": name, "info": info, "history": hist, "forecast": fc}
