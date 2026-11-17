-- V2 schema: road graph + traffic observations. Runs once on an empty volume.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgrouting;

-- ---------------------------------------------------------------- graph
-- built by scripts/v2/build_graph.py, loaded by scripts/v2/load_graph_db.py
CREATE TABLE road_node (
    node_id  bigint PRIMARY KEY,          -- OSM node id; negative = synthetic split point
    geom     geometry(Point, 4326) NOT NULL
);
CREATE INDEX road_node_geom ON road_node USING gist (geom);

CREATE TABLE road_edge (
    edge_id      integer PRIMARY KEY,
    source       bigint NOT NULL REFERENCES road_node,
    target       bigint NOT NULL REFERENCES road_node,
    osm_id       bigint,
    highway      text,
    name         text,
    oneway       smallint NOT NULL,       -- 1 = only source->target
    length_m     real NOT NULL,
    lanes        smallint,                 -- OSM lanes (NULL if unknown)
    maxspeed     smallint,                 -- OSM maxspeed km/h (NULL if unknown)
    cost         double precision NOT NULL,   -- pgRouting: source->target
    reverse_cost double precision NOT NULL,   -- pgRouting: target->source, -1 = forbidden
    x1 double precision, y1 double precision, x2 double precision, y2 double precision,  -- for pgr_aStar
    geom         geometry(LineString, 4326) NOT NULL
);
CREATE INDEX road_edge_geom ON road_edge USING gist (geom);
CREATE INDEX road_edge_source ON road_edge (source);
CREATE INDEX road_edge_target ON road_edge (target);

-- ---------------------------------------------------------------- scraping
CREATE TABLE scrape_run (
    run_id      serial PRIMARY KEY,
    ts          timestamptz NOT NULL,        -- slot time (rounded to 30 min)
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    zoom        smallint NOT NULL,
    tiles_total integer,
    tiles_ok    integer,
    tiles_empty integer,
    tiles_err   integer,
    rows_written integer,
    note        text
);
CREATE UNIQUE INDEX scrape_run_ts ON scrape_run (ts, zoom);

-- one row per (time slot, edge, direction) that had coloured pixels
CREATE TABLE traffic_obs (
    ts       timestamptz NOT NULL,
    edge_id  integer NOT NULL REFERENCES road_edge,
    dir      smallint NOT NULL,           -- +1 source->target, -1 reverse, 0 one-way/unknown
    score    real NOT NULL,               -- weighted 1.5 (free) .. 9.5 (jam)
    n_px     smallint NOT NULL,
    n_free smallint, n_slow smallint, n_moderate smallint, n_heavy smallint, n_jam smallint,
    PRIMARY KEY (ts, edge_id, dir)
);
CREATE INDEX traffic_obs_ts_brin ON traffic_obs USING brin (ts);
CREATE INDEX traffic_obs_edge ON traffic_obs (edge_id, dir, ts);

CREATE TABLE weather_obs (
    ts          timestamptz PRIMARY KEY,
    temp_c      real,
    humidity    real,
    wind_ms     real,
    precip_1h   real,
    visibility  real,
    description text
);

-- ---------------------------------------------------------------- model output (stage 4/5)
CREATE TABLE forecast (
    made_at   timestamptz NOT NULL,
    target_ts timestamptz NOT NULL,
    edge_id   integer NOT NULL REFERENCES road_edge,
    dir       smallint NOT NULL,
    score     real NOT NULL,
    p_congested real,
    PRIMARY KEY (target_ts, edge_id, dir, made_at)
);
-- "latest forecast" lookups (max(made_at), WHERE made_at = ...) and the 14-day cleanup in
-- predict_v2 filter by made_at; without this index they scan the whole table
CREATE INDEX forecast_made_at ON forecast (made_at);

-- latest observation per edge/direction (for the map)
CREATE VIEW traffic_now AS
SELECT DISTINCT ON (edge_id, dir) ts, edge_id, dir, score, n_px
FROM traffic_obs
WHERE ts > now() - interval '2 hours'
ORDER BY edge_id, dir, ts DESC;
