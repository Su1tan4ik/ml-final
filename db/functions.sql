-- Helpers for traffic-aware routing. Idempotent: psql -f db/functions.sql
-- Free-flow speed (m/s) by OSM road class (rough urban values).
CREATE OR REPLACE FUNCTION free_speed_ms(highway text) RETURNS double precision
LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE replace(highway, '_link', '')
    WHEN 'motorway' THEN 90 WHEN 'trunk' THEN 80 WHEN 'primary' THEN 60
    WHEN 'secondary' THEN 50 WHEN 'tertiary' THEN 40 WHEN 'unclassified' THEN 30
    WHEN 'residential' THEN 25 WHEN 'service' THEN 15 WHEN 'living_street' THEN 10
    ELSE 20 END / 3.6
$$;

-- Share of free-flow speed for a traffic score (1.5 free .. 9.5 jam), linear between anchors.
CREATE OR REPLACE FUNCTION congestion_factor(score double precision) RETURNS double precision
LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE
    WHEN score IS NULL OR score <= 1.5 THEN 1.0
    WHEN score <= 3.5 THEN 1.0  - (score - 1.5) / 2 * 0.40   -- 1.00 -> 0.60
    WHEN score <= 5.5 THEN 0.60 - (score - 3.5) / 2 * 0.20   -- 0.60 -> 0.40
    WHEN score <= 7.5 THEN 0.40 - (score - 5.5) / 2 * 0.15   -- 0.40 -> 0.25
    ELSE greatest(0.10, 0.25 - (score - 7.5) / 2 * 0.13)     -- 0.25 -> 0.12
  END
$$;
