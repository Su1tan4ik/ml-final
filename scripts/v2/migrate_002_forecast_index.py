"""Migration 002 (2026-11-17): index forecast(made_at) on an existing DB (new DBs get it from db/schema.sql).

    docker compose exec collector python migrate_002_forecast_index.py
"""
from db import connect

with connect() as conn:
    conn.execute("CREATE INDEX IF NOT EXISTS forecast_made_at ON forecast (made_at)")
    conn.execute("ANALYZE forecast")
    print("forecast_made_at: ok")
