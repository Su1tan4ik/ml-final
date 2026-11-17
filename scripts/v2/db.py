"""DB connection helper. Settings via env (or .env): PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD."""
from __future__ import annotations

import os

import psycopg

from common import ROOT


def _load_dotenv():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def connect() -> psycopg.Connection:
    _load_dotenv()
    return psycopg.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5442")),
        dbname=os.getenv("PG_DB", "traffic"),
        user=os.getenv("PG_USER", "traffic"),
        password=os.getenv("PG_PASSWORD", "traffic_dev_only"),
    )
