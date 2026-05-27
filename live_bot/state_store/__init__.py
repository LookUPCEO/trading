"""State persistence layer.

Modules:
  schema   : sqlite DDL + migrations
  db       : single-connection DAO with context manager
  cache    : parquet-on-disk kline / funding / OI cache
  heartbeat: append-only CSV for the WS collector health signal
"""
from __future__ import annotations

from pathlib import Path

# Project layout (resolved from this file's location).
ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
CACHE_DIR = ROOT / "data" / "cache"
LIQ_DIR = ROOT / "data" / "liquidations"

for p in (STATE_DIR, CACHE_DIR, LIQ_DIR):
    p.mkdir(parents=True, exist_ok=True)

DB_PATH = STATE_DIR / "strategies.sqlite"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.csv"
HALT_PATH = ROOT / "state" / "HALT"
