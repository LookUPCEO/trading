"""SQLite schema + migrations for the state layer.

Tables
------
strategies    : one row per registered strategy, the source of truth for status.
trade_log     : every simulated / paper / live trade ever produced.
gate_events   : every state transition decided by the supervisor.
audit         : generic key/value events (data audits, kill switches, manual promotions).

Migrations are additive-only. Downgrades are not supported; a bad deploy is rolled
forward with another migration.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

DDL = [
    # ---- strategies ----
    """
    CREATE TABLE IF NOT EXISTS strategies (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        name                TEXT NOT NULL UNIQUE,
        signal_module       TEXT NOT NULL,
        signal_fn           TEXT NOT NULL,
        tp_atr              REAL NOT NULL,
        sl_atr              REAL NOT NULL,
        status              TEXT NOT NULL CHECK (status IN
                              ('PAPER','LIVE_SHADOW','LIVE_SMALL_CAPITAL','OFF')),
        paused_by_auto      INTEGER NOT NULL DEFAULT 0,
        last_eval_at        TEXT,
        last_pf_180d        REAL,
        last_change_point_at TEXT,
        notes               TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_strategies_status ON strategies(status)",

    # ---- trade_log ----
    """
    CREATE TABLE IF NOT EXISTS trade_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id     INTEGER NOT NULL,
        mode            TEXT NOT NULL CHECK (mode IN ('backtest','paper','live')),
        ts_entry        TEXT NOT NULL,
        ts_exit         TEXT,
        side            TEXT NOT NULL CHECK (side IN ('long','short')),
        entry           REAL NOT NULL,
        exit            REAL,
        qty             REAL NOT NULL,
        pnl             REAL,
        reason          TEXT,
        session         TEXT,
        regime_tag      TEXT,
        FOREIGN KEY (strategy_id) REFERENCES strategies(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_trade_log_strat_ts ON trade_log(strategy_id, ts_entry)",
    "CREATE INDEX IF NOT EXISTS ix_trade_log_mode ON trade_log(mode)",

    # ---- gate_events ----
    """
    CREATE TABLE IF NOT EXISTS gate_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id     INTEGER NOT NULL,
        ts              TEXT NOT NULL DEFAULT (datetime('now')),
        from_status     TEXT NOT NULL,
        to_status       TEXT NOT NULL,
        reason          TEXT NOT NULL,
        pf_180d         REAL,
        n_trades_180d   INTEGER,
        actor           TEXT NOT NULL DEFAULT 'supervisor',
        FOREIGN KEY (strategy_id) REFERENCES strategies(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_gate_events_strat ON gate_events(strategy_id, ts)",

    # ---- audit ----
    """
    CREATE TABLE IF NOT EXISTS audit (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT NOT NULL DEFAULT (datetime('now')),
        kind            TEXT NOT NULL,
        payload         TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_audit_kind_ts ON audit(kind, ts)",

    # ---- meta ----
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key             TEXT PRIMARY KEY,
        value           TEXT NOT NULL
    )
    """,
]


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if missing and record schema version. Idempotent."""
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    for stmt in DDL:
        cur.execute(stmt)
    cur.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
