"""DAO for the state sqlite DB.

Usage:
    with open_db() as db:
        db.register_strategy(name="R1_v1", signal_module="...", signal_fn="...",
                             tp_atr=3.0, sl_atr=1.5, status="PAPER")

All write methods return the row id (or count). Connections are short-lived; the
`with` block guarantees commit or rollback. WAL mode + foreign keys are on.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import DB_PATH
from .schema import init_schema


VALID_STATUSES = ("PAPER", "LIVE_SHADOW", "LIVE_SMALL_CAPITAL", "OFF")


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or datetime.now(timezone.utc)).isoformat(timespec="seconds")


class DB:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---------- strategies ----------
    def register_strategy(self, *, name: str, signal_module: str, signal_fn: str,
                          tp_atr: float, sl_atr: float,
                          status: str = "PAPER",
                          notes: str = "") -> int:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        cur = self.conn.execute(
            """INSERT INTO strategies(name, signal_module, signal_fn, tp_atr, sl_atr,
                                       status, notes)
               VALUES(?,?,?,?,?,?,?)""",
            (name, signal_module, signal_fn, tp_atr, sl_atr, status, notes),
        )
        return int(cur.lastrowid)

    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM strategies WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_strategies(self, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = self.conn.execute(
                f"SELECT * FROM strategies WHERE status IN ({placeholders}) ORDER BY name",
                tuple(statuses),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM strategies ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_status(self, strategy_id: int, new_status: str,
                   reason: str, actor: str = "supervisor",
                   pf_180d: Optional[float] = None,
                   n_trades_180d: Optional[int] = None,
                   paused_by_auto: Optional[bool] = None) -> None:
        if new_status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {new_status}")
        row = self.conn.execute(
            "SELECT status, paused_by_auto FROM strategies WHERE id = ?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown strategy_id: {strategy_id}")
        old_status = row["status"]
        if old_status == new_status and paused_by_auto is None:
            return                               # no-op, don't spam gate_events

        updates = ["status = ?", "last_eval_at = ?"]
        params: List[Any] = [new_status, _iso()]
        if paused_by_auto is not None:
            updates.append("paused_by_auto = ?")
            params.append(1 if paused_by_auto else 0)
        if pf_180d is not None:
            updates.append("last_pf_180d = ?")
            params.append(float(pf_180d))
        params.append(strategy_id)
        self.conn.execute(
            f"UPDATE strategies SET {', '.join(updates)} WHERE id = ?", params
        )

        self.conn.execute(
            """INSERT INTO gate_events(strategy_id, from_status, to_status, reason,
                                        pf_180d, n_trades_180d, actor)
               VALUES(?,?,?,?,?,?,?)""",
            (strategy_id, old_status, new_status, reason,
             pf_180d, n_trades_180d, actor),
        )

    def update_eval_metrics(self, strategy_id: int, pf_180d: float,
                            last_change_point_at: Optional[str]) -> None:
        self.conn.execute(
            """UPDATE strategies
               SET last_pf_180d = ?, last_change_point_at = ?, last_eval_at = ?
               WHERE id = ?""",
            (pf_180d, last_change_point_at, _iso(), strategy_id),
        )

    # ---------- trade_log ----------
    def insert_trade(self, *, strategy_id: int, mode: str,
                     ts_entry: str, ts_exit: Optional[str],
                     side: str, entry: float, exit: Optional[float],
                     qty: float, pnl: Optional[float], reason: Optional[str],
                     session: Optional[str] = None,
                     regime_tag: Optional[str] = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO trade_log(strategy_id, mode, ts_entry, ts_exit,
                                     side, entry, exit, qty, pnl, reason,
                                     session, regime_tag)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (strategy_id, mode, ts_entry, ts_exit, side, entry, exit, qty,
             pnl, reason, session, regime_tag),
        )
        return int(cur.lastrowid)

    def trades_since(self, strategy_id: int, since_iso: str,
                     modes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if modes:
            placeholders = ",".join("?" for _ in modes)
            rows = self.conn.execute(
                f"""SELECT * FROM trade_log
                    WHERE strategy_id = ? AND ts_entry >= ?
                      AND mode IN ({placeholders})
                    ORDER BY ts_entry""",
                (strategy_id, since_iso, *modes),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM trade_log
                   WHERE strategy_id = ? AND ts_entry >= ?
                   ORDER BY ts_entry""",
                (strategy_id, since_iso),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- audit ----------
    def audit(self, kind: str, payload: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO audit(kind, payload) VALUES(?, ?)",
            (kind, json.dumps(payload, default=str)),
        )

    # ---------- gate events ----------
    def recent_gate_events(self, strategy_id: Optional[int] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        if strategy_id:
            rows = self.conn.execute(
                "SELECT * FROM gate_events WHERE strategy_id = ? ORDER BY ts DESC LIMIT ?",
                (strategy_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM gate_events ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


@contextmanager
def open_db(path: Path = DB_PATH) -> Iterator[DB]:
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    conn.execute("BEGIN")
    try:
        yield DB(conn)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
