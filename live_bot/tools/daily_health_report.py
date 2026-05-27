"""Daily data-asset health report.

Runs once per day. For each symbol, produces:

  * rows collected  : liquidation events (24h + lifetime),
                      klines / OI / funding rows added (24h).
  * missing minutes : minutes during which the WS collector was NOT "OK"
                      according to heartbeat.csv. REST gaps are reported
                      separately as missing-bars.
  * disconnect_count: number of non-"OK" heartbeat statuses seen in 24h.
  * duplicate_rows  : duplicates still present in any *.parquet file
                      (should always be 0 post-dedupe; an anomaly signal).

Output
------
1. Human-readable summary on stdout.
2. JSON file at state/health/<YYYY-MM-DD>.json.
3. One audit row `daily_health` with the full JSON payload.

Exit 0 always (a cron that pages on non-zero would be too noisy for a
research lab). Call sites that want to alert should read the audit table.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live_bot.state_store import LIQ_DIR, HEARTBEAT_PATH, STATE_DIR, cache             # noqa: E402
from live_bot.state_store.db import open_db                                             # noqa: E402
from live_bot.state_store.heartbeat import HEARTBEAT_INTERVAL_S                         # noqa: E402

log = logging.getLogger("daily_health")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


# =========================================================
# Liquidation parquet aggregation
# =========================================================
def _load_all_liq(symbol: str, liq_dir: Path = LIQ_DIR) -> pd.DataFrame:
    """Read every parquet file for `symbol` and return a single df."""
    sym_dir = liq_dir / symbol
    if not sym_dir.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(sym_dir.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:
            log.warning("unreadable parquet %s: %s", f, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _liq_stats(symbol: str, now: datetime,
               liq_dir: Optional[Path] = None) -> Dict[str, Any]:
    # Late binding so tests can monkeypatch the module-level LIQ_DIR.
    df = _load_all_liq(symbol, liq_dir if liq_dir is not None else LIQ_DIR)
    if df.empty:
        return {
            "symbol": symbol, "total_events": 0, "events_24h": 0,
            "duplicate_rows": 0, "first_ts": None, "last_ts": None,
        }
    total = len(df)
    # Duplicates still present (post-dedupe): should be 0.
    dup_subset = [c for c in ("ts", "side", "qty", "price") if c in df.columns]
    dupes = int(df.duplicated(subset=dup_subset).sum()) if dup_subset else 0
    ts_ms = pd.to_datetime(df["ts"], unit="ms", utc=True) if "ts" in df.columns else None
    cutoff = now - timedelta(hours=24)
    events_24h = int((ts_ms > cutoff).sum()) if ts_ms is not None else 0
    return {
        "symbol": symbol,
        "total_events": total,
        "events_24h": events_24h,
        "duplicate_rows": dupes,
        "first_ts": ts_ms.min().isoformat() if ts_ms is not None and not ts_ms.empty else None,
        "last_ts": ts_ms.max().isoformat() if ts_ms is not None and not ts_ms.empty else None,
    }


# =========================================================
# Heartbeat analysis
# =========================================================
def _heartbeat_in_window(symbol: str, since: datetime,
                         until: datetime,
                         path: Path = HEARTBEAT_PATH) -> Dict[str, Any]:
    """Count per-symbol heartbeats in the time window.

    Every entry represents HEARTBEAT_INTERVAL_S of claimed uptime if status=="OK".
    Non-OK entries are disconnect transitions. Time after the last heartbeat and
    before `until` with no row is considered unreported.
    """
    if not path.exists():
        return {"heartbeats_seen": 0, "ok_count": 0, "disconnect_count": 0,
                "missing_minutes": int((until - since).total_seconds() // 60),
                "unreported_minutes": int((until - since).total_seconds() // 60)}

    ok = 0
    disconnects = 0
    last_ts: Optional[datetime] = None
    earliest: Optional[datetime] = None
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("symbol") != symbol:
                continue
            try:
                ts = datetime.fromisoformat(row["ts"])
            except Exception:
                continue
            if ts < since or ts > until:
                continue
            earliest = ts if earliest is None else min(earliest, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
            if row.get("connection_status") == "OK":
                ok += 1
            else:
                disconnects += 1
    seen = ok + disconnects
    # "Missing minutes" = window duration minus (OK heartbeats × interval).
    interval_min = HEARTBEAT_INTERVAL_S // 60
    covered_min = ok * interval_min
    window_min = int((until - since).total_seconds() // 60)
    missing_min = max(window_min - covered_min, 0)
    unreported_min = max(window_min - seen * interval_min, 0)
    return {
        "heartbeats_seen": seen,
        "ok_count": ok,
        "disconnect_count": disconnects,
        "missing_minutes": missing_min,
        "unreported_minutes": unreported_min,
        "first_heartbeat_in_window": earliest.isoformat() if earliest else None,
        "last_heartbeat_in_window": last_ts.isoformat() if last_ts else None,
    }


# =========================================================
# REST cache rows added in last 24h
# =========================================================
def _rest_rows_24h(symbol: str, interval: str, now: datetime) -> Dict[str, int]:
    out = {}
    cutoff = now - timedelta(hours=24)
    for label, loader in [
        ("klines", lambda: cache.load_klines(symbol, interval)),
        ("funding", lambda: cache.load_funding(symbol)),
        ("oi", lambda: cache.load_oi(symbol, interval)),
    ]:
        df = loader()
        if df.empty or "timestamp" not in df.columns:
            out[label] = 0
            continue
        ts = pd.to_datetime(df["timestamp"], utc=True)
        out[label] = int((ts > cutoff).sum())
    return out


# =========================================================
# Orchestrator
# =========================================================
@dataclass
class SymbolHealth:
    symbol: str
    liq_total_events: int
    liq_events_24h: int
    liq_duplicate_rows: int
    liq_first_ts: Optional[str]
    liq_last_ts: Optional[str]
    hb_heartbeats_seen: int
    hb_ok_count: int
    hb_disconnect_count: int
    hb_missing_minutes: int
    hb_unreported_minutes: int
    rest_klines_24h: int
    rest_funding_24h: int
    rest_oi_24h: int


@dataclass
class DailyHealthReport:
    generated_at: str
    window_start: str
    window_end: str
    per_symbol: List[SymbolHealth] = field(default_factory=list)
    audit_row_count_24h: int = 0
    audit_errors_24h: int = 0


def build_report(symbols: List[str], interval: str, now: datetime) -> DailyHealthReport:
    since = now - timedelta(hours=24)
    per_symbol = []
    for sym in symbols:
        liq = _liq_stats(sym, now)
        hb = _heartbeat_in_window(sym, since, now)
        rest = _rest_rows_24h(sym, interval, now)
        per_symbol.append(SymbolHealth(
            symbol=sym,
            liq_total_events=liq["total_events"],
            liq_events_24h=liq["events_24h"],
            liq_duplicate_rows=liq["duplicate_rows"],
            liq_first_ts=liq["first_ts"],
            liq_last_ts=liq["last_ts"],
            hb_heartbeats_seen=hb["heartbeats_seen"],
            hb_ok_count=hb["ok_count"],
            hb_disconnect_count=hb["disconnect_count"],
            hb_missing_minutes=hb["missing_minutes"],
            hb_unreported_minutes=hb["unreported_minutes"],
            rest_klines_24h=rest["klines"],
            rest_funding_24h=rest["funding"],
            rest_oi_24h=rest["oi"],
        ))

    # Audit activity in last 24h.
    with open_db() as db:
        rows = db.conn.execute(
            "SELECT COUNT(*) AS c FROM audit WHERE ts >= ?", (since.isoformat(),),
        ).fetchone()
        total_audit = int(rows["c"])
        err_rows = db.conn.execute(
            """SELECT COUNT(*) AS c FROM audit
               WHERE ts >= ? AND kind LIKE '%error%'""", (since.isoformat(),),
        ).fetchone()
        err_audit = int(err_rows["c"])

    return DailyHealthReport(
        generated_at=now.isoformat(timespec="seconds"),
        window_start=since.isoformat(timespec="seconds"),
        window_end=now.isoformat(timespec="seconds"),
        per_symbol=per_symbol,
        audit_row_count_24h=total_audit,
        audit_errors_24h=err_audit,
    )


def _report_to_dict(r: DailyHealthReport) -> Dict[str, Any]:
    return {
        "generated_at": r.generated_at,
        "window_start": r.window_start,
        "window_end": r.window_end,
        "per_symbol": [asdict(s) for s in r.per_symbol],
        "audit_row_count_24h": r.audit_row_count_24h,
        "audit_errors_24h": r.audit_errors_24h,
    }


def print_summary(r: DailyHealthReport) -> None:
    print(f"=== DAILY HEALTH  {r.window_start}  →  {r.window_end} ===")
    print(f"{'symbol':<10} {'liq_24h':>8} {'liq_all':>10} {'dupes':>6} "
          f"{'hb_ok':>6} {'disc':>5} {'miss_min':>10} {'klines_24h':>10} "
          f"{'oi_24h':>8} {'funding_24h':>12}")
    for s in r.per_symbol:
        print(f"{s.symbol:<10} {s.liq_events_24h:>8d} {s.liq_total_events:>10d} "
              f"{s.liq_duplicate_rows:>6d} {s.hb_ok_count:>6d} "
              f"{s.hb_disconnect_count:>5d} {s.hb_missing_minutes:>10d} "
              f"{s.rest_klines_24h:>10d} {s.rest_oi_24h:>8d} "
              f"{s.rest_funding_24h:>12d}")
    print(f"audit: {r.audit_row_count_24h} rows / {r.audit_errors_24h} errors in 24h")


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily data-asset health report")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--interval", default="60")
    ap.add_argument("--output-dir",
                    default=str(STATE_DIR / "health"),
                    help="where to write the JSON file")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    report = build_report(args.symbols, args.interval, now)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{now.strftime('%Y-%m-%d')}.json"
    payload = _report_to_dict(report)
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out_path)

    with open_db() as db:
        db.audit("daily_health", payload)

    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
