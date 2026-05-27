"""Append-only heartbeat log for the WS collectors.

Writers (tools/ws_liquidations.py) append one line every HEARTBEAT_INTERVAL_S
with (iso_ts, symbol, events_since_last, connection_status). Readers (the
supervisor, the dashboard) look at the tail and alert if the newest row is
older than STALE_SECONDS.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import HEARTBEAT_PATH

HEARTBEAT_INTERVAL_S = 300
STALE_SECONDS = 1_800                    # 30 min


def write(symbol: str, events_since_last: int, connection_status: str,
          path: Path = HEARTBEAT_PATH) -> None:
    """Atomic line append. Safe under concurrent writers thanks to O_APPEND semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "symbol", "events_since_last", "connection_status"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    symbol, events_since_last, connection_status])


def tail_last_per_symbol(path: Path = HEARTBEAT_PATH) -> Dict[str, Dict]:
    """Read the file and return the most-recent entry per symbol."""
    if not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["symbol"]] = row
    return out


def stale_collectors(symbols: List[str], now: Optional[datetime] = None,
                     path: Path = HEARTBEAT_PATH) -> List[str]:
    """Return the subset of `symbols` whose heartbeat is either missing or stale."""
    now = now or datetime.now(timezone.utc)
    last = tail_last_per_symbol(path)
    stale: List[str] = []
    for s in symbols:
        row = last.get(s)
        if row is None:
            stale.append(s)
            continue
        ts = datetime.fromisoformat(row["ts"])
        if (now - ts).total_seconds() > STALE_SECONDS:
            stale.append(s)
    return stale
