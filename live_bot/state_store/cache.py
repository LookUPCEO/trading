"""Parquet-on-disk cache for klines / funding / open-interest.

Layout (rooted at data/cache/):
    klines/<SYMBOL>_<INTERVAL>.parquet
    funding/<SYMBOL>.parquet
    oi/<SYMBOL>_<INTERVAL>.parquet

Each file stores all historical rows, de-duped on 'timestamp', sorted ascending.
The cache is append-only: never delete rows, only upsert. This means a crashed
mid-write can be recovered by re-running `cache_refresh` — idempotent.

Concurrency: a writer lockfile prevents two processes from appending at once.
Readers are always safe (pandas.read_parquet is atomic on fully-written files).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import CACHE_DIR

log = logging.getLogger(__name__)

KLINES_DIR = CACHE_DIR / "klines"
FUNDING_DIR = CACHE_DIR / "funding"
OI_DIR = CACHE_DIR / "oi"
for p in (KLINES_DIR, FUNDING_DIR, OI_DIR):
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# Lockfile (per cache file)
# =========================================================
class _FileLock:
    def __init__(self, path: Path, stale_seconds: int = 300):
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.stale = stale_seconds

    def __enter__(self):
        # Reap stale locks (crashed writer).
        if self.lock_path.exists():
            age = time.time() - self.lock_path.stat().st_mtime
            if age > self.stale:
                log.warning("removing stale lock (%ds old): %s", int(age), self.lock_path)
                self.lock_path.unlink(missing_ok=True)
        # Acquire.
        for _ in range(60):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                time.sleep(0.5)
        raise TimeoutError(f"could not acquire {self.lock_path} after 30s")

    def __exit__(self, *_):
        self.lock_path.unlink(missing_ok=True)


# =========================================================
# Generic read/upsert
# =========================================================
def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _upsert(path: Path, new: pd.DataFrame, key: str = "timestamp") -> pd.DataFrame:
    with _FileLock(path):
        existing = _read(path)
        combined = (pd.concat([existing, new], ignore_index=True)
                    .drop_duplicates(subset=[key])
                    .sort_values(key)
                    .reset_index(drop=True))
        tmp = path.with_suffix(path.suffix + ".tmp")
        combined.to_parquet(tmp, index=False)
        os.replace(tmp, path)       # atomic rename
    return combined


# =========================================================
# Public API
# =========================================================
def kline_path(symbol: str, interval: str) -> Path:
    return KLINES_DIR / f"{symbol}_{interval}.parquet"


def funding_path(symbol: str) -> Path:
    return FUNDING_DIR / f"{symbol}.parquet"


def oi_path(symbol: str, interval: str) -> Path:
    return OI_DIR / f"{symbol}_{interval}.parquet"


def load_klines(symbol: str, interval: str) -> pd.DataFrame:
    return _read(kline_path(symbol, interval))


def load_funding(symbol: str) -> pd.DataFrame:
    return _read(funding_path(symbol))


def load_oi(symbol: str, interval: str) -> pd.DataFrame:
    return _read(oi_path(symbol, interval))


def upsert_klines(symbol: str, interval: str, new: pd.DataFrame) -> pd.DataFrame:
    return _upsert(kline_path(symbol, interval), new)


def upsert_funding(symbol: str, new: pd.DataFrame) -> pd.DataFrame:
    return _upsert(funding_path(symbol), new)


def upsert_oi(symbol: str, interval: str, new: pd.DataFrame) -> pd.DataFrame:
    return _upsert(oi_path(symbol, interval), new)


def last_timestamp_ms(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    ts = df["timestamp"].iloc[-1]
    return int(pd.Timestamp(ts).value // 1_000_000)


# =========================================================
# Refresh helper: given a fetch function and a cache loader, fetch only the
# delta since the last cached bar, then upsert.
# =========================================================
def incremental_refresh(load_fn: Callable[[], pd.DataFrame],
                        fetch_fn: Callable[[int, int], pd.DataFrame],
                        upsert_fn: Callable[[pd.DataFrame], pd.DataFrame],
                        default_start_ms: int,
                        now_ms: int) -> dict:
    """Generic delta-refresh. Returns stats dict."""
    existing = load_fn()
    last_ms = last_timestamp_ms(existing)
    start_ms = (last_ms + 1) if last_ms is not None else default_start_ms
    if start_ms >= now_ms:
        return {"rows_before": len(existing), "rows_added": 0,
                "start_ms": start_ms, "end_ms": now_ms}
    new = fetch_fn(start_ms, now_ms)
    if new is None or new.empty:
        return {"rows_before": len(existing), "rows_added": 0,
                "start_ms": start_ms, "end_ms": now_ms}
    combined = upsert_fn(new)
    return {
        "rows_before": len(existing),
        "rows_added": len(combined) - len(existing),
        "rows_after": len(combined),
        "start_ms": start_ms,
        "end_ms": now_ms,
    }
