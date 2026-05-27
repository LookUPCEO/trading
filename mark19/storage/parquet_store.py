"""
Parquet storage for Mark19 data pipeline.

Design:
  - Date-partitioned files: data/{data_type}/{exchange}/{symbol}/YYYY-MM-DD.parquet
  - UTC timestamps, ns precision
  - Snappy compression
  - Append-safe (reads existing + merges + writes)
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def path_for(data_type: str, exchange: str, symbol: str, day: date) -> Path:
    """Returns the parquet path for a given day."""
    return (
        DATA_ROOT / data_type / exchange / symbol
        / f"{day.isoformat()}.parquet"
    )


def write_append(
    df: pd.DataFrame,
    data_type: str,
    exchange: str,
    symbol: str,
    timestamp_col: str = "timestamp",
    dedup_cols: Optional[list[str]] = None,
) -> None:
    """
    Append-safe write. Splits df by date (UTC) of timestamp_col,
    writes each to its day file. If file exists, merges + deduplicates.

    dedup_cols: composite keys to dedupe on (e.g. ['timestamp','exchange']).
                Defaults to [timestamp_col] for backward compat.
    """
    if df.empty:
        return

    if timestamp_col not in df.columns:
        raise ValueError(f"df missing column {timestamp_col}")

    # Ensure UTC
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    df[timestamp_col] = ts
    df["_day"] = ts.dt.date

    dedup_subset = dedup_cols if dedup_cols else [timestamp_col]

    for day, chunk in df.groupby("_day"):
        chunk = chunk.drop(columns=["_day"])
        path = path_for(data_type, exchange, symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, chunk], ignore_index=True)
            combined = combined.drop_duplicates(subset=dedup_subset).sort_values(timestamp_col)
        else:
            combined = chunk.sort_values(timestamp_col)

        combined.to_parquet(path, compression="snappy", index=False)


def read_range(
    data_type: str,
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Read parquet files covering [start, end] (UTC).
    Returns concatenated DataFrame sorted by timestamp.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # bybit_tardis fallback: if requested but missing for a given day, try bybit_tardis_trial
    # (Track C/D data is stored under bybit_tardis_trial; this lets old code using bybit_tardis
    # transparently access trial data when the canonical file is absent.)
    fallback_exchanges = []
    if exchange == "bybit_tardis":
        fallback_exchanges = ["bybit_tardis_trial"]

    days = pd.date_range(start.date(), end.date(), freq="D").date
    dfs = []
    for day in days:
        path = path_for(data_type, exchange, symbol, day)
        if not path.exists():
            for fb_ex in fallback_exchanges:
                fb_path = path_for(data_type, fb_ex, symbol, day)
                if fb_path.exists():
                    path = fb_path
                    break
        if path.exists():
            # Retry on transient '<Buffer>' / magic-bytes errors caused by
            # collector mid-write race. Up to 5 attempts with 200ms backoff.
            last_err = None
            for attempt in range(5):
                try:
                    dfs.append(pd.read_parquet(path))
                    break
                except Exception as e:
                    msg = str(e).lower()
                    transient = ("<buffer>" in msg or "magic bytes" in msg
                                 or "footer" in msg or "corrupt snappy" in msg
                                 or "invalid column metadata" in msg)
                    if not transient or attempt == 4:
                        raise
                    last_err = e
                    time.sleep(0.2)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df[(df[ts_col] >= start) & (df[ts_col] <= end)]
        df = df.sort_values(ts_col).reset_index(drop=True)

    return df


def list_days(data_type: str, exchange: str, symbol: str) -> list[date]:
    """List all days with data for this type/exchange/symbol."""
    base = DATA_ROOT / data_type / exchange / symbol
    if not base.exists():
        return []
    days = []
    for f in base.glob("*.parquet"):
        try:
            days.append(date.fromisoformat(f.stem))
        except ValueError:
            continue
    return sorted(days)
