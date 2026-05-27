"""Parquet retry helper for LIVE bot data fetcher.

Use this wrapper around pandas.read_parquet calls in build_live_dataset
to handle transient '<Buffer>' / 'magic bytes not found' errors caused by
race condition with collectors writing parquet files.
"""
import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def read_parquet_with_retry(path, retries=3, wait_ms=100):
    """Read parquet with retry on transient corruption errors.

    Args:
        path: parquet file path or Path-like
        retries: number of retry attempts (total reads = 1 + retries)
        wait_ms: milliseconds to wait between retries

    Returns:
        pd.DataFrame

    Raises:
        Original exception if all retries exhausted.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            transient = (
                "magic bytes" in err_str
                or "<buffer>" in err_str
                or "footer" in err_str
                or "corrupt snappy" in err_str
                or "invalid column metadata" in err_str
            )
            if not transient or attempt == retries:
                raise
            log.warning(
                f"Parquet transient error (attempt {attempt+1}/{retries+1}) on {path}: {e}; "
                f"retrying after {wait_ms}ms"
            )
            time.sleep(wait_ms / 1000.0)
    raise last_err  # unreachable but explicit


def read_parquet_range(paths, retries=3, wait_ms=100):
    """Read multiple parquet files with per-file retry, return concat DataFrame.

    Skips files that fail after all retries (logs warning).
    """
    dfs = []
    for p in paths:
        try:
            df = read_parquet_with_retry(p, retries=retries, wait_ms=wait_ms)
            dfs.append(df)
        except Exception as e:
            log.error(f"Skipping {p} after retries: {e}")
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
