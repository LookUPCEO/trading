"""Derivative Ticker Adapter for live trading.

Backtest 는 bybit_tardis 의 derivative_ticker 사용:
  funding_rate, predicted_funding_rate, open_interest,
  last_price, index_price, mark_price

Live 환경 (4b-1):
- funding_current/combined parquet 사용 (bybit_funding 만)
- OI/predicted_funding/index_price/mark_price 는 train_median fillna
- 향후 시도 19 에서 v5 /market/tickers collector 추가 예정
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Live-prefixed DT columns (matches data_prep.py output: keep + 'dt_' prefix)
DT_COLS = [
    "dt_funding_rate",
    "dt_predicted_funding_rate",
    "dt_open_interest",
    "dt_last_price",
    "dt_index_price",
    "dt_mark_price",
]


def load_funding_current(
    start: datetime,
    end: datetime,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load funding_current parquet within [start, end]. Returns DataFrame indexed by timestamp."""
    if base_dir is None:
        base_dir = Path("/Users/dohun/Desktop/Mark/mark19/data/funding_current/combined/ETHUSDT")

    if not base_dir.exists():
        log.warning(f"Funding current dir not found: {base_dir}")
        return pd.DataFrame()

    files = sorted(base_dir.glob("*.parquet"))
    if not files:
        log.warning(f"No funding_current parquet files in {base_dir}")
        return pd.DataFrame()

    # Filter files by date range (filename = YYYY-MM-DD)
    relevant = []
    for f in files:
        try:
            day = pd.to_datetime(f.stem, utc=True)
            if day >= (pd.to_datetime(start, utc=True) - pd.Timedelta(days=1)) and day <= pd.to_datetime(end, utc=True):
                relevant.append(f)
        except Exception:
            relevant.append(f)

    if not relevant:
        relevant = files  # fallback: all

    dfs = []
    for f in relevant:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            log.warning(f"Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined[(combined["timestamp"] >= pd.to_datetime(start, utc=True)) &
                            (combined["timestamp"] <= pd.to_datetime(end, utc=True))]
        combined = combined.set_index("timestamp").sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

    return combined


def synthesize_dt_dataframe(
    funding_df: pd.DataFrame,
    grid_index: pd.DatetimeIndex,
    train_medians: dict,
) -> pd.DataFrame:
    """Build a derivative_ticker-equivalent DataFrame on the 1-second grid.

    Columns are already prefixed with `dt_` to match data_prep.build_date_dataset output.
    """
    dt_df = pd.DataFrame(index=grid_index)

    # 1) Funding rate from collector
    if not funding_df.empty and "bybit_funding" in funding_df.columns:
        s = funding_df["bybit_funding"].copy()
        # Floor index to 1s, drop dupes (collector samples are sub-second)
        s.index = s.index.floor("1s")
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(grid_index, method="ffill")
        dt_df["dt_funding_rate"] = s
    else:
        dt_df["dt_funding_rate"] = np.nan

    # 2) Other DT features → train medians (live collector 없음)
    for col in DT_COLS:
        if col == "dt_funding_rate":
            continue
        dt_df[col] = train_medians.get(col, np.nan)

    # 3) Final NaN safety: per-column fillna with train_medians, then 0
    for col in dt_df.columns:
        med = train_medians.get(col)
        if med is not None:
            dt_df[col] = dt_df[col].fillna(med)
        dt_df[col] = dt_df[col].fillna(0)

    return dt_df
