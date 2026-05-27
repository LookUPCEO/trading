"""
Tardis CSV → Mark19 parquet converter.

Tardis CSV schema:
  - trades: exchange, symbol, timestamp, local_timestamp, id, side, price, amount
  - book_snapshot_25: exchange, symbol, timestamp, local_timestamp, asks[0..24].price/amount, bids[0..24].price/amount
  - liquidations: exchange, symbol, timestamp, local_timestamp, id, side, price, amount
  - derivative_ticker: exchange, symbol, timestamp, local_timestamp, funding_timestamp,
      funding_rate, predicted_funding_rate, open_interest, last_price, index_price, mark_price

All timestamps in microseconds since epoch.
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DEPTH = 25  # Tardis book_snapshot_25


def _us_to_dt(us_series: pd.Series) -> pd.Series:
    """Microseconds since epoch → tz-aware UTC datetime."""
    return pd.to_datetime(us_series.astype("int64"), unit="us", utc=True)


def convert_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tardis trades → Mark19 trades schema.

    Mark19 columns: timestamp, symbol, side, price, size, trade_id, tick_direction, block_trade
    """
    out = pd.DataFrame()
    out["timestamp"] = _us_to_dt(df["timestamp"])
    out["symbol"] = df["symbol"].str.upper()
    out["side"] = df["side"].str.capitalize()  # buy → Buy, sell → Sell
    out["price"] = df["price"].astype(float)
    out["size"] = df["amount"].astype(float)
    out["trade_id"] = df["id"].astype(str)
    out["tick_direction"] = None
    out["block_trade"] = False
    return out


def convert_liquidations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tardis liquidations → Mark19 liquidation schema.

    Mark19 columns: timestamp, symbol, side, price, size, notional, received_at
    """
    out = pd.DataFrame()
    out["timestamp"] = _us_to_dt(df["timestamp"])
    out["symbol"] = df["symbol"].str.upper()
    out["side"] = df["side"].str.capitalize()
    out["price"] = df["price"].astype(float)
    out["size"] = df["amount"].astype(float)
    out["notional"] = out["price"] * out["size"]
    out["received_at"] = _us_to_dt(df["local_timestamp"])
    return out


def convert_book_snapshot_25(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tardis book_snapshot_25 → Mark19 orderbook schema (25 levels, sub-sampled to 1s).

    Tardis 는 25 levels 만 → bid_25..49 / ask_25..49 는 NaN.
    Tardis 는 ~35Hz → 1Hz 로 sub-sample (각 초의 마지막 snapshot).
    """
    df = df.copy()
    df["timestamp"] = _us_to_dt(df["timestamp"])
    df["bucket"] = df["timestamp"].dt.floor("1s")

    df_1s = df.drop_duplicates(subset=["bucket"], keep="last").reset_index(drop=True)

    out = pd.DataFrame()
    out["timestamp"] = df_1s["bucket"]
    out["update_id"] = 0
    out["sequence"] = 0

    for i in range(DEPTH):
        out[f"bid_{i}_price"] = df_1s[f"bids[{i}].price"].astype(float)
        out[f"bid_{i}_size"] = df_1s[f"bids[{i}].amount"].astype(float)
        out[f"ask_{i}_price"] = df_1s[f"asks[{i}].price"].astype(float)
        out[f"ask_{i}_size"] = df_1s[f"asks[{i}].amount"].astype(float)

    for i in range(DEPTH, 50):
        out[f"bid_{i}_price"] = np.nan
        out[f"bid_{i}_size"] = np.nan
        out[f"ask_{i}_price"] = np.nan
        out[f"ask_{i}_size"] = np.nan

    return out


def convert_derivative_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tardis derivative_ticker → 새 데이터 타입 'derivative_ticker'.

    이 데이터는 funding_current (10분 폴링) 보다 풍부:
      - funding_rate (실시간)
      - open_interest, mark_price, index_price (신규)
    """
    out = pd.DataFrame()
    out["timestamp"] = _us_to_dt(df["timestamp"])
    out["symbol"] = df["symbol"].str.upper()
    out["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    out["predicted_funding_rate"] = pd.to_numeric(df["predicted_funding_rate"], errors="coerce")

    ft = pd.to_numeric(df["funding_timestamp"], errors="coerce")
    out["funding_timestamp"] = pd.to_datetime(ft, unit="us", utc=True, errors="coerce")

    out["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    out["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
    out["index_price"] = pd.to_numeric(df["index_price"], errors="coerce")
    out["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")

    return out
