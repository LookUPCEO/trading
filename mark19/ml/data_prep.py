"""ML data preparation: combine 12 dates into train/val/test splits."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from mark19.storage import read_range
from mark19.features.orderbook import compute_all_pointwise
from mark19.features.orderbook_timeseries import compute_rolling_stats, compute_obi_persistence
from mark19.features.trades import aggregate_to_1s, compute_rolling_features as compute_trades_rolling
from mark19.features.liquidation import compute_liquidation_features
from mark19.features.lagged import add_lagged_features
from mark19.features.cross import add_cross_features
from mark19.features.microstructure import add_microstructure_features
from mark19.features.adaptive import add_adaptive_features


EXCHANGE = "bybit_tardis"
SYMBOL = "ETHUSDT"

DATES_TRAIN = [
    "2022-01-01", "2022-04-01", "2022-05-01", "2022-07-01",
    "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01",
    "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01",
    "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01",
    "2023-11-01",
    "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01",
]
DATES_VAL = ["2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01"]
DATES_TEST = [
    "2024-11-01", "2024-12-01",
    "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01",
]

LAG_FEATURES = [
    "tr_trade_count_300s",
    "tr_trades_per_sec_300s",
    "tr_large_trade_count_300s",
    "tr_total_volume_300s",
    "liq_liq_count_300s",
    "liq_liq_notional_300s",
    "ob_obi_top5",
    "ob_spread",
    "dt_funding_rate",
    "dt_open_interest",
]
LAGS = [1, 5]


PRICE_RAW_PATTERNS = [
    "ob_mid_price",
    "ob_mid_price_mean_",
    "ob_mid_price_std_",
    "tr_vwap",
    "dt_last_price",
    "dt_index_price",
    "dt_mark_price",
]


def is_price_raw(col):
    for pat in PRICE_RAW_PATTERNS:
        if col == pat or col.startswith(pat):
            return True
    return False


def build_date_dataset(date_str: str, log, exchange: str = None, symbol: str = None) -> pd.DataFrame:
    """Build per-date feature dataset.

    Args:
        date_str: 'YYYY-MM-DD'
        log: logger
        exchange: data source exchange (default: module-level EXCHANGE = 'bybit_tardis').
            Use 'bybit_tardis_trial' to read from Tardis trial download path.
        symbol: trading symbol (default: module-level SYMBOL = 'ETHUSDT').
    """
    if exchange is None:
        exchange = EXCHANGE
    if symbol is None:
        symbol = SYMBOL

    y, m, d = map(int, date_str.split("-"))
    start = datetime(y, m, d, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    log.info(f"  Building dataset for {date_str} (exchange={exchange}, symbol={symbol})")

    features = {}

    ob_raw = read_range("orderbook", exchange, symbol, start, end)
    if len(ob_raw) > 100:
        ob_pw = compute_all_pointwise(ob_raw)
        ob_rs = compute_rolling_stats(ob_pw, "mid_price", [60, 300, 900])
        ob_op = compute_obi_persistence(ob_pw, "obi_top5", [60, 300])
        ob_pw_idx = ob_pw.set_index("timestamp") if "timestamp" in ob_pw.columns else ob_pw
        features["orderbook"] = pd.concat([ob_pw_idx, ob_rs, ob_op], axis=1).reset_index()

    tr_raw = read_range("trades", exchange, symbol, start, end)
    if len(tr_raw) > 1000:
        tr_agg = aggregate_to_1s(tr_raw)
        tr_rolling = compute_trades_rolling(tr_agg, [60, 300, 900])
        features["trades"] = pd.merge(tr_agg, tr_rolling, on="timestamp", how="outer")

    liq_raw = read_range("liquidation", exchange, symbol, start, end)
    if len(liq_raw) > 5:
        features["liquidation"] = compute_liquidation_features(liq_raw, [60, 300, 3600])

    dt_raw = read_range("derivative_ticker", exchange, symbol, start, end)
    if len(dt_raw) > 100:
        dt = dt_raw.copy().sort_values("timestamp")
        dt["timestamp"] = pd.to_datetime(dt["timestamp"], utc=True).dt.floor("1s")
        dt = dt.drop_duplicates("timestamp", keep="last")
        features["derivative_ticker"] = dt

    if "orderbook" not in features:
        return pd.DataFrame()

    base = features["orderbook"].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True).dt.floor("1s")
    base = base.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")

    full_idx = pd.date_range(base.index.min(), base.index.max(), freq="1s", tz="UTC")
    combined = base.reindex(full_idx)
    combined.columns = [f"ob_{c}" for c in combined.columns]

    if "trades" in features:
        t = features["trades"].copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True).dt.floor("1s")
        t = t.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        t = t.reindex(full_idx)
        t.columns = [f"tr_{c}" for c in t.columns]
        combined = combined.join(t)

    if "liquidation" in features:
        l = features["liquidation"].copy()
        l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True).dt.floor("1s")
        l = l.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        l = l.reindex(full_idx, fill_value=0)
        l.columns = [f"liq_{c}" for c in l.columns]
        combined = combined.join(l)

    if "derivative_ticker" in features:
        d = features["derivative_ticker"].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.floor("1s")
        d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        d = d.reindex(full_idx, method="ffill", limit=300)
        keep = ["funding_rate", "predicted_funding_rate", "open_interest",
                "last_price", "index_price", "mark_price"]
        d = d[[c for c in keep if c in d.columns]]
        d.columns = [f"dt_{c}" for c in d.columns]
        combined = combined.join(d)

    if "ob_mid_price" not in combined.columns:
        return pd.DataFrame()

    mid = combined["ob_mid_price"]
    for N in [300, 900, 3600]:
        min_p = max(N // 2, 1)
        future_mid = mid.shift(-N)
        combined[f"target_return_{N}s"] = (future_mid - mid) / mid * 100
        combined[f"target_volatility_{N}s"] = mid.rolling(N, min_periods=min_p).std().shift(-(N-1))
        combined[f"target_max_drawdown_{N}s"] = (mid.rolling(N, min_periods=min_p).min().shift(-(N-1)) - mid) / mid * 100
        combined[f"target_max_runup_{N}s"] = (mid.rolling(N, min_periods=min_p).max().shift(-(N-1)) - mid) / mid * 100

    combined = combined.reset_index().rename(columns={"index": "timestamp"})

    df_1min = combined.iloc[::60].copy().reset_index(drop=True)

    available_lag = [f for f in LAG_FEATURES if f in df_1min.columns]
    df_1min = add_lagged_features(df_1min, available_lag, LAGS)
    df_1min = add_cross_features(df_1min)
    df_1min = add_adaptive_features(df_1min)

    df_1min["_source_date"] = date_str

    return df_1min


def build_split(dates: list, log, exchange: str = None, symbol: str = None) -> pd.DataFrame:
    dfs = []
    for date_str in dates:
        df = build_date_dataset(date_str, log, exchange=exchange, symbol=symbol)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"  Combined {len(dfs)} dates: {len(combined)} rows × {len(combined.columns)} cols")
    return combined


def get_feature_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if c == "timestamp" or c == "_source_date":
            continue
        if c.startswith("target_"):
            continue
        if is_price_raw(c):
            continue
        cols.append(c)
    return cols
