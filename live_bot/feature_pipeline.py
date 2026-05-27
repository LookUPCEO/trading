"""Live Feature Pipeline.

Builds 1-minute grid features for live trading, mirroring
mark19/ml/data_prep.py::build_date_dataset but:
  - reads exchange='bybit' (live collectors) instead of 'bybit_tardis'
  - synthesizes derivative_ticker from funding_current + train_medians
  - skips target_* computation (future unknown at live time)
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from mark19.storage import read_range
from mark19.features.orderbook import compute_all_pointwise
from mark19.features.orderbook_timeseries import compute_rolling_stats, compute_obi_persistence
from mark19.features.trades import aggregate_to_1s, compute_rolling_features as compute_trades_rolling
from mark19.features.liquidation import compute_liquidation_features
from mark19.features.lagged import add_lagged_features
from mark19.features.cross import add_cross_features
from mark19.features.adaptive import add_adaptive_features
from mark19.ml.data_prep import LAG_FEATURES, LAGS

from live_bot.dt_adapter import load_funding_current, synthesize_dt_dataframe

log = logging.getLogger(__name__)

EXCHANGE_LIVE = "bybit"   # live collectors (vs backtest "bybit_tardis")
SYMBOL = "ETHUSDT"


def build_live_dataset(
    now: Optional[datetime] = None,
    lookback_hours: int = 25,
    train_medians: Optional[dict] = None,
) -> pd.DataFrame:
    """Mirror of build_date_dataset for live data.

    Returns a 1-minute grid DataFrame. Latest row = most recent 1-min bar.
    Returns empty DataFrame if orderbook data is missing.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if train_medians is None:
        train_medians = {}

    end = now
    start = now - timedelta(hours=lookback_hours)
    log.info(f"Building live dataset: [{start.isoformat()} ~ {end.isoformat()}]")

    features = {}

    # ---- Read collector data ----
    ob_raw = read_range("orderbook", EXCHANGE_LIVE, SYMBOL, start, end)
    log.info(f"  orderbook: {len(ob_raw)} rows")
    if len(ob_raw) > 100:
        ob_pw = compute_all_pointwise(ob_raw)
        ob_rs = compute_rolling_stats(ob_pw, "mid_price", [60, 300, 900])
        ob_op = compute_obi_persistence(ob_pw, "obi_top5", [60, 300])
        ob_pw_idx = ob_pw.set_index("timestamp") if "timestamp" in ob_pw.columns else ob_pw
        features["orderbook"] = pd.concat([ob_pw_idx, ob_rs, ob_op], axis=1).reset_index()

    tr_raw = read_range("trades", EXCHANGE_LIVE, SYMBOL, start, end)
    log.info(f"  trades: {len(tr_raw)} rows")
    if len(tr_raw) > 1000:
        tr_agg = aggregate_to_1s(tr_raw)
        tr_rolling = compute_trades_rolling(tr_agg, [60, 300, 900])
        features["trades"] = pd.merge(tr_agg, tr_rolling, on="timestamp", how="outer")

    liq_raw = read_range("liquidation", EXCHANGE_LIVE, SYMBOL, start, end)
    log.info(f"  liquidation: {len(liq_raw)} rows")
    if len(liq_raw) > 5:
        features["liquidation"] = compute_liquidation_features(liq_raw, [60, 300, 3600])

    funding_df = load_funding_current(start, end)
    log.info(f"  funding_current: {len(funding_df)} rows")

    if "orderbook" not in features:
        log.error("No orderbook data — cannot build features")
        return pd.DataFrame()

    # ---- Build 1-second grid (orderbook = base) ----
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

    # ---- DT (live-specific synthesis) ----
    dt_df = synthesize_dt_dataframe(funding_df=funding_df, grid_index=full_idx, train_medians=train_medians)
    combined = combined.join(dt_df)

    if "ob_mid_price" not in combined.columns:
        log.error("ob_mid_price missing after join — orderbook schema mismatch")
        return pd.DataFrame()

    # ---- No targets in live (future unknown) ----

    # ---- 1s → 1min downsample ----
    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    df_1min = combined.iloc[::60].copy().reset_index(drop=True)
    log.info(f"  1-min grid: {len(df_1min)} rows")

    # ---- Lagged + Cross + Adaptive ----
    available_lag = [f for f in LAG_FEATURES if f in df_1min.columns]
    df_1min = add_lagged_features(df_1min, available_lag, LAGS)
    df_1min = add_cross_features(df_1min)
    df_1min = add_adaptive_features(df_1min)

    log.info(f"  Total columns: {len(df_1min.columns)}")
    return df_1min


def get_latest_features(
    feature_cols: list,
    train_medians: dict,
    now: Optional[datetime] = None,
    lookback_hours: int = 25,
) -> Optional[pd.Series]:
    """Return the latest 1-minute feature row, aligned to feature_cols, NaN-filled.

    Returns None if no data available.
    """
    df = build_live_dataset(now=now, lookback_hours=lookback_hours, train_medians=train_medians)
    if df.empty:
        return None

    latest = df.iloc[-1]
    train_medians_s = pd.Series(train_medians)

    feature_row = latest.reindex(feature_cols)
    feature_row = feature_row.replace([np.inf, -np.inf], np.nan)
    feature_row = feature_row.fillna(train_medians_s).fillna(0)

    feature_row.name = latest.get("timestamp", None)
    return feature_row
