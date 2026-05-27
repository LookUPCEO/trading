"""
Trades features.

Pre-aggregate to 1-second buckets (vectorized), then compute rolling stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LARGE_TRADE_THRESHOLD = 1.0  # ETH


def aggregate_to_1s(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw trades to 1-second buckets (vectorized).

    Returns DataFrame with timestamp column and aggregated metrics.
    Empty seconds are filled with zeros (vwap=NaN).
    """
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["bucket"] = df["timestamp"].dt.floor("1s")

    # Pre-compute columns for vectorized aggregation
    df["is_buy"] = (df["side"] == "Buy").astype(int)
    df["is_sell"] = (df["side"] == "Sell").astype(int)
    df["buy_volume_per"] = df["size"] * df["is_buy"]
    df["sell_volume_per"] = df["size"] * df["is_sell"]
    df["is_large"] = (df["size"] >= LARGE_TRADE_THRESHOLD).astype(int)
    df["is_plus"] = (df["tick_direction"] == "PlusTick").astype(int)
    df["is_minus"] = (df["tick_direction"] == "MinusTick").astype(int)
    df["is_zero"] = df["tick_direction"].isin(["ZeroPlusTick", "ZeroMinusTick"]).astype(int)
    df["price_volume"] = df["price"] * df["size"]

    # Vectorized groupby agg
    agg = df.groupby("bucket").agg(
        buy_volume=("buy_volume_per", "sum"),
        sell_volume=("sell_volume_per", "sum"),
        buy_count=("is_buy", "sum"),
        sell_count=("is_sell", "sum"),
        total_count=("is_buy", "size"),
        avg_size=("size", "mean"),
        max_size=("size", "max"),
        large_trade_count=("is_large", "sum"),
        tick_plus_count=("is_plus", "sum"),
        tick_minus_count=("is_minus", "sum"),
        tick_zero_count=("is_zero", "sum"),
        sum_price_volume=("price_volume", "sum"),
        sum_volume=("size", "sum"),
    )

    # Reindex to fill empty seconds with 0 (no trades = legitimate zero)
    full_idx = pd.date_range(
        agg.index.min(), agg.index.max(), freq="1s", tz="UTC"
    )
    agg = agg.reindex(full_idx, fill_value=0)

    # Compute derived (after reindex so empty seconds get NaN where appropriate)
    agg["vwap"] = np.where(
        agg["sum_volume"] > 0,
        agg["sum_price_volume"] / agg["sum_volume"].replace(0, np.nan),
        np.nan,
    )
    agg["total_volume"] = agg["buy_volume"] + agg["sell_volume"]
    agg["buy_ratio"] = np.where(
        agg["total_volume"] > 0,
        agg["buy_volume"] / agg["total_volume"],
        0.5,
    )
    agg["tick_buy_ratio"] = np.where(
        (agg["tick_plus_count"] + agg["tick_minus_count"]) > 0,
        agg["tick_plus_count"] / (agg["tick_plus_count"] + agg["tick_minus_count"]),
        0.5,
    )

    # Drop intermediate
    agg = agg.drop(columns=["sum_price_volume", "sum_volume"])

    agg.index.name = "timestamp"
    return agg.reset_index()


def compute_rolling_features(
    agg_1s: pd.DataFrame,
    windows_seconds: list[int] = [60, 300, 900],
) -> pd.DataFrame:
    """
    Rolling stats over 1s aggregated data.
    """
    if agg_1s.empty:
        return pd.DataFrame()

    df = agg_1s.set_index("timestamp").sort_index()
    out = pd.DataFrame(index=df.index)

    for w in windows_seconds:
        win = f"{w}s"

        buy_vol_w = df["buy_volume"].rolling(win).sum()
        sell_vol_w = df["sell_volume"].rolling(win).sum()
        total_vol_w = buy_vol_w + sell_vol_w
        total_count_w = df["total_count"].rolling(win).sum()

        # Volume metrics
        out[f"buy_ratio_{w}s"] = np.where(
            total_vol_w > 0, buy_vol_w / total_vol_w.replace(0, np.nan), 0.5
        )
        out[f"volume_imbalance_{w}s"] = np.where(
            total_vol_w > 0,
            (buy_vol_w - sell_vol_w) / total_vol_w.replace(0, np.nan),
            0.0,
        )
        out[f"total_volume_{w}s"] = total_vol_w

        # Count metrics
        out[f"trade_count_{w}s"] = total_count_w
        out[f"trades_per_sec_{w}s"] = total_count_w / w

        # Average trade size
        out[f"avg_size_{w}s"] = total_vol_w / total_count_w.replace(0, np.nan)

        # Large trade
        large_count_w = df["large_trade_count"].rolling(win).sum()
        out[f"large_trade_count_{w}s"] = large_count_w
        out[f"large_trade_ratio_{w}s"] = large_count_w / total_count_w.replace(0, np.nan)

        # Tick rule
        plus_w = df["tick_plus_count"].rolling(win).sum()
        minus_w = df["tick_minus_count"].rolling(win).sum()
        plusminus_w = plus_w + minus_w
        out[f"tick_buy_ratio_{w}s"] = np.where(
            plusminus_w > 0, plus_w / plusminus_w.replace(0, np.nan), 0.5
        )

    return out.reset_index()
