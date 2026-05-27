"""Liquidation features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_liquidation_features(
    liq: pd.DataFrame,
    windows_seconds: list[int] = [60, 300, 3600],
) -> pd.DataFrame:
    """
    Expand liquidations to 1s grid, compute rolling counts/notionals/ratios.
    """
    if liq.empty:
        return pd.DataFrame()

    df = liq.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["bucket"] = df["timestamp"].dt.floor("1s")

    # Pre-computed columns for vectorized agg
    df["is_buy"] = (df["side"] == "Buy").astype(int)
    df["is_sell"] = (df["side"] == "Sell").astype(int)
    df["buy_notional_per"] = df["notional"] * df["is_buy"]
    df["sell_notional_per"] = df["notional"] * df["is_sell"]

    # Vectorized aggregation
    agg = df.groupby("bucket").agg(
        count=("notional", "size"),
        notional=("notional", "sum"),
        max_single=("notional", "max"),
        buy_count=("is_buy", "sum"),
        sell_count=("is_sell", "sum"),
        buy_notional=("buy_notional_per", "sum"),
        sell_notional=("sell_notional_per", "sum"),
    )

    # Reindex to full 1s grid
    full_idx = pd.date_range(
        agg.index.min(), agg.index.max(), freq="1s", tz="UTC"
    )
    agg = agg.reindex(full_idx, fill_value=0)
    agg.index.name = "timestamp"

    # Rolling features
    out = pd.DataFrame(index=agg.index)

    for w in windows_seconds:
        win = f"{w}s"

        count_w = agg["count"].rolling(win).sum()
        notional_w = agg["notional"].rolling(win).sum()
        buy_count_w = agg["buy_count"].rolling(win).sum()
        sell_count_w = agg["sell_count"].rolling(win).sum()
        buy_not_w = agg["buy_notional"].rolling(win).sum()
        sell_not_w = agg["sell_notional"].rolling(win).sum()
        max_single_w = agg["max_single"].rolling(win).max()

        out[f"liq_count_{w}s"] = count_w
        out[f"liq_notional_{w}s"] = notional_w
        out[f"liq_max_single_{w}s"] = max_single_w

        # Buy ratio by count
        total_count = buy_count_w + sell_count_w
        out[f"liq_buy_ratio_count_{w}s"] = np.where(
            total_count > 0,
            buy_count_w / total_count.replace(0, np.nan),
            0.5,
        )

        # Buy ratio by notional
        total_not = buy_not_w + sell_not_w
        out[f"liq_buy_ratio_notional_{w}s"] = np.where(
            total_not > 0,
            buy_not_w / total_not.replace(0, np.nan),
            0.5,
        )

    return out.reset_index()
