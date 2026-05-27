"""
Point-in-time Order Book features.

Each function takes a single row (or DataFrame) and returns feature(s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEPTH = 50  # max levels stored


def compute_spread(df: pd.DataFrame) -> pd.DataFrame:
    """spread, spread_pct, mid_price."""
    out = pd.DataFrame(index=df.index)
    out["mid_price"] = (df["bid_0_price"] + df["ask_0_price"]) / 2
    out["spread"] = df["ask_0_price"] - df["bid_0_price"]
    out["spread_pct"] = out["spread"] / out["mid_price"]
    return out


def compute_imbalance(df: pd.DataFrame, levels: list[int] = [1, 5, 10, 50]) -> pd.DataFrame:
    """
    Order book imbalance at various depths.
    OBI = (bid_size - ask_size) / (bid_size + ask_size)
    Range: [-1, +1]. Positive = buy pressure dominant.
    """
    out = pd.DataFrame(index=df.index)

    for k in levels:
        bid_cols = [f"bid_{i}_size" for i in range(k)]
        ask_cols = [f"ask_{i}_size" for i in range(k)]

        bid_sum = df[bid_cols].sum(axis=1, skipna=True)
        ask_sum = df[ask_cols].sum(axis=1, skipna=True)
        total = bid_sum + ask_sum

        # Avoid div by zero
        obi = np.where(total > 0, (bid_sum - ask_sum) / total, 0.0)
        out[f"obi_top{k}"] = obi

    return out


def compute_depth(df: pd.DataFrame, levels: list[int] = [1, 5, 10, 50]) -> pd.DataFrame:
    """Cumulative depth at various levels."""
    out = pd.DataFrame(index=df.index)

    for k in levels:
        bid_cols = [f"bid_{i}_size" for i in range(k)]
        ask_cols = [f"ask_{i}_size" for i in range(k)]

        out[f"bid_depth_{k}"] = df[bid_cols].sum(axis=1, skipna=True)
        out[f"ask_depth_{k}"] = df[ask_cols].sum(axis=1, skipna=True)
        out[f"total_depth_{k}"] = out[f"bid_depth_{k}"] + out[f"ask_depth_{k}"]

    return out


def compute_slope(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """
    Price slope across order book levels.
    bid_slope: how fast bid prices fall as we go deeper
    ask_slope: how fast ask prices rise
    """
    out = pd.DataFrame(index=df.index)

    bid_top = df["bid_0_price"]
    bid_far = df[f"bid_{span-1}_price"]
    ask_top = df["ask_0_price"]
    ask_far = df[f"ask_{span-1}_price"]

    out[f"bid_slope_{span}"] = (bid_top - bid_far) / span
    out[f"ask_slope_{span}"] = (ask_far - ask_top) / span

    return out


def compute_all_pointwise(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all point-wise features."""
    parts = [
        df[["timestamp"]] if "timestamp" in df.columns else pd.DataFrame(index=df.index),
        compute_spread(df),
        compute_imbalance(df),
        compute_depth(df),
        compute_slope(df, span=10),
    ]
    return pd.concat(parts, axis=1)
