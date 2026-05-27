"""
Rolling / time-series Order Book features.
Operate on a feature DataFrame indexed by timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_stats(
    feat_df: pd.DataFrame,
    base_col: str = "mid_price",
    windows_seconds: list[int] = [60, 300, 900],
) -> pd.DataFrame:
    """
    Rolling std/mean over time-based windows.
    Assumes feat_df has a 'timestamp' column or DatetimeIndex.
    """
    if "timestamp" in feat_df.columns:
        df = feat_df.set_index("timestamp").sort_index()
    else:
        df = feat_df.sort_index()

    out = pd.DataFrame(index=df.index)

    if base_col not in df.columns:
        return out

    series = df[base_col]

    for w in windows_seconds:
        win = f"{w}s"
        out[f"{base_col}_std_{w}s"] = series.rolling(win).std()
        out[f"{base_col}_mean_{w}s"] = series.rolling(win).mean()
        # Z-score: how many std from mean
        out[f"{base_col}_zscore_{w}s"] = (
            (series - out[f"{base_col}_mean_{w}s"]) / out[f"{base_col}_std_{w}s"]
        )

    return out


def compute_obi_persistence(
    feat_df: pd.DataFrame,
    obi_col: str = "obi_top5",
    windows_seconds: list[int] = [60, 300],
) -> pd.DataFrame:
    """
    OBI persistence: rolling mean of imbalance.
    Strong sustained imbalance is more meaningful than instant.
    """
    if "timestamp" in feat_df.columns:
        df = feat_df.set_index("timestamp").sort_index()
    else:
        df = feat_df.sort_index()

    out = pd.DataFrame(index=df.index)

    if obi_col not in df.columns:
        return out

    series = df[obi_col]

    for w in windows_seconds:
        win = f"{w}s"
        out[f"{obi_col}_mean_{w}s"] = series.rolling(win).mean()
        # Sign consistency: % of time same direction
        sign = np.sign(series)
        out[f"{obi_col}_consistency_{w}s"] = (
            sign.rolling(win).mean()  # ranges -1 to +1
        )

    return out
