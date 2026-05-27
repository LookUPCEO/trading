"""Lagged + change features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_lagged_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    lags: list[int] = [1, 5],
) -> pd.DataFrame:
    """
    Add lag, change, and pct_change features.

    Assumes df is 1-min sub-sampled (shift(1) = 1 min ago).
    """
    out = df.copy()

    for feat in feature_cols:
        if feat not in out.columns:
            continue

        for lag in lags:
            lag_col = f"{feat}_lag_{lag}m"
            change_col = f"{feat}_change_{lag}m"
            pct_col = f"{feat}_pct_change_{lag}m"

            out[lag_col] = out[feat].shift(lag)
            out[change_col] = out[feat] - out[lag_col]
            out[pct_col] = np.where(
                np.abs(out[lag_col]) > 1e-10,
                (out[feat] - out[lag_col]) / out[lag_col] * 100,
                0.0,
            )

    return out
