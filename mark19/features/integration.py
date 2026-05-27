"""
Integrate all feature streams into a 1s-grid dataset with targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mark19.storage import read_range


def load_all_features(start, end) -> dict[str, pd.DataFrame]:
    """Load all pre-computed features."""
    return {
        "orderbook": read_range("orderbook_features", "bybit", "ETHUSDT", start, end),
        "trades": read_range("trades_features", "bybit", "ETHUSDT", start, end),
        "liquidation": read_range("liquidation_features", "bybit", "ETHUSDT", start, end),
        "cross_price": read_range("cross_exchange_features_price", "combined", "ETHUSDT", start, end),
        "cross_funding": read_range("cross_exchange_features_funding", "combined", "ETHUSDT", start, end),
    }


def integrate_to_1s_grid(streams: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align all streams to common 1s grid using orderbook as base.

    All stream timestamps are floored to 1s before join to handle
    sub-second timestamp precision differences.
    """
    ob = streams.get("orderbook")
    if ob is None or ob.empty:
        raise ValueError("orderbook features required")

    base = ob.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True).dt.floor("1s")
    base = base.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")

    full_idx = pd.date_range(base.index.min(), base.index.max(), freq="1s", tz="UTC")
    combined = base.reindex(full_idx)
    combined.columns = [f"ob_{c}" for c in combined.columns]

    # Trades (1s native, NaN for empty)
    tr = streams.get("trades")
    if tr is not None and not tr.empty:
        t = tr.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True).dt.floor("1s")
        t = t.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        t = t.reindex(full_idx)
        t.columns = [f"tr_{c}" for c in t.columns]
        combined = combined.join(t)

    # Liquidation (1s native, fill 0 for empty seconds)
    liq = streams.get("liquidation")
    if liq is not None and not liq.empty:
        l = liq.copy()
        l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True).dt.floor("1s")
        l = l.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        l = l.reindex(full_idx, fill_value=0)
        l.columns = [f"liq_{c}" for c in l.columns]
        combined = combined.join(l)

    # Cross-exchange price (10s polling, ffill up to 30s)
    cp = streams.get("cross_price")
    if cp is not None and not cp.empty:
        c = cp.copy()
        c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True).dt.floor("1s")
        c = c.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        c = c.reindex(full_idx, method="ffill", limit=30)
        c.columns = [f"cx_{col}" for col in c.columns]
        combined = combined.join(c)

    # Cross-exchange funding (10min polling, ffill up to 25min)
    cf = streams.get("cross_funding")
    if cf is not None and not cf.empty:
        f = cf.copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).dt.floor("1s")
        f = f.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        f = f.reindex(full_idx, method="ffill", limit=1500)
        f.columns = [f"cf_{col}" for col in f.columns]
        combined = combined.join(f)

    return combined.reset_index().rename(columns={"index": "timestamp"})


def add_targets(
    df: pd.DataFrame,
    mid_col: str = "ob_mid_price",
    horizons_seconds: list[int] = [300, 900, 3600],
) -> pd.DataFrame:
    """
    Add forward-looking targets.

    Uses min_periods to allow partial windows when mid has NaN gaps.
    A window is valid if at least 50% of points are non-NaN.
    """
    if mid_col not in df.columns:
        raise ValueError(f"{mid_col} not in dataframe")

    out = df.copy().sort_values("timestamp").set_index("timestamp")
    mid = out[mid_col]

    for N in horizons_seconds:
        min_p = max(N // 2, 1)  # at least 50% non-NaN required

        # Future return: simple shift (already works)
        future_mid = mid.shift(-N)
        out[f"target_return_{N}s"] = (future_mid - mid) / mid * 100

        # Future window stats with min_periods
        # rolling(N, min_periods=min_p).X then shift(-(N-1)) for forward window
        out[f"target_volatility_{N}s"] = (
            mid.rolling(N, min_periods=min_p).std().shift(-(N-1))
        )
        out[f"target_max_drawdown_{N}s"] = (
            (mid.rolling(N, min_periods=min_p).min().shift(-(N-1)) - mid) / mid * 100
        )
        out[f"target_max_runup_{N}s"] = (
            (mid.rolling(N, min_periods=min_p).max().shift(-(N-1)) - mid) / mid * 100
        )

    return out.reset_index()
