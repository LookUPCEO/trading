"""Cross-exchange features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spread features from cross-exchange prices.

    Input: timestamp, bybit_eth_usd, binance_eth_usd, okx_eth_usd, upbit_eth_krw
    """
    if prices.empty:
        return pd.DataFrame()

    df = prices.copy().sort_values("timestamp").reset_index(drop=True)
    out = pd.DataFrame({"timestamp": df["timestamp"]})

    bb = df["bybit_eth_usd"]
    bn = df["binance_eth_usd"]
    ok = df["okx_eth_usd"]

    # Pairwise spreads (bps)
    out["spread_bb_bn_bps"] = (bb - bn) / bb.replace(0, np.nan) * 10000
    out["spread_bb_ok_bps"] = (bb - ok) / bb.replace(0, np.nan) * 10000
    out["spread_bn_ok_bps"] = (bn - ok) / bn.replace(0, np.nan) * 10000

    # Max absolute spread
    out["max_spread_bps"] = pd.concat([
        out["spread_bb_bn_bps"].abs(),
        out["spread_bb_ok_bps"].abs(),
        out["spread_bn_ok_bps"].abs(),
    ], axis=1).max(axis=1)

    # 3-exchange std normalized
    prices_3 = pd.concat([bb, bn, ok], axis=1)
    out["price_std_3ex"] = prices_3.std(axis=1)
    out["price_mean_3ex"] = prices_3.mean(axis=1)
    out["spread_std_bps"] = (
        out["price_std_3ex"] / out["price_mean_3ex"].replace(0, np.nan) * 10000
    )

    # Bybit lead score (vectorized)
    max_p = prices_3.max(axis=1)
    min_p = prices_3.min(axis=1)
    has_nan = bb.isna() | bn.isna() | ok.isna()

    out["bybit_lead_score"] = np.where(
        has_nan, np.nan,
        np.where(bb == max_p, 1,
            np.where(bb == min_p, -1, 0))
    )

    # Pass through Upbit raw (for later kimchi premium with FX)
    out["upbit_eth_krw"] = df["upbit_eth_krw"]

    return out


def compute_funding_features(funding: pd.DataFrame) -> pd.DataFrame:
    """
    Compute funding rate features.

    Input: timestamp, bybit_funding, binance_funding, okx_funding (+ next_time fields)
    """
    if funding.empty:
        return pd.DataFrame()

    df = funding.copy().sort_values("timestamp").reset_index(drop=True)
    out = pd.DataFrame({"timestamp": df["timestamp"]})

    bb = df["bybit_funding"]
    bn = df["binance_funding"]
    ok = df["okx_funding"]

    out["bybit_funding"] = bb
    out["binance_funding"] = bn
    out["okx_funding"] = ok

    out["funding_diff_bb_bn"] = bb - bn
    out["funding_diff_bb_ok"] = bb - ok
    out["funding_diff_bn_ok"] = bn - ok

    funding_3 = pd.concat([bb, bn, ok], axis=1)
    out["funding_max"] = funding_3.max(axis=1)
    out["funding_min"] = funding_3.min(axis=1)
    out["funding_max_diff"] = out["funding_max"] - out["funding_min"]
    out["funding_mean"] = funding_3.mean(axis=1)
    out["funding_std"] = funding_3.std(axis=1)

    return out
