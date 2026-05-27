"""Microstructure features for direction prediction.

Based on:
- Lee-Ready (trade direction), VPIN, BVC
- OFI (Order Flow Imbalance)
- Quote dynamics, liquidation cascade
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}

    # Trade direction momentum
    if "tr_buy_ratio_300s" in df.columns:
        new_cols["ms_buy_ratio_momentum_300_900"] = df["tr_buy_ratio_300s"] - df.get("tr_buy_ratio_900s", df["tr_buy_ratio_300s"])

    if "tr_buy_ratio_300s" in df.columns and "tr_buy_ratio_60s" in df.columns:
        new_cols["ms_buy_ratio_accel"] = df["tr_buy_ratio_60s"] - df["tr_buy_ratio_300s"]

    # OFI
    if "tr_buy_volume_300s" in df.columns and "tr_sell_volume_300s" in df.columns:
        new_cols["ms_ofi_300"] = df["tr_buy_volume_300s"] - df["tr_sell_volume_300s"]
        total = df["tr_buy_volume_300s"] + df["tr_sell_volume_300s"]
        new_cols["ms_ofi_ratio_300"] = (df["tr_buy_volume_300s"] - df["tr_sell_volume_300s"]) / total.replace(0, np.nan)

    if "tr_buy_volume_900s" in df.columns and "tr_sell_volume_900s" in df.columns:
        new_cols["ms_ofi_900"] = df["tr_buy_volume_900s"] - df["tr_sell_volume_900s"]

    # Bid/Ask depth ratio
    if "ob_bid_depth_5" in df.columns and "ob_ask_depth_5" in df.columns:
        total = df["ob_bid_depth_5"] + df["ob_ask_depth_5"]
        new_cols["ms_depth_ratio_5"] = (df["ob_bid_depth_5"] - df["ob_ask_depth_5"]) / total.replace(0, np.nan)

    if "ob_bid_depth_10" in df.columns and "ob_ask_depth_10" in df.columns:
        total = df["ob_bid_depth_10"] + df["ob_ask_depth_10"]
        new_cols["ms_depth_ratio_10"] = (df["ob_bid_depth_10"] - df["ob_ask_depth_10"]) / total.replace(0, np.nan)

    if "ob_bid_depth_50" in df.columns and "ob_ask_depth_50" in df.columns:
        total = df["ob_bid_depth_50"] + df["ob_ask_depth_50"]
        new_cols["ms_depth_ratio_50"] = (df["ob_bid_depth_50"] - df["ob_ask_depth_50"]) / total.replace(0, np.nan)

    # Depth concentration
    if "ob_bid_depth_1" in df.columns and "ob_bid_depth_10" in df.columns:
        new_cols["ms_bid_concentration"] = df["ob_bid_depth_1"] / df["ob_bid_depth_10"].replace(0, np.nan)

    if "ob_ask_depth_1" in df.columns and "ob_ask_depth_10" in df.columns:
        new_cols["ms_ask_concentration"] = df["ob_ask_depth_1"] / df["ob_ask_depth_10"].replace(0, np.nan)

    # Spread dynamics
    if "ob_spread_pct" in df.columns and "ob_spread_pct_lag_5m" in df.columns:
        new_cols["ms_spread_change_5m"] = df["ob_spread_pct"] - df["ob_spread_pct_lag_5m"]

    # Trade size patterns
    if "tr_large_trade_count_300s" in df.columns and "tr_trade_count_300s" in df.columns:
        new_cols["ms_large_trade_ratio_300"] = df["tr_large_trade_count_300s"] / df["tr_trade_count_300s"].replace(0, np.nan)

    if "tr_avg_trade_size_300s" in df.columns and "tr_avg_trade_size_900s" in df.columns:
        new_cols["ms_avg_size_momentum"] = df["tr_avg_trade_size_300s"] / df["tr_avg_trade_size_900s"].replace(0, np.nan)

    # Liquidation cascade
    if "liq_liq_count_60s" in df.columns and "liq_liq_count_300s" in df.columns:
        new_cols["ms_liq_acceleration"] = df["liq_liq_count_60s"] / (df["liq_liq_count_300s"] / 5).replace(0, np.nan)

    if "liq_liq_buy_ratio_count_300s" in df.columns and "liq_liq_buy_ratio_count_3600s" in df.columns:
        new_cols["ms_liq_direction_shift"] = df["liq_liq_buy_ratio_count_300s"] - df["liq_liq_buy_ratio_count_3600s"]

    # Realized vol patterns
    if "ob_mid_price_std_300s" in df.columns and "ob_mid_price_std_900s" in df.columns:
        new_cols["ms_vol_acceleration"] = df["ob_mid_price_std_300s"] / df["ob_mid_price_std_900s"].replace(0, np.nan)

    # Funding dynamics
    if "dt_funding_rate" in df.columns:
        new_cols["ms_funding_sign"] = np.sign(df["dt_funding_rate"])
        new_cols["ms_funding_magnitude"] = df["dt_funding_rate"].abs()

    if "dt_open_interest" in df.columns and "dt_open_interest_lag_5m" in df.columns:
        new_cols["ms_oi_change_5m"] = (df["dt_open_interest"] - df["dt_open_interest_lag_5m"]) / df["dt_open_interest_lag_5m"].replace(0, np.nan)

    # Cross-exchange premium
    cross_exchange_cols = [c for c in df.columns if c.startswith("ce_")]
    for c in cross_exchange_cols[:3]:
        if "premium" in c.lower() or "spread" in c.lower():
            new_cols[f"ms_{c}_abs"] = df[c].abs()

    new_df = df.copy()
    for col, values in new_cols.items():
        new_df[col] = values

    return new_df


def get_microstructure_feature_names() -> list:
    return [
        "ms_buy_ratio_momentum_300_900", "ms_buy_ratio_accel",
        "ms_ofi_300", "ms_ofi_ratio_300", "ms_ofi_900",
        "ms_depth_ratio_5", "ms_depth_ratio_10", "ms_depth_ratio_50",
        "ms_bid_concentration", "ms_ask_concentration",
        "ms_spread_change_5m",
        "ms_large_trade_ratio_300", "ms_avg_size_momentum",
        "ms_liq_acceleration", "ms_liq_direction_shift",
        "ms_vol_acceleration",
        "ms_funding_sign", "ms_funding_magnitude",
        "ms_oi_change_5m",
    ]
