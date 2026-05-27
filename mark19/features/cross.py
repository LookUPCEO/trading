"""Cross-feature interactions for direction prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-feature interactions to dataframe."""

    new_cols = {}

    # Order Book × Trade activity
    if "ob_obi_top1" in df.columns and "tr_trades_per_sec_300s" in df.columns:
        new_cols["cross_obi1_x_trade_freq_300"] = df["ob_obi_top1"] * df["tr_trades_per_sec_300s"]

    if "ob_obi_top5" in df.columns and "tr_trades_per_sec_300s" in df.columns:
        new_cols["cross_obi5_x_trade_freq_300"] = df["ob_obi_top5"] * df["tr_trades_per_sec_300s"]

    if "ob_obi_top1" in df.columns and "tr_volume_imbalance_300s" in df.columns:
        new_cols["cross_obi1_x_volimb_300"] = df["ob_obi_top1"] * df["tr_volume_imbalance_300s"]

    if "ob_obi_top5" in df.columns and "tr_volume_imbalance_900s" in df.columns:
        new_cols["cross_obi5_x_volimb_900"] = df["ob_obi_top5"] * df["tr_volume_imbalance_900s"]

    # Liquidation × Order Book
    if "liq_liq_count_300s" in df.columns and "ob_obi_top1" in df.columns:
        new_cols["cross_liq_x_obi1_300"] = df["liq_liq_count_300s"] * df["ob_obi_top1"]

    if "liq_liq_buy_ratio_count_300s" in df.columns and "ob_obi_top1" in df.columns:
        new_cols["cross_liq_buyratio_x_obi1"] = (df["liq_liq_buy_ratio_count_300s"] - 0.5) * df["ob_obi_top1"]

    # Funding × OI
    if "dt_funding_rate" in df.columns and "dt_open_interest" in df.columns:
        new_cols["cross_funding_x_oi"] = df["dt_funding_rate"] * df["dt_open_interest"]

    # Trade × Volatility regime
    if "tr_trades_per_sec_300s" in df.columns and "ob_mid_price_std_300s" in df.columns:
        new_cols["cross_trade_x_vol_300"] = df["tr_trades_per_sec_300s"] * df["ob_mid_price_std_300s"]

    # Multi-level OBI consensus
    obi_levels = [c for c in ["ob_obi_top1", "ob_obi_top5", "ob_obi_top10"] if c in df.columns]
    if len(obi_levels) >= 2:
        signs = df[obi_levels].apply(np.sign)
        consensus = signs.abs().sum(axis=1) == len(obi_levels)
        new_cols["cross_obi_consensus"] = consensus.astype(int) * np.sign(df[obi_levels[0]])
        new_cols["cross_obi_mean"] = df[obi_levels].mean(axis=1)

    # Lag interactions
    if "ob_obi_top1" in df.columns and "ob_obi_top1_lag_1m" in df.columns:
        new_cols["cross_obi1_change_1m"] = df["ob_obi_top1"] - df["ob_obi_top1_lag_1m"]

    if "ob_obi_top1" in df.columns and "ob_obi_top1_lag_5m" in df.columns:
        new_cols["cross_obi1_change_5m"] = df["ob_obi_top1"] - df["ob_obi_top1_lag_5m"]

    if "tr_trades_per_sec_300s" in df.columns and "tr_trades_per_sec_300s_lag_5m" in df.columns:
        new_cols["cross_trade_freq_change_5m"] = df["tr_trades_per_sec_300s"] / df["tr_trades_per_sec_300s_lag_5m"].replace(0, np.nan)

    # Spread × Volume
    if "ob_spread_pct" in df.columns and "tr_total_volume_300s" in df.columns:
        new_cols["cross_spread_x_volume"] = df["ob_spread_pct"] * np.log1p(df["tr_total_volume_300s"])

    # Depth × OBI
    if "ob_total_depth_50" in df.columns and "ob_obi_top5" in df.columns:
        new_cols["cross_depth50_x_obi5"] = np.log1p(df["ob_total_depth_50"]) * df["ob_obi_top5"]

    new_df = df.copy()
    for col, values in new_cols.items():
        new_df[col] = values

    return new_df


def get_cross_feature_names() -> list:
    return [
        "cross_obi1_x_trade_freq_300", "cross_obi5_x_trade_freq_300",
        "cross_obi1_x_volimb_300", "cross_obi5_x_volimb_900",
        "cross_liq_x_obi1_300", "cross_liq_buyratio_x_obi1",
        "cross_funding_x_oi",
        "cross_trade_x_vol_300",
        "cross_obi_consensus", "cross_obi_mean",
        "cross_obi1_change_1m", "cross_obi1_change_5m",
        "cross_trade_freq_change_5m",
        "cross_spread_x_volume",
        "cross_depth50_x_obi5",
    ]
