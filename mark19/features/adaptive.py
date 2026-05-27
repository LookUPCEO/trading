"""Adaptive features: Rolling z-score + Relative features.

목적: Covariate shift (시기별 시장 변화) 보정
- 절대값 features → 상대값 features
- Train/Test 의 distribution 차이 자연 해결
"""
import numpy as np
import pandas as pd


def add_adaptive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add adaptive (rolling-based) features."""

    new_cols = {}

    WINDOW_1H = 60       # 1 hour (1-min granularity)
    WINDOW_1D = 1440     # 1 day
    WINDOW_7D = 1440 * 7 # 7 days

    # CATEGORY 1: Volume relative
    if "tr_total_volume_300s" in df.columns:
        rolling_mean_1h = df["tr_total_volume_300s"].rolling(WINDOW_1H, min_periods=10).mean()
        new_cols["adapt_volume_rel_1h"] = df["tr_total_volume_300s"] / rolling_mean_1h.replace(0, np.nan)

        rolling_mean_1d = df["tr_total_volume_300s"].rolling(WINDOW_1D, min_periods=60).mean()
        new_cols["adapt_volume_rel_1d"] = df["tr_total_volume_300s"] / rolling_mean_1d.replace(0, np.nan)

    # CATEGORY 2: Volatility relative
    if "ob_mid_price_std_300s" in df.columns:
        rolling_vol_1d = df["ob_mid_price_std_300s"].rolling(WINDOW_1D, min_periods=60).mean()
        new_cols["adapt_vol_rel_1d"] = df["ob_mid_price_std_300s"] / rolling_vol_1d.replace(0, np.nan)

        rolling_vol_mean = df["ob_mid_price_std_300s"].rolling(WINDOW_1D, min_periods=60).mean()
        rolling_vol_std = df["ob_mid_price_std_300s"].rolling(WINDOW_1D, min_periods=60).std()
        new_cols["adapt_vol_zscore_1d"] = (df["ob_mid_price_std_300s"] - rolling_vol_mean) / rolling_vol_std.replace(0, np.nan)

    # CATEGORY 3: Spread relative
    if "ob_spread_pct" in df.columns:
        rolling_spread_mean = df["ob_spread_pct"].rolling(WINDOW_1D, min_periods=60).mean()
        new_cols["adapt_spread_rel_1d"] = df["ob_spread_pct"] / rolling_spread_mean.replace(0, np.nan)

    # CATEGORY 4: Depth relative
    if "ob_total_depth_50" in df.columns:
        rolling_depth_mean = df["ob_total_depth_50"].rolling(WINDOW_1D, min_periods=60).mean()
        new_cols["adapt_depth_rel_1d"] = df["ob_total_depth_50"] / rolling_depth_mean.replace(0, np.nan)

        rolling_depth_mean_7d = df["ob_total_depth_50"].rolling(WINDOW_7D, min_periods=1440).mean()
        rolling_depth_std_7d = df["ob_total_depth_50"].rolling(WINDOW_7D, min_periods=1440).std()
        new_cols["adapt_depth_zscore_7d"] = (df["ob_total_depth_50"] - rolling_depth_mean_7d) / rolling_depth_std_7d.replace(0, np.nan)

    # CATEGORY 5: Trade activity relative
    if "tr_trades_per_sec_300s" in df.columns:
        rolling_freq_mean = df["tr_trades_per_sec_300s"].rolling(WINDOW_1D, min_periods=60).mean()
        new_cols["adapt_trade_freq_rel_1d"] = df["tr_trades_per_sec_300s"] / rolling_freq_mean.replace(0, np.nan)

        rolling_freq_std = df["tr_trades_per_sec_300s"].rolling(WINDOW_1D, min_periods=60).std()
        new_cols["adapt_trade_freq_zscore_1d"] = (df["tr_trades_per_sec_300s"] - rolling_freq_mean) / rolling_freq_std.replace(0, np.nan)

    # CATEGORY 6: OBI percentile (rolling)
    if "ob_obi_top1" in df.columns:
        new_cols["adapt_obi_top1_pct_1d"] = df["ob_obi_top1"].rolling(WINDOW_1D, min_periods=60).rank(pct=True)

        obi_abs = df["ob_obi_top1"].abs()
        rolling_obi_abs_mean = obi_abs.rolling(WINDOW_1D, min_periods=60).mean()
        rolling_obi_abs_std = obi_abs.rolling(WINDOW_1D, min_periods=60).std()
        new_cols["adapt_obi_abs_zscore_1d"] = (obi_abs - rolling_obi_abs_mean) / rolling_obi_abs_std.replace(0, np.nan)

    # CATEGORY 7: Liquidation relative
    if "liq_liq_count_300s" in df.columns:
        rolling_liq_mean = df["liq_liq_count_300s"].rolling(WINDOW_1D, min_periods=60).mean()
        new_cols["adapt_liq_count_rel_1d"] = df["liq_liq_count_300s"] / rolling_liq_mean.replace(0, np.nan).fillna(1)

    # CATEGORY 8: Funding rate z-score
    if "dt_funding_rate" in df.columns:
        rolling_funding_mean = df["dt_funding_rate"].rolling(WINDOW_1D, min_periods=60).mean()
        rolling_funding_std = df["dt_funding_rate"].rolling(WINDOW_1D, min_periods=60).std()
        new_cols["adapt_funding_zscore_1d"] = (df["dt_funding_rate"] - rolling_funding_mean) / rolling_funding_std.replace(0, np.nan)

    # CATEGORY 9: Open Interest relative
    if "dt_open_interest" in df.columns:
        rolling_oi_mean_1d = df["dt_open_interest"].rolling(WINDOW_1D, min_periods=60).mean()
        new_cols["adapt_oi_rel_1d"] = df["dt_open_interest"] / rolling_oi_mean_1d.replace(0, np.nan)

        rolling_oi_mean_7d = df["dt_open_interest"].rolling(WINDOW_7D, min_periods=1440).mean()
        rolling_oi_std_7d = df["dt_open_interest"].rolling(WINDOW_7D, min_periods=1440).std()
        new_cols["adapt_oi_zscore_7d"] = (df["dt_open_interest"] - rolling_oi_mean_7d) / rolling_oi_std_7d.replace(0, np.nan)

    new_df = df.copy()
    for col, values in new_cols.items():
        new_df[col] = values

    return new_df


def get_adaptive_feature_names() -> list:
    return [
        "adapt_volume_rel_1h", "adapt_volume_rel_1d",
        "adapt_vol_rel_1d", "adapt_vol_zscore_1d",
        "adapt_spread_rel_1d",
        "adapt_depth_rel_1d", "adapt_depth_zscore_7d",
        "adapt_trade_freq_rel_1d", "adapt_trade_freq_zscore_1d",
        "adapt_obi_top1_pct_1d", "adapt_obi_abs_zscore_1d",
        "adapt_liq_count_rel_1d",
        "adapt_funding_zscore_1d",
        "adapt_oi_rel_1d", "adapt_oi_zscore_7d",
    ]
