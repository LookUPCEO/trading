"""
Stage 2.5: Detrending + Sub-sampling.

- Sub-sample 1s grid to 1min (every 60th row)
- Drop raw price features (auto-correlation source)
- Keep transformations: spread, OBI, zscore, funding, depths, etc.
- Add 1m and 5m mid_price returns as new targets
"""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range


# Features to EXCLUDE: raw price level (auto-correlated with itself)
PRICE_RAW_PATTERNS = [
    "ob_mid_price",
    "ob_mid_price_mean_",
    "tr_vwap",
    "cx_price_mean_3ex",
    "cx_price_std_3ex",
    "cx_bybit_eth_usd",
    "cx_binance_eth_usd",
    "cx_okx_eth_usd",
    "cx_upbit_eth_krw",
]


def is_price_raw(col_name: str) -> bool:
    for pat in PRICE_RAW_PATTERNS:
        if col_name == pat or col_name.startswith(pat):
            return True
    return False


def apply_bh_fdr(p_values: pd.Series, alpha: float = 0.05) -> pd.Series:
    n = len(p_values)
    if n == 0:
        return pd.Series([], dtype=bool)
    sorted_idx = p_values.sort_values().index
    sorted_p = p_values.loc[sorted_idx].values
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * alpha
    passes = sorted_p <= thresholds
    if not passes.any():
        return pd.Series(False, index=p_values.index)
    max_k = ranks[passes].max()
    result = pd.Series(False, index=p_values.index)
    pass_idx = sorted_idx[:max_k]
    result.loc[pass_idx] = True
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    try:
        from scipy import stats as scistats
    except ImportError:
        log.error("scipy required")
        return

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)

    log.info("Loading integrated dataset")
    df = read_range("integrated_dataset", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(df)} rows × {len(df.columns)} cols")

    # Sub-sample
    log.info("Sub-sampling to 1 minute (iloc[::60])")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_1min = df.iloc[::60].copy().reset_index(drop=True)
    log.info(f"  sub-sampled: {len(df_1min)} rows")

    # Sanity check sub-sample interval
    diffs = df_1min["timestamp"].diff().dropna()
    if len(diffs) > 0:
        median_int = diffs.median().total_seconds()
        max_int = diffs.max().total_seconds()
        log.info(f"  interval: median={median_int}s, max={max_int}s")
        if median_int != 60:
            log.warning(f"  expected 60s median, got {median_int}s")

    # Add 1m and 5m returns (use ob_mid_price before excluding it)
    if "ob_mid_price" in df_1min.columns:
        mid = df_1min["ob_mid_price"]
        df_1min["target_return_1m"] = (mid.shift(-1) - mid) / mid * 100
        df_1min["target_return_5m"] = (mid.shift(-5) - mid) / mid * 100

    # Identify columns
    all_cols = [c for c in df_1min.columns if c != "timestamp"]
    target_cols = [c for c in all_cols if c.startswith("target_")]
    feature_cols_all = [c for c in all_cols if not c.startswith("target_")]

    feature_cols = [c for c in feature_cols_all if not is_price_raw(c)]
    dropped = [c for c in feature_cols_all if is_price_raw(c)]

    log.info(f"Features: {len(feature_cols_all)} → {len(feature_cols)} after price-raw drop")
    log.info(f"Dropped {len(dropped)}: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
    log.info(f"Targets: {len(target_cols)}: {target_cols}")

    # Compute correlations
    log.info(f"Computing {len(feature_cols) * len(target_cols)} correlations")

    results = []
    skipped = 0

    for feat in feature_cols:
        for targ in target_cols:
            valid = df_1min[[feat, targ]].dropna()
            if len(valid) < 50:
                skipped += 1
                continue
            if valid[feat].std() < 1e-10 or valid[targ].std() < 1e-10:
                skipped += 1
                continue

            try:
                pearson_r, pearson_p = scistats.pearsonr(valid[feat], valid[targ])
                spearman_r, spearman_p = scistats.spearmanr(valid[feat], valid[targ])

                results.append({
                    "feature": feat,
                    "target": targ,
                    "n": len(valid),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                })
            except Exception:
                skipped += 1
                continue

    res = pd.DataFrame(results)
    log.info(f"Computed {len(res)} pairs (skipped {skipped})")

    if len(res) == 0:
        return

    n_tests = len(res)
    bonferroni_alpha = 0.05 / n_tests

    res["bonferroni_pass_pearson"] = res["pearson_p"] < bonferroni_alpha
    res["bonferroni_pass_spearman"] = res["spearman_p"] < bonferroni_alpha
    res["pearson_fdr_pass"] = apply_bh_fdr(res["pearson_p"])
    res["spearman_fdr_pass"] = apply_bh_fdr(res["spearman_p"])

    print()
    print("=" * 80)
    print("STAGE 2.5: Detrend + 1min Sub-sample")
    print("=" * 80)
    print(f"\nSub-sampled rows: {len(df_1min)}")
    print(f"Features (price-raw dropped): {len(feature_cols)}")
    print(f"Total pairs: {n_tests}")
    print(f"Bonferroni alpha: {bonferroni_alpha:.2e}")
    print()

    p_lt_05 = (res['pearson_p'] < 0.05).sum()
    p_lt_01 = (res['pearson_p'] < 0.01).sum()
    p_lt_001 = (res['pearson_p'] < 0.001).sum()

    print("=" * 80)
    print("P-VALUE DISTRIBUTION (Pearson)")
    print("=" * 80)
    print(f"  p < 0.05:    {p_lt_05} ({p_lt_05/n_tests*100:.1f}%)")
    print(f"  p < 0.01:    {p_lt_01} ({p_lt_01/n_tests*100:.1f}%)")
    print(f"  p < 0.001:   {p_lt_001} ({p_lt_001/n_tests*100:.1f}%)")
    print(f"  Bonferroni:  {res['bonferroni_pass_pearson'].sum()}")
    print(f"  FDR<0.05:    {res['pearson_fdr_pass'].sum()}")

    expected_05 = n_tests * 0.05
    print(f"\n  vs random expectation:")
    print(f"    p<0.05: {p_lt_05} vs expected {expected_05:.0f} -> {p_lt_05/expected_05:.2f}x")
    print(f"    Stage 2 (1s, with prices): 17.94x")
    print()

    # Top by abs Pearson
    print("=" * 80)
    print("TOP 30 by |Pearson r|")
    print("=" * 80)
    res_sorted = res.copy()
    res_sorted["abs_pearson"] = res_sorted["pearson_r"].abs()
    top = res_sorted.sort_values("abs_pearson", ascending=False).head(30)

    print(f"{'feature':<42} {'target':<28} {'r':>8} {'p':>10} {'n':>6} {'FDR':>5} {'Bonf':>5}")
    print("-" * 110)
    for _, row in top.iterrows():
        feat = row['feature'][:41]
        targ = row['target'][:27]
        fdr = "Y" if row['pearson_fdr_pass'] else " "
        bonf = "Y" if row['bonferroni_pass_pearson'] else " "
        print(f"{feat:<42} {targ:<28} {row['pearson_r']:+.4f} {row['pearson_p']:.2e} {int(row['n']):>6} {fdr:>5} {bonf:>5}")

    # Top Spearman
    print()
    print("=" * 80)
    print("TOP 20 by |Spearman r|")
    print("=" * 80)
    res_sp = res.copy()
    res_sp["abs_spearman"] = res_sp["spearman_r"].abs()
    top_sp = res_sp.sort_values("abs_spearman", ascending=False).head(20)

    print(f"{'feature':<42} {'target':<28} {'r':>8} {'p':>10} {'FDR':>5}")
    print("-" * 95)
    for _, row in top_sp.iterrows():
        feat = row['feature'][:41]
        targ = row['target'][:27]
        fdr = "Y" if row['spearman_fdr_pass'] else " "
        print(f"{feat:<42} {targ:<28} {row['spearman_r']:+.4f} {row['spearman_p']:.2e} {fdr:>5}")

    # Summary by category
    print()
    print("=" * 80)
    print("SUMMARY BY FEATURE CATEGORY")
    print("=" * 80)

    def get_prefix(col):
        for p in ["ob_", "tr_", "liq_", "cx_", "cf_"]:
            if col.startswith(p):
                return p[:-1]
        return "other"

    res["feature_prefix"] = res["feature"].apply(get_prefix)
    summary = res.groupby("feature_prefix").agg(
        total_pairs=("pearson_r", "size"),
        max_abs_pearson=("pearson_r", lambda x: x.abs().max()),
        mean_abs_pearson=("pearson_r", lambda x: x.abs().mean()),
        n_p_05=("pearson_p", lambda x: (x < 0.05).sum()),
        n_fdr_pass=("pearson_fdr_pass", "sum"),
    )
    print(summary.to_string())

    # Save
    out_path = Path("data/analysis_results")
    out_path.mkdir(exist_ok=True, parents=True)
    res.to_parquet(out_path / "stage2_5_detrended.parquet")
    res.to_csv(out_path / "stage2_5_detrended.csv", index=False)
    log.info(f"Saved: {out_path}/stage2_5_detrended.parquet")


if __name__ == "__main__":
    main()
