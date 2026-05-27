"""
Stage 2: Feature × Target correlation analysis.

For each (feature, target) pair:
  - Pearson r, p-value
  - Spearman r, p-value

Multiple testing correction:
  - Bonferroni: alpha / N
  - FDR (Benjamini-Hochberg): largest k where p(k) <= (k/N) * alpha

Note: 1-second time series has high auto-correlation. p-values may be
inflated. Treat as preliminary scan; validate with effect size and OOS.
"""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range


def apply_bh_fdr(p_values: pd.Series, alpha: float = 0.05) -> pd.Series:
    """
    Benjamini-Hochberg FDR.

    Returns boolean Series same length as p_values, True for rejected.
    """
    n = len(p_values)
    if n == 0:
        return pd.Series([], dtype=bool)

    # Sort, compute threshold, find max k passing
    sorted_idx = p_values.sort_values().index
    sorted_p = p_values.loc[sorted_idx].values
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * alpha

    passes = sorted_p <= thresholds
    if not passes.any():
        return pd.Series(False, index=p_values.index)

    max_k = ranks[passes].max()

    # Mark all with rank <= max_k as pass
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
        log.error("scipy required: pip install scipy")
        return

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)

    log.info("Loading integrated dataset")
    df = read_range("integrated_dataset", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(df)} rows × {len(df.columns)} cols")

    if len(df) < 1000:
        log.warning("not enough data")
        return

    feature_cols = [c for c in df.columns if c != "timestamp" and not c.startswith("target_")]
    target_cols = [c for c in df.columns if c.startswith("target_")]

    log.info(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}")
    log.info(f"Total combinations: {len(feature_cols) * len(target_cols)}")

    results = []
    skipped = 0

    for feat in feature_cols:
        for targ in target_cols:
            valid = df[[feat, targ]].dropna()
            if len(valid) < 100:
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
            except Exception as e:
                log.warning(f"{feat} × {targ}: {e}")
                skipped += 1
                continue

    res = pd.DataFrame(results)
    log.info(f"Computed {len(res)} valid pairs (skipped {skipped})")

    if len(res) == 0:
        log.error("no valid pairs")
        return

    n_tests = len(res)
    bonferroni_alpha = 0.05 / n_tests

    # Bonferroni
    res["bonferroni_pass_pearson"] = res["pearson_p"] < bonferroni_alpha
    res["bonferroni_pass_spearman"] = res["spearman_p"] < bonferroni_alpha

    # FDR (Benjamini-Hochberg)
    res["pearson_fdr_pass"] = apply_bh_fdr(res["pearson_p"], alpha=0.05)
    res["spearman_fdr_pass"] = apply_bh_fdr(res["spearman_p"], alpha=0.05)

    print()
    print("=" * 80)
    print("STAGE 2: Feature × Target Correlation Matrix")
    print("=" * 80)
    print(f"\nTotal pairs: {n_tests}")
    print(f"Bonferroni alpha: {bonferroni_alpha:.2e}")
    print(f"\n[Auto-correlation warning] 1s grid → p-values may be inflated.")
    print(f"Treat as preliminary; rely on effect size + future OOS validation.")
    print()

    # P-value distribution
    print("=" * 80)
    print("P-VALUE DISTRIBUTION (Pearson)")
    print("=" * 80)
    p_lt_05 = (res['pearson_p'] < 0.05).sum()
    p_lt_01 = (res['pearson_p'] < 0.01).sum()
    p_lt_001 = (res['pearson_p'] < 0.001).sum()

    print(f"  p < 0.05:    {p_lt_05} ({p_lt_05/n_tests*100:.1f}%)")
    print(f"  p < 0.01:    {p_lt_01} ({p_lt_01/n_tests*100:.1f}%)")
    print(f"  p < 0.001:   {p_lt_001} ({p_lt_001/n_tests*100:.1f}%)")
    print(f"  Bonferroni:  {res['bonferroni_pass_pearson'].sum()}")
    print(f"  FDR<0.05:    {res['pearson_fdr_pass'].sum()}")

    expected_05 = n_tests * 0.05
    print(f"\n  vs random expectation:")
    print(f"    p<0.05: actual {p_lt_05} vs expected {expected_05:.0f} → {p_lt_05/expected_05:.2f}x")

    # Top by absolute Pearson r
    print()
    print("=" * 80)
    print("TOP 30 by |Pearson r|")
    print("=" * 80)
    res_sorted = res.copy()
    res_sorted["abs_pearson"] = res_sorted["pearson_r"].abs()
    top = res_sorted.sort_values("abs_pearson", ascending=False).head(30)

    print(f"{'feature':<42} {'target':<28} {'r':>8} {'p':>10} {'n':>7} {'FDR':>5} {'Bonf':>5}")
    print("-" * 110)
    for _, row in top.iterrows():
        feat = row['feature'][:41]
        targ = row['target'][:27]
        fdr = "Y" if row['pearson_fdr_pass'] else " "
        bonf = "Y" if row['bonferroni_pass_pearson'] else " "
        print(f"{feat:<42} {targ:<28} {row['pearson_r']:+.4f} {row['pearson_p']:.2e} {int(row['n']):>7} {fdr:>5} {bonf:>5}")

    # Top by absolute Spearman r
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

    # Summary by feature category
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
        n_bonf_pass=("bonferroni_pass_pearson", "sum"),
    )
    print(summary.to_string())

    # Save
    out_path = Path("data/analysis_results")
    out_path.mkdir(exist_ok=True, parents=True)
    res.to_parquet(out_path / "stage2_correlation_matrix.parquet")
    res.to_csv(out_path / "stage2_correlation_matrix.csv", index=False)
    log.info(f"Saved: {out_path}/stage2_correlation_matrix.parquet")


if __name__ == "__main__":
    main()
