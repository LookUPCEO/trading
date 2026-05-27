"""SHADOW direction 진단 v2 — Self 데이터만 (Tardis SSD unavailable)."""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset

_spec36 = importlib.util.spec_from_file_location("_sido36", _HERE / "sido36_regime_invariant.py")
_mod36 = importlib.util.module_from_spec(_spec36)
_spec36.loader.exec_module(_mod36)
add_normalized_features = _mod36.add_normalized_features
HIGH_SHIFT_FEATURES = _mod36.HIGH_SHIFT_FEATURES


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("DIAGNOSE v2: SHADOW direction SHORT-zero (Self-only)")
    log.info("=" * 70)

    bundle = joblib.load("/Users/dohun/Desktop/Mark/mark19/models/mark36_v1.joblib")
    LRV = bundle["lr_vol"]; SV = bundle["scaler_vol"]
    XGBD = bundle["xgb_dir"]
    FEAT_COLS = bundle["feature_cols"]
    HIGH_SHIFT = bundle["high_shift_features"]
    TRAIN_MEDIANS = pd.Series(bundle["train_medians"])
    log.info(f"  Model: {len(FEAT_COLS)} features, best_iter {XGBD.best_iteration}")

    SELF_DATES = [f"2026-04-{d:02d}" for d in range(21, 31)] + \
                 [f"2026-05-{d:02d}" for d in range(1, 10)]
    log.info(f"\nBuilding Self {len(SELF_DATES)} dates (no Tardis fallback medians since SSD unmounted)")

    # Use mark36's train_medians as fallback medians
    self_dfs = {}
    for d in SELF_DATES:
        try:
            df = build_self_date_dataset(d, log, train_medians=TRAIN_MEDIANS)
            df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
            if len(df) > 0:
                self_dfs[d] = df
                log.info(f"  Self {d}: {len(df)} rows")
        except Exception as e:
            log.warning(f"  Self {d}: {e}")

    if not self_dfs:
        log.error("No Self data!"); return

    log.info("\nApplying day-mean normalization (sido36 method)...")
    for d in self_dfs:
        self_dfs[d] = add_normalized_features(self_dfs[d], log)

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"

    norm_features = [c for c in next(iter(self_dfs.values())).columns if c.endswith("_norm")]
    log.info(f"  Norm features: {len(norm_features)}")

    # ---- A. Per-day inference dir_proba distribution ----
    print()
    print("=" * 80)
    print("[A] Per-day DIR_PROBA distribution (mark36_v1 inference)")
    print("=" * 80)
    print(f"\n{'Date':<14} {'N':<8} {'min':<8} {'q25':<8} {'med':<8} {'q75':<8} {'max':<8} {'>0.55':<8} {'<0.45':<8}")
    print("-" * 90)

    period_results = {}
    for d in sorted(self_dfs.keys()):
        df = self_dfs[d]
        # Use bundle's feature_cols as the feature set
        meds = TRAIN_MEDIANS.reindex(FEAT_COLS)
        X = df.reindex(columns=FEAT_COLS).copy().replace([np.inf, -np.inf], np.nan).fillna(meds).fillna(0)
        if X.shape[0] == 0: continue
        try:
            dir_proba = XGBD.predict_proba(X.values)[:, 1]
        except Exception as e:
            log.error(f"  predict {d}: {e}")
            continue
        period_results[d] = {
            "n": len(df), "min": float(dir_proba.min()), "max": float(dir_proba.max()),
            "q25": float(np.quantile(dir_proba, 0.25)), "med": float(np.median(dir_proba)),
            "q75": float(np.quantile(dir_proba, 0.75)),
            "above_055": float((dir_proba > 0.55).mean()),
            "below_045": float((dir_proba < 0.45).mean()),
            "actual_long": float((df[dir_target] > 0.20).mean()),
            "actual_short": float((df[dir_target] < -0.20).mean()),
        }
        r = period_results[d]
        print(f"{d:<14} {r['n']:<8} {r['min']:<8.4f} {r['q25']:<8.4f} {r['med']:<8.4f} {r['q75']:<8.4f} {r['max']:<8.4f} {r['above_055']*100:<8.1f}% {r['below_045']*100:<8.1f}%")

    # Period split: backtest (4/21-30) vs SHADOW (5/2-5/9)
    bt_dates = [d for d in period_results if d <= "2026-04-30"]
    sh_dates = [d for d in period_results if d >= "2026-05-02"]

    def aggregate(date_list, label):
        if not date_list: return None
        ns = [period_results[d]["n"] for d in date_list]
        a55 = sum(period_results[d]["above_055"] * period_results[d]["n"] for d in date_list) / sum(ns)
        b45 = sum(period_results[d]["below_045"] * period_results[d]["n"] for d in date_list) / sum(ns)
        actual_l = sum(period_results[d]["actual_long"] * period_results[d]["n"] for d in date_list) / sum(ns)
        actual_s = sum(period_results[d]["actual_short"] * period_results[d]["n"] for d in date_list) / sum(ns)
        meds = [period_results[d]["med"] for d in date_list]
        print(f"\n  {label} ({len(date_list)} days, {sum(ns)} rows):")
        print(f"    median dir_proba range: {min(meds):.4f}~{max(meds):.4f}")
        print(f"    >0.55: {a55*100:.2f}%  (LONG signal rate)")
        print(f"    <0.45: {b45*100:.2f}%  (SHORT signal rate)")
        print(f"    actual LONG (>{0.20}%): {actual_l*100:.2f}%")
        print(f"    actual SHORT (<-{0.20}%): {actual_s*100:.2f}%")
        return {"above_055": a55, "below_045": b45, "actual_long": actual_l, "actual_short": actual_s}

    bt_agg = aggregate(bt_dates, "Backtest period 4/21-30")
    sh_agg = aggregate(sh_dates, "SHADOW period 5/2-5/9")

    # ---- B. Class balance check (Self all rows) ----
    print()
    print("=" * 80)
    print("[B] Self DATA class balance (4/21-5/9)")
    print("=" * 80)
    all_self = pd.concat([self_dfs[d] for d in self_dfs], ignore_index=True)
    n_total = len(all_self)
    n_long = (all_self[dir_target] > 0.20).sum()
    n_short = (all_self[dir_target] < -0.20).sum()
    print(f"  Total: {n_total}, LONG (>0.20%): {n_long} ({n_long/n_total*100:.1f}%)")
    print(f"  SHORT (<-0.20%): {n_short} ({n_short/n_total*100:.1f}%)")
    print(f"  ratio L/S: {n_long/max(n_short,1):.2f}")

    # ---- C. ETH 5/2-5/9 market regime ----
    print()
    print("=" * 80)
    print("[C] ETH market regime 5/2-5/9")
    print("=" * 80)
    if sh_dates:
        sh_df = pd.concat([self_dfs[d] for d in sh_dates], ignore_index=True)
        rs = sh_df[dir_target].values
        print(f"  rows: {len(sh_df)}")
        print(f"  mean: {rs.mean():+.4f}%, std: {rs.std():.4f}")
        print(f"  >0%: {(rs > 0).mean()*100:.1f}%")
        print(f"  >0.20%: {(rs > 0.20).sum()} ({(rs > 0.20).mean()*100:.2f}%)")
        print(f"  <-0.20%: {(rs < -0.20).sum()} ({(rs < -0.20).mean()*100:.2f}%)")
        trend = rs.sum() / max(np.abs(rs).sum(), 1)
        print(f"  trend (signed/abs): {trend:.3f}")
        if abs(trend) < 0.10: print("  → CHOP (no clear trend)")
        elif trend > 0.10: print("  → UPTREND")
        else: print("  → DOWNTREND")

    # ---- D. Lookahead bias: live-style vs backtest-style normalization ----
    print()
    print("=" * 80)
    print("[D] Lookahead bias check (live vs backtest norm)")
    print("=" * 80)
    if "2026-05-05" in self_dfs:
        d = "2026-05-05"
        df = self_dfs[d].sort_values("_ts").reset_index(drop=True)
        # Backtest-style: full-day mean already in df["{f}_norm"]
        # Live-style: cumulative session mean (expanding)
        sample_feat = "ob_total_depth_10"  # one of high-shift features
        if sample_feat in df.columns:
            bt_norm_col = f"{sample_feat}_norm"
            if bt_norm_col in df.columns:
                # Compute live-style
                live_mean = df[sample_feat].expanding().mean()
                live_norm = df[sample_feat] / live_mean.replace(0, np.nan)
                bt_norm = df[bt_norm_col]
                # Compare at start, mid, end of day
                for label, idx in [("first 10 rows", slice(0, 10)),
                                    ("middle (rows 500-510)", slice(500, 510)),
                                    ("last 10 rows", slice(-10, None))]:
                    print(f"\n  {sample_feat} on {d} - {label}:")
                    bt_v = bt_norm.iloc[idx]
                    live_v = live_norm.iloc[idx]
                    print(f"    backtest_norm (full-day mean): {bt_v.mean():.4f} ± {bt_v.std():.4f}")
                    print(f"    live_norm (cumulative):        {live_v.mean():.4f} ± {live_v.std():.4f}")
                    print(f"    diff: {(bt_v.mean() - live_v.mean()):+.4f}")

    out = {
        "period_results": period_results,
        "backtest_aggregate": bt_agg,
        "shadow_aggregate": sh_agg,
        "self_class_balance": {"total": int(n_total), "long": int(n_long), "short": int(n_short),
                                "long_ratio": float(n_long / max(n_total, 1))},
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/diagnose_shadow_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # ---- Diagnosis ----
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if bt_agg and sh_agg:
        print(f"\n  Backtest > 0.55: {bt_agg['above_055']*100:.2f}%, < 0.45: {bt_agg['below_045']*100:.2f}%")
        print(f"  SHADOW > 0.55: {sh_agg['above_055']*100:.2f}%, < 0.45: {sh_agg['below_045']*100:.2f}%")

        if sh_agg["below_045"] < 0.005:
            print(f"\n  🚨 SHORT signal 거의 0 ({sh_agg['below_045']*100:.3f}%)")

        # Backtest also low?
        if bt_agg["below_045"] < 0.05 and sh_agg["below_045"] < 0.005:
            print(f"  → Backtest 도 낮은 SHORT 비율 (모델 자체 LONG bias)")

        # Compare actual
        if sh_agg["actual_short"] > 0.10 and sh_agg["below_045"] < 0.01:
            print(f"  ⚠️  실제 SHORT 시장 {sh_agg['actual_short']*100:.1f}% but model SHORT 신호 {sh_agg['below_045']*100:.2f}% → MISS")

    log.info("\n진단 v2 complete")


if __name__ == "__main__":
    main()
