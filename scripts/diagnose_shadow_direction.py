"""SHADOW direction 88h 진단: SHORT-zero 원인 + 5 항목 분석."""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns
from mark19.storage import read_range
from live_bot.parquet_retry import read_parquet_with_retry

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
    log.info("DIAGNOSE: SHADOW direction SHORT-zero")
    log.info("=" * 70)

    # ---- 1. Load model + check ----
    bundle = joblib.load("/Users/dohun/Desktop/Mark/mark19/models/mark36_v1.joblib")
    LRV = bundle["lr_vol"]; SV = bundle["scaler_vol"]
    XGBD = bundle["xgb_dir"]
    FEAT_COLS = bundle["feature_cols"]
    HIGH_SHIFT = bundle["high_shift_features"]
    TRAIN_MEDIANS = pd.Series(bundle["train_medians"])
    log.info(f"  Model loaded: {len(FEAT_COLS)} features, {len(HIGH_SHIFT)} high_shift, n_estimators best_iter {XGBD.best_iteration}")

    # ---- 2. Build Tardis data for backtest period analysis (full sample) ----
    log.info("\n[A] Building Tardis (for train balance check)...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"], inplace=True)
    tardis_train = add_normalized_features(tardis_train, log)

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    T = 0.20

    # ---- 3. Train direction balance ----
    print()
    print("=" * 80)
    print("[A] TRAIN DATA DIRECTION BALANCE (Tardis 26 dates)")
    print("=" * 80)
    n_total = len(tardis_train)
    n_above_T = (tardis_train[dir_target].abs() > T).sum()
    n_long = (tardis_train[dir_target] > T).sum()
    n_short = (tardis_train[dir_target] < -T).sum()
    print(f"  Total rows: {n_total}")
    print(f"  Above |T|={T}%: {n_above_T} ({n_above_T/n_total*100:.1f}%)")
    print(f"  LONG samples (>{T}%):  {n_long} ({n_long/n_above_T*100:.1f}% of above-T)")
    print(f"  SHORT samples (<-{T}%): {n_short} ({n_short/n_above_T*100:.1f}% of above-T)")
    print(f"  Long/short ratio: {n_long/max(n_short,1):.2f}")
    if abs(n_long - n_short) / n_total > 0.05:
        print(f"  ⚠️  IMBALANCE detected (>5% gap)")

    # ---- 4. Build Self all dates 4/21-30 + recent (5/2-5/9 collector) ----
    log.info("\n[B] Building Self 4/21-30 + recent...")
    SELF_DATES = [f"2026-04-{d:02d}" for d in range(21, 31)] + \
                 [f"2026-05-{d:02d}" for d in range(1, 10)]
    self_dfs = {}
    for d in SELF_DATES:
        try:
            df = build_self_date_dataset(d, log, train_medians=TRAIN_MEDIANS)
            df = df.dropna(subset=[vol_target, dir_target])
            if len(df) > 0:
                self_dfs[d] = df
        except Exception as e:
            log.warning(f"  Self {d}: {e}")
    log.info(f"  Self dates loaded: {len(self_dfs)}")

    # ---- 5. Apply norm features (proper day-mean) ----
    for d in self_dfs:
        self_dfs[d] = add_normalized_features(self_dfs[d], log)

    # ---- 6. Compute dir_proba distribution per period ----
    norm_features = [c for c in tardis_train.columns if c.endswith("_norm")]
    feat_set = [f for f in get_feature_columns(tardis_train) if f not in HIGH_SHIFT_FEATURES] + norm_features

    print()
    print("=" * 80)
    print("[B] DIR_PROBA DISTRIBUTION per period (mark36_v1 inference)")
    print("=" * 80)

    def predict_period(df, label):
        meds = TRAIN_MEDIANS.reindex(feat_set)
        X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan).fillna(meds).fillna(0)
        dir_proba = XGBD.predict_proba(X.values)[:, 1]
        return dir_proba

    # Backtest period (4/21-30): properly normalized day-mean
    bt_period_dfs = []
    for d in SELF_DATES:
        if d in self_dfs and d <= "2026-04-30":
            bt_period_dfs.append(self_dfs[d])
    bt_df = pd.concat(bt_period_dfs, ignore_index=True) if bt_period_dfs else pd.DataFrame()
    if len(bt_df) > 0:
        dir_proba_bt = predict_period(bt_df, "Backtest 4/21-30")
        print(f"\n  Backtest period 4/21-30 (n={len(bt_df)}):")
        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            print(f"    q{int(q*100):02d}: {np.quantile(dir_proba_bt, q):.4f}")
        print(f"    > 0.55: {(dir_proba_bt > 0.55).mean()*100:.1f}%  (LONG signals)")
        print(f"    < 0.45: {(dir_proba_bt < 0.45).mean()*100:.1f}%  (SHORT signals)")

    # SHADOW period (5/2-5/9): also day-mean normalized
    sh_period_dfs = []
    for d in SELF_DATES:
        if d in self_dfs and d >= "2026-05-02":
            sh_period_dfs.append(self_dfs[d])
    sh_df = pd.concat(sh_period_dfs, ignore_index=True) if sh_period_dfs else pd.DataFrame()
    if len(sh_df) > 0:
        dir_proba_sh = predict_period(sh_df, "SHADOW 5/2-5/9")
        print(f"\n  SHADOW period 5/2-5/9 (n={len(sh_df)}):")
        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            print(f"    q{int(q*100):02d}: {np.quantile(dir_proba_sh, q):.4f}")
        print(f"    > 0.55: {(dir_proba_sh > 0.55).mean()*100:.1f}%  (LONG signals)")
        print(f"    < 0.45: {(dir_proba_sh < 0.45).mean()*100:.1f}%  (SHORT signals)")

    # ---- 7. Compare distributions ----
    if len(bt_df) > 0 and len(sh_df) > 0:
        from scipy import stats as scistats
        ks, p = scistats.ks_2samp(dir_proba_bt, dir_proba_sh)
        print(f"\n  KS test (backtest vs shadow dir_proba): KS={ks:.3f}  p={p:.2e}")
        print(f"  Backtest mean: {dir_proba_bt.mean():.4f}  Shadow mean: {dir_proba_sh.mean():.4f}")
        print(f"  Backtest std:  {dir_proba_bt.std():.4f}  Shadow std:  {dir_proba_sh.std():.4f}")

    # ---- 8. ETH market regime 5/2-5/9 ----
    print()
    print("=" * 80)
    print("[C] ETH MARKET REGIME 5/2-5/9 (actual returns)")
    print("=" * 80)
    if len(sh_df) > 0:
        rs = sh_df[dir_target].values
        print(f"\n  rows: {len(sh_df)}")
        print(f"  mean return: {rs.mean():+.4f}%")
        print(f"  std return:  {rs.std():.4f}")
        print(f"  >0% rows: {(rs > 0).mean()*100:.1f}%")
        print(f"  >0.20% LONG samples: {(rs > 0.20).sum()} ({(rs > 0.20).mean()*100:.1f}%)")
        print(f"  <-0.20% SHORT samples: {(rs < -0.20).sum()} ({(rs < -0.20).mean()*100:.1f}%)")
        print(f"  trend (sign sum / abs sum): {(rs.sum() / max(np.abs(rs).sum(), 1)):.3f}")

    # ---- 9. Norm features lookahead bias check ----
    print()
    print("=" * 80)
    print("[D] NORM FEATURES LOOKAHEAD BIAS CHECK")
    print("=" * 80)
    print("\n  Backtest add_normalized_features():")
    print("  - groupby('_source_date').transform('mean')")
    print("  - 한 일자 전체 평균을 그 일자 모든 row 에 적용")
    print("  → Inference 시점 (intraday)에 그 day의 future row들 평균을 사용")
    print("  ⚠️  LOOKAHEAD BIAS 의심")

    print("\n  SHADOW direction live:")
    print("  - 매 1분 build_live_dataset → 누적된 session buffer 평균 사용")
    print("  - 진짜 inference-time (only past data)")
    print("  → 두 환경의 norm features 값 다름")

    # Demonstrate: compute "live-style" norm (cumulative) vs "backtest-style" norm (full-day)
    if len(self_dfs) > 0:
        sample_d = list(self_dfs.keys())[-3]  # third-to-last day
        df = self_dfs[sample_d].copy()
        # "Backtest" norm: already done (full-day)
        # "Live" norm: rolling cumulative mean
        for f in HIGH_SHIFT_FEATURES[:3]:
            if f not in df.columns: continue
            # cumulative mean (live-style)
            live_mean = df[f].expanding().mean().shift(1).bfill()
            backtest_mean = df[f].mean()
            df[f"{f}_live_norm"] = df[f] / live_mean.replace(0, np.nan)
            df[f"{f}_bt_norm"] = df[f] / backtest_mean
        print(f"\n  Sample day {sample_d}:")
        for f in HIGH_SHIFT_FEATURES[:3]:
            if f"{f}_live_norm" in df.columns:
                live_n = df[f"{f}_live_norm"].dropna()
                bt_n = df[f"{f}_bt_norm"].dropna()
                if len(live_n) > 0 and len(bt_n) > 0:
                    print(f"    {f}: live mean {live_n.mean():.4f} (std {live_n.std():.4f}) vs bt mean {bt_n.mean():.4f} (std {bt_n.std():.4f})")

    # ---- Save ----
    out = {
        "train_balance": {
            "n_total": int(n_total),
            "n_long_above_T": int(n_long),
            "n_short_above_T": int(n_short),
            "long_share": float(n_long / max(n_above_T, 1)),
            "short_share": float(n_short / max(n_above_T, 1)),
        }
    }
    if len(bt_df) > 0:
        out["backtest_dir_proba"] = {
            "mean": float(dir_proba_bt.mean()),
            "std": float(dir_proba_bt.std()),
            "above_055": float((dir_proba_bt > 0.55).mean()),
            "below_045": float((dir_proba_bt < 0.45).mean()),
        }
    if len(sh_df) > 0:
        out["shadow_dir_proba"] = {
            "mean": float(dir_proba_sh.mean()),
            "std": float(dir_proba_sh.std()),
            "above_055": float((dir_proba_sh > 0.55).mean()),
            "below_045": float((dir_proba_sh < 0.45).mean()),
        }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/diagnose_shadow_direction.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # ---- Diagnosis ----
    print()
    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    if len(bt_df) > 0:
        below_045_bt = (dir_proba_bt < 0.45).mean()
        if below_045_bt < 0.01:
            print(f"\n  [Case D probable] Backtest period에서도 < 0.45 비율 {below_045_bt*100:.2f}%")
            print(f"  → 모델 자체가 0.45 미만으로 거의 안 떨어짐 (bias)")

    if len(sh_df) > 0:
        below_045_sh = (dir_proba_sh < 0.45).mean()
        print(f"  SHADOW period에서 < 0.45 비율: {below_045_sh*100:.2f}%")

    if len(self_dfs) > 0:
        # Lookahead diagnostic
        print(f"\n  Norm features = day-mean (groupby _source_date)")
        print(f"  Backtest training: 학습/테스트 모두 day-mean 사용 → 학습 시 leakage 가능")
        print(f"  SHADOW live: cumulative session-mean → 다른 분포")
        print(f"  → Case B (lookahead bias) likely 강력 후보")

    log.info("\n진단 complete")


if __name__ == "__main__":
    main()
