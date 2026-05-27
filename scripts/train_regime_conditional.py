"""Regime-Conditional Direction Models (시도 8)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import (
    DATES_TRAIN, DATES_VAL, DATES_TEST,
    build_split, get_feature_columns,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("REGIME-CONDITIONAL DIRECTION MODEL (시도 8)")
    log.info("=" * 70)

    log.info("\nBuilding datasets...")
    train_df_full = build_split(DATES_TRAIN, log)
    val_df_full = build_split(DATES_VAL, log)
    test_df_full = build_split(DATES_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"

    for df in [train_df_full, val_df_full, test_df_full]:
        df.dropna(subset=[vol_target, dir_target], inplace=True)

    feature_cols = get_feature_columns(train_df_full)
    log.info(f"Features: {len(feature_cols)}")

    X_train_raw = train_df_full.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_raw.median(numeric_only=True)

    def make_X(df, feat_cols, train_medians):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(train_medians).fillna(0)
        return X_filled

    X_train = make_X(train_df_full, feature_cols, train_feature_medians)
    X_val = make_X(val_df_full, feature_cols, train_feature_medians)
    X_test = make_X(test_df_full, feature_cols, train_feature_medians)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    log.info("\n=== PHASE 1: Vol classifier ===")
    train_vol_median = train_df_full[vol_target].median()
    y_vol_train = (train_df_full[vol_target] > train_vol_median).astype(int).values

    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    X_val_vol_scaled = scaler_vol.transform(X_val)
    X_test_vol_scaled = scaler_vol.transform(X_test)

    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)

    vol_proba_train = lr_vol.predict_proba(X_train_vol_scaled)[:, 1]
    vol_proba_val = lr_vol.predict_proba(X_val_vol_scaled)[:, 1]
    vol_proba_test = lr_vol.predict_proba(X_test_vol_scaled)[:, 1]

    y_vol_test = (test_df_full[vol_target] > train_vol_median).astype(int).values
    log.info(f"  Vol AUC: {roc_auc_score(y_vol_test, vol_proba_test):.3f}")

    HIGH_VOL_THRESH = 0.65
    LOW_VOL_THRESH = 0.35

    train_df_full = train_df_full.copy()
    val_df_full = val_df_full.copy()
    test_df_full = test_df_full.copy()

    train_df_full["vol_proba"] = vol_proba_train
    val_df_full["vol_proba"] = vol_proba_val
    test_df_full["vol_proba"] = vol_proba_test

    def get_regime(vol_proba):
        if vol_proba > HIGH_VOL_THRESH:
            return "high"
        elif vol_proba < LOW_VOL_THRESH:
            return "low"
        else:
            return "normal"

    train_df_full["regime"] = train_df_full["vol_proba"].apply(get_regime)
    val_df_full["regime"] = val_df_full["vol_proba"].apply(get_regime)
    test_df_full["regime"] = test_df_full["vol_proba"].apply(get_regime)

    print()
    print("=" * 80)
    print("REGIME DISTRIBUTION")
    print("=" * 80)
    for regime in ["high", "normal", "low"]:
        train_n = (train_df_full["regime"] == regime).sum()
        val_n = (val_df_full["regime"] == regime).sum()
        test_n = (test_df_full["regime"] == regime).sum()
        print(f"  {regime:<8}: train {train_n} ({train_n/len(train_df_full)*100:.1f}%), val {val_n}, test {test_n}")

    print()
    print("=" * 80)
    print("PHASE 3: Direction models per regime")
    print("=" * 80)

    T = 0.20

    regime_models = {}
    regime_scalers = {}
    regime_aucs = {}

    for regime in ["high", "normal", "low"]:
        print(f"\n--- Training {regime} regime model ---")

        train_mask = (train_df_full["regime"] == regime) & (train_df_full[dir_target].abs() > T)
        val_mask = (val_df_full["regime"] == regime) & (val_df_full[dir_target].abs() > T)

        X_train_r = X_train[train_mask].values
        y_train_r = (train_df_full.loc[train_mask, dir_target] > 0).astype(int).values

        X_val_r = X_val[val_mask].values
        y_val_r = (val_df_full.loc[val_mask, dir_target] > 0).astype(int).values

        print(f"  Train sample: {len(X_train_r)}")
        print(f"  Val sample: {len(X_val_r)}")

        if len(X_train_r) < 200:
            print(f"  WARN: Too few - using fallback (full)")
            train_mask_fallback = train_df_full[dir_target].abs() > T
            X_train_r = X_train[train_mask_fallback].values
            y_train_r = (train_df_full.loc[train_mask_fallback, dir_target] > 0).astype(int).values

        if y_train_r.sum() < 50 or (len(y_train_r) - y_train_r.sum()) < 50:
            print(f"  WARN: Class imbalance - skip")
            continue

        scaler = StandardScaler()
        X_train_r_scaled = scaler.fit_transform(X_train_r)

        lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lr.fit(X_train_r_scaled, y_train_r)

        regime_models[regime] = lr
        regime_scalers[regime] = scaler

        if len(X_val_r) >= 50 and y_val_r.sum() >= 10 and (len(y_val_r) - y_val_r.sum()) >= 10:
            X_val_r_scaled = scaler.transform(X_val_r)
            val_proba = lr.predict_proba(X_val_r_scaled)[:, 1]
            val_auc = roc_auc_score(y_val_r, val_proba)
            print(f"  Val AUC ({regime}): {val_auc:.3f}")
            regime_aucs[regime] = val_auc
        else:
            print(f"  Val: insufficient samples")

    print()
    print("=" * 80)
    print("PHASE 4: Combined regime-conditional prediction (test)")
    print("=" * 80)

    dir_proba_test_regime = np.zeros(len(test_df_full))
    test_X = X_test.values

    for i, regime in enumerate(test_df_full["regime"].values):
        if regime in regime_models:
            X_one = test_X[i].reshape(1, -1)
            X_one_scaled = regime_scalers[regime].transform(X_one)
            dir_proba_test_regime[i] = regime_models[regime].predict_proba(X_one_scaled)[0, 1]
        else:
            dir_proba_test_regime[i] = 0.5

    print("\n--- Baseline (single model, 시도 6) ---")
    train_mask_full = train_df_full[dir_target].abs() > T
    X_train_full_dir = X_train[train_mask_full].values
    y_train_full_dir = (train_df_full.loc[train_mask_full, dir_target] > 0).astype(int).values

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full_dir)

    lr_full = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_full.fit(X_train_full_scaled, y_train_full_dir)

    X_test_full_scaled = scaler_full.transform(test_X)
    dir_proba_test_baseline = lr_full.predict_proba(X_test_full_scaled)[:, 1]

    test_dir_mask = test_df_full[dir_target].abs() > T
    y_test_subset = (test_df_full.loc[test_dir_mask, dir_target] > 0).astype(int).values

    baseline_auc = roc_auc_score(y_test_subset, dir_proba_test_baseline[test_dir_mask.values])
    regime_auc = roc_auc_score(y_test_subset, dir_proba_test_regime[test_dir_mask.values])

    print(f"\n  Baseline (single):       Test AUC {baseline_auc:.3f}")
    print(f"  Regime-conditional:      Test AUC {regime_auc:.3f}")
    print(f"  Improvement: {regime_auc - baseline_auc:+.3f}")

    print(f"\n  Per-regime test AUC:")
    for regime in ["high", "normal", "low"]:
        regime_mask = (test_df_full["regime"] == regime) & test_dir_mask
        if regime_mask.sum() > 20:
            y_r = (test_df_full.loc[regime_mask, dir_target] > 0).astype(int).values
            if len(set(y_r)) > 1:
                auc_r = roc_auc_score(y_r, dir_proba_test_regime[regime_mask.values])
                print(f"    {regime}: n={regime_mask.sum()}, AUC={auc_r:.3f}")

    print()
    print("=" * 80)
    print("PHASE 5: Trading simulation comparison")
    print("=" * 80)

    test_df = test_df_full.copy().reset_index(drop=True)
    test_df["dir_proba_baseline"] = dir_proba_test_baseline
    test_df["dir_proba_regime"] = dir_proba_test_regime
    test_df["actual_return"] = test_df[dir_target].values

    ts_col = None
    for c in ["_ts", "ts", "timestamp"]:
        if c in test_df.columns:
            ts_col = c
            break
    test_df = test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    DIR_THRESH = 0.65
    VOL_THRESH = 0.6

    FEE_SCENARIOS = {
        "Maker (-0.05%)": -0.05,
        "Mixed (0.03%)": 0.03,
        "Taker (0.11%)": 0.11,
    }

    def backtest(dir_proba_col, fee, label):
        date_pnls = []
        all_trades = []

        for date_str in DATES_TEST:
            date_df = test_df[test_df["_source_date"] == date_str].copy().sort_values(ts_col).reset_index(drop=True)
            if len(date_df) == 0:
                continue

            if len(date_df) > 2:
                ts_diffs = date_df[ts_col].diff().dropna()
                median_diff = ts_diffs.median()
                if hasattr(median_diff, 'total_seconds'):
                    interval_sec = median_diff.total_seconds()
                else:
                    interval_sec = median_diff / 1e9 if median_diff > 1e9 else median_diff
                lockout_rows = max(int(3600 / max(interval_sec, 1)), 1)
            else:
                lockout_rows = 60

            i = 0
            n = len(date_df)
            date_trades = []

            while i < n:
                row = date_df.iloc[i]
                vol_proba = row["vol_proba"]
                dir_proba = row[dir_proba_col]
                actual = row["actual_return"]

                if pd.isna(actual):
                    i += 1
                    continue

                trade = False
                direction = 0

                if vol_proba > VOL_THRESH:
                    if dir_proba > DIR_THRESH:
                        trade = True; direction = 1
                    elif dir_proba < (1 - DIR_THRESH):
                        trade = True; direction = -1

                if trade:
                    pnl = direction * actual - fee
                    date_trades.append({"pnl": pnl})
                    i += lockout_rows
                else:
                    i += 1

            if date_trades:
                pnl_sum = sum(t["pnl"] for t in date_trades)
                wins = sum(1 for t in date_trades if t["pnl"] > 0)
                date_pnls.append({
                    "date": date_str,
                    "n_trades": len(date_trades),
                    "pnl_sum": pnl_sum,
                    "win_rate": wins / len(date_trades),
                })
                all_trades.extend(date_trades)
            else:
                date_pnls.append({"date": date_str, "n_trades": 0, "pnl_sum": 0, "win_rate": 0})

        total_trades = sum(d["n_trades"] for d in date_pnls)
        daily_pnls = [d["pnl_sum"] for d in date_pnls]
        daily_avg = np.mean(daily_pnls) if daily_pnls else 0
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 0
        sharpe = daily_avg / max(daily_std, 0.001)

        cum_pnl = np.cumsum([d["pnl_sum"] for d in date_pnls])
        peak = np.maximum.accumulate(cum_pnl) if len(cum_pnl) > 0 else [0]
        drawdowns = peak - cum_pnl if len(cum_pnl) > 0 else [0]
        max_dd = max(drawdowns) if len(drawdowns) > 0 else 0

        wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
        win_rate = len(wins) / max(len(all_trades), 1)

        return {
            "label": label,
            "total_trades": total_trades,
            "daily_avg": daily_avg,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "date_pnls": date_pnls,
        }

    print(f"\n{'Approach':<30} {'Fee':<20} {'Trades':<8} {'Win%':<8} {'Daily':<11} {'Sharpe':<10} {'MaxDD':<10}")
    print("-" * 100)

    all_results = {}

    for fee_name, fee in FEE_SCENARIOS.items():
        baseline_r = backtest("dir_proba_baseline", fee, "baseline")
        regime_r = backtest("dir_proba_regime", fee, "regime")

        all_results[(fee_name, "baseline")] = baseline_r
        all_results[(fee_name, "regime")] = regime_r

        print(f"{'Baseline (single)':<30} {fee_name:<20} {baseline_r['total_trades']:<8} {baseline_r['win_rate']:<8.3f} {baseline_r['daily_avg']:<+11.3f}% {baseline_r['sharpe']:<10.2f} {baseline_r['max_dd']:<10.3f}%")
        print(f"{'Regime conditional':<30} {fee_name:<20} {regime_r['total_trades']:<8} {regime_r['win_rate']:<8.3f} {regime_r['daily_avg']:<+11.3f}% {regime_r['sharpe']:<10.2f} {regime_r['max_dd']:<10.3f}%")
        print()

    print()
    print("=" * 80)
    print("PER-DATE BREAKDOWN (Maker, regime-conditional)")
    print("=" * 80)
    for d in all_results[("Maker (-0.05%)", "regime")]["date_pnls"]:
        print(f"  {d['date']}: trades={d['n_trades']:<3} pnl={d['pnl_sum']:+.3f}% winrate={d['win_rate']:.3f}")

    print()
    print("=" * 80)
    print("결론")
    print("=" * 80)
    print()
    print("Direction AUC 변화:")
    print(f"  Baseline:   {baseline_auc:.3f}")
    print(f"  Regime:     {regime_auc:.3f}")
    print(f"  Δ:          {regime_auc - baseline_auc:+.3f}")
    print()
    print("일 1% 달성:")
    for fee_name in FEE_SCENARIOS:
        b = all_results[(fee_name, "baseline")]
        r = all_results[(fee_name, "regime")]
        print(f"  {fee_name}:")
        print(f"    Baseline: {b['daily_avg']:+.2f}% (Sh {b['sharpe']:.2f})")
        print(f"    Regime:   {r['daily_avg']:+.2f}% (Sh {r['sharpe']:.2f})")
        print(f"    Δ: {r['daily_avg'] - b['daily_avg']:+.2f}%p")
    print()
    print("Decision:")
    print("  - Regime daily > baseline + Sharpe > 1.0 -> 시도 8 success, 시도 14 next")
    print("  - Regime ~ baseline -> small effect, 시도 14 direct")
    print("  - Regime < baseline -> 시도 8 fail, 시도 9 (ensemble) or 14")

    log.info("Done")


if __name__ == "__main__":
    main()
