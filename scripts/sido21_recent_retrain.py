"""시도 21: Recent-only Retrain (Tardis 2024+ → fallback 2023+)."""
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import (
    DATES_TRAIN, DATES_VAL, DATES_TEST,
    build_split, get_feature_columns,
)

# Reuse 시도 20's self_data builder via importlib (scripts/ has no __init__.py)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_backtest_self_data",
    Path(__file__).resolve().parent / "backtest_self_data.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset


def is_recent(date_str, min_year=2024):
    return int(date_str.split("-")[0]) >= min_year


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("시도 21: Recent-only Retrain")
    log.info("=" * 70)
    np.random.seed(42)

    # ---- 1. Date filter (2024+, fallback 2023+ if too few) ----
    MIN_YEAR = 2024
    RECENT_TRAIN = [d for d in DATES_TRAIN if is_recent(d, MIN_YEAR)]
    RECENT_VAL = [d for d in DATES_VAL if is_recent(d, MIN_YEAR)]
    RECENT_TEST = [d for d in DATES_TEST if is_recent(d, MIN_YEAR)]

    if len(RECENT_TRAIN) < 10:
        log.warning(f"2024+ train={len(RECENT_TRAIN)} < 10, fallback to 2023+")
        MIN_YEAR = 2023
        RECENT_TRAIN = [d for d in DATES_TRAIN if is_recent(d, MIN_YEAR)]
        RECENT_VAL = [d for d in DATES_VAL if is_recent(d, MIN_YEAR)]
        RECENT_TEST = [d for d in DATES_TEST if is_recent(d, MIN_YEAR)]

    log.info(f"\nMin year: {MIN_YEAR}")
    log.info(f"  Train: {len(RECENT_TRAIN)} - {RECENT_TRAIN}")
    log.info(f"  Val:   {len(RECENT_VAL)} - {RECENT_VAL}")
    log.info(f"  Test:  {len(RECENT_TEST)} - {RECENT_TEST}")

    # ---- 2. Build datasets ----
    log.info("\nBuilding Tardis recent datasets...")
    train_df = build_split(RECENT_TRAIN, log)
    val_df = build_split(RECENT_VAL, log)
    test_df = build_split(RECENT_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=[vol_target, dir_target], inplace=True)

    feature_cols = get_feature_columns(train_df)
    log.info(f"Features: {len(feature_cols)}")

    X_train_raw = train_df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_medians = X_train_raw.median(numeric_only=True)

    def make_X(df):
        X = df.reindex(columns=feature_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(train_medians).fillna(0)

    X_train = make_X(train_df)
    X_val = make_X(val_df)
    X_test = make_X(test_df)

    # ---- 3. Train ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    train_vol_median = float(train_df[vol_target].median())
    y_vol_train = (train_df[vol_target] > train_vol_median).astype(int).values
    y_vol_val = (val_df[vol_target] > train_vol_median).astype(int).values
    y_vol_test = (test_df[vol_target] > train_vol_median).astype(int).values

    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    X_val_vol_scaled = scaler_vol.transform(X_val)
    X_test_vol_scaled = scaler_vol.transform(X_test)

    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)

    vol_auc_train = roc_auc_score(y_vol_train, lr_vol.predict_proba(X_train_vol_scaled)[:, 1])
    vol_auc_val = roc_auc_score(y_vol_val, lr_vol.predict_proba(X_val_vol_scaled)[:, 1])
    vol_auc_test = roc_auc_score(y_vol_test, lr_vol.predict_proba(X_test_vol_scaled)[:, 1])

    log.info(f"\nVol  AUC: train {vol_auc_train:.3f} | val {vol_auc_val:.3f} | test {vol_auc_test:.3f}")

    T = 0.20
    train_dir_mask = train_df[dir_target].abs() > T
    val_dir_mask = val_df[dir_target].abs() > T
    test_dir_mask = test_df[dir_target].abs() > T

    X_train_dir = X_train[train_dir_mask].values
    y_dir_train = (train_df.loc[train_dir_mask, dir_target] > 0).astype(int).values
    X_val_dir = X_val[val_dir_mask].values
    y_dir_val = (val_df.loc[val_dir_mask, dir_target] > 0).astype(int).values
    X_test_dir = X_test[test_dir_mask].values
    y_dir_test = (test_df.loc[test_dir_mask, dir_target] > 0).astype(int).values

    scaler_dir = StandardScaler()
    X_train_dir_scaled = scaler_dir.fit_transform(X_train_dir)
    X_val_dir_scaled = scaler_dir.transform(X_val_dir)
    X_test_dir_scaled = scaler_dir.transform(X_test_dir)

    lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_dir.fit(X_train_dir_scaled, y_dir_train)

    dir_auc_train = roc_auc_score(y_dir_train, lr_dir.predict_proba(X_train_dir_scaled)[:, 1])
    dir_auc_val = roc_auc_score(y_dir_val, lr_dir.predict_proba(X_val_dir_scaled)[:, 1])
    dir_auc_test = roc_auc_score(y_dir_test, lr_dir.predict_proba(X_test_dir_scaled)[:, 1])

    log.info(f"Dir  AUC: train {dir_auc_train:.3f} | val {dir_auc_val:.3f} | test {dir_auc_test:.3f}")

    # ---- 4. Self 2026 OOS ----
    log.info("\nBuilding self 2026 data...")
    SELF_DATES = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = []
    for date_str in SELF_DATES:
        try:
            df = build_self_date_dataset(date_str, log, train_medians=train_medians)
            if len(df) > 0:
                self_dfs.append(df)
        except Exception as e:
            log.error(f"  {date_str}: {e}")
    self_df = pd.concat(self_dfs, ignore_index=True)
    self_df.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"Self total: {len(self_df)} rows")

    X_self = make_X(self_df)
    self_vol_proba = lr_vol.predict_proba(scaler_vol.transform(X_self))[:, 1]
    self_dir_proba = lr_dir.predict_proba(scaler_dir.transform(X_self.values))[:, 1]

    y_vol_self = (self_df[vol_target] > train_vol_median).astype(int).values
    vol_auc_self = roc_auc_score(y_vol_self, self_vol_proba)

    self_dir_mask = self_df[dir_target].abs() > T
    if self_dir_mask.sum() > 10:
        y_dir_self = (self_df.loc[self_dir_mask, dir_target] > 0).astype(int).values
        dir_auc_self = roc_auc_score(y_dir_self, self_dir_proba[self_dir_mask.values])
    else:
        dir_auc_self = float("nan")

    log.info(f"\nSelf 2026: Vol {vol_auc_self:.3f} | Dir {dir_auc_self:.3f}")

    # ---- 5. Drift backtest on self ----
    self_df["vol_proba"] = self_vol_proba
    self_df["dir_proba"] = self_dir_proba
    self_df["actual_return"] = self_df[dir_target].values

    DIR_THRESH, VOL_THRESH = 0.65, 0.6
    LOCKOUT_ROWS = 60
    SL_PCT = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025
    MAX_HOLD = 30

    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in self_df.columns), None)
    price_col = next((c for c in ["ob_mid_price", "mid"] if c in self_df.columns), None)
    self_df = self_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    def drift_fill(d_df, idx, direction):
        if idx >= len(d_df): return False, 0
        entry = d_df.iloc[idx][price_col]
        if pd.isna(entry): return False, 0
        limit = entry * (0.99995 if direction == 1 else 1.00005)
        for t in range(1, MAX_HOLD + 1):
            if idx + t >= len(d_df): return False, t
            intra = d_df.iloc[idx + t][price_col]
            if pd.isna(intra): continue
            if direction == 1 and intra <= limit: return True, t
            if direction == -1 and intra >= limit: return True, t
            limit = intra * (0.99995 if direction == 1 else 1.00005)
        return False, MAX_HOLD

    daily_results = []
    for date_str in SELF_DATES:
        d_df = self_df[self_df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
        if len(d_df) < 100:
            daily_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                  "maker_rate": 0, "win_rate": 0})
            continue
        trades = []
        i, n = 0, len(d_df)
        while i < n:
            row = d_df.iloc[i]
            if pd.isna(row["actual_return"]) or pd.isna(row[price_col]):
                i += 1; continue
            direction = 0; trade = False
            if row["vol_proba"] > VOL_THRESH:
                if row["dir_proba"] > DIR_THRESH: direction = 1; trade = True
                elif row["dir_proba"] < (1 - DIR_THRESH): direction = -1; trade = True
            if trade:
                entry = row[price_col]
                ar = direction * row["actual_return"]
                sl_hit = False
                for t in range(1, LOCKOUT_ROWS + 1):
                    if i + t >= n: break
                    intra = d_df.iloc[i + t][price_col]
                    if pd.isna(intra): continue
                    pnl = direction * (intra - entry) / entry * 100
                    if pnl <= -SL_PCT:
                        ar = -SL_PCT; sl_hit = True; break
                if sl_hit:
                    fee_exit = FEE_TAKER; ft = "sl"
                else:
                    filled, _ = drift_fill(d_df, i + LOCKOUT_ROWS, -direction)
                    fee_exit = FEE_MAKER if filled else FEE_TAKER
                    ft = "maker" if filled else "taker"
                net_pnl = ar - (FEE_TAKER + fee_exit)
                trades.append({"net_pnl": net_pnl, "fill_type": ft, "sl": sl_hit})
                i += LOCKOUT_ROWS
            else:
                i += 1
        if trades:
            pnl = sum(t["net_pnl"] for t in trades)
            mr = sum(1 for t in trades if t["fill_type"] == "maker") / len(trades)
            wr = sum(1 for t in trades if t["net_pnl"] > 0) / len(trades)
            daily_results.append({"date": date_str, "n_trades": len(trades),
                                  "pnl_sum": pnl, "maker_rate": mr, "win_rate": wr})
        else:
            daily_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                  "maker_rate": 0, "win_rate": 0})

    daily_pnls = [d["pnl_sum"] for d in daily_results]
    daily_avg = float(np.mean(daily_pnls))
    daily_std = float(np.std(daily_pnls))
    sharpe = daily_avg / max(daily_std, 0.001)
    avg_maker = float(np.mean([d["maker_rate"] for d in daily_results if d["n_trades"] > 0] or [0]))
    avg_win = float(np.mean([d["win_rate"] for d in daily_results if d["n_trades"] > 0] or [0]))

    # ---- 6. Reports ----
    print()
    print("=" * 80)
    print("시도 17 vs 시도 21 (Self 2026 OOS)")
    print("=" * 80)
    print(f"\n{'Metric':<28} {'시도 17':<12} {'시도 21':<12} {'Δ':<10}")
    print("-" * 70)
    print(f"{'Train dates':<28} {'26':<12} {len(RECENT_TRAIN):<12} {len(RECENT_TRAIN)-26:<+10}")
    print(f"{'Vol AUC (Tardis test)':<28} {'~0.762':<12} {vol_auc_test:<12.3f} {vol_auc_test-0.762:<+10.3f}")
    print(f"{'Dir AUC (Tardis test)':<28} {'~0.545':<12} {dir_auc_test:<12.3f} {dir_auc_test-0.545:<+10.3f}")
    print(f"{'Vol AUC (Self 2026)':<28} {'0.793':<12} {vol_auc_self:<12.3f} {vol_auc_self-0.793:<+10.3f}")
    print(f"{'Dir AUC (Self 2026)':<28} {'0.480':<12} {dir_auc_self:<12.3f} {dir_auc_self-0.480:<+10.3f}")
    print(f"{'Self daily avg':<28} {'-0.134%':<12} {f'{daily_avg:+.3f}%':<12} {daily_avg+0.134:<+10.3f}p")
    print(f"{'Self Sharpe':<28} {'-0.08':<12} {sharpe:<12.2f} {sharpe+0.08:<+10.2f}")
    print(f"{'Self maker rate':<28} {'92.1%':<12} {f'{avg_maker*100:.1f}%':<12} {(avg_maker-0.921)*100:<+10.1f}p")

    print()
    print("Per-date (Self 2026):")
    print(f"{'Date':<14} {'Trades':<8} {'PnL':<12} {'Maker%':<8} {'Win%':<8}")
    print("-" * 60)
    for d in daily_results:
        print(f"{d['date']:<14} {d['n_trades']:<8} {d['pnl_sum']:<+12.3f} {d['maker_rate']*100:<8.1f} {d['win_rate']*100:<8.1f}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if not np.isnan(dir_auc_self):
        if dir_auc_self >= 0.55:
            print(f"\nGOOD Direction AUC 회복 (0.480 -> {dir_auc_self:.3f}). LIVE 적용 검토.")
        elif dir_auc_self >= 0.50:
            print(f"\nPARTIAL Dir AUC ({dir_auc_self:.3f}) — 시도 22 (Hybrid) 권장.")
        else:
            print(f"\nFAIL Dir AUC ({dir_auc_self:.3f}) < 0.50. 다른 접근 필요.")

    # ---- 7. Save ----
    out = {
        "lr_vol": lr_vol, "scaler_vol": scaler_vol,
        "lr_dir": lr_dir, "scaler_dir": scaler_dir,
        "feature_cols": feature_cols,
        "train_medians": train_medians.to_dict(),
        "train_vol_median": train_vol_median,
        "T": T,
        "metadata": {
            "min_year": MIN_YEAR,
            "train_dates": RECENT_TRAIN,
            "val_dates": RECENT_VAL,
            "test_dates": RECENT_TEST,
            "vol_auc_test": float(vol_auc_test),
            "dir_auc_test": float(dir_auc_test),
            "vol_auc_self": float(vol_auc_self),
            "dir_auc_self": float(dir_auc_self) if not np.isnan(dir_auc_self) else None,
            "self_daily_avg": daily_avg,
            "self_sharpe": sharpe,
            "self_maker_rate": avg_maker,
            "self_win_rate": avg_win,
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark21_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"Model: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido21_recent_retrain.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"], "daily_results": daily_results}, f, indent=2, default=str)
    log.info(f"JSON:  {json_path}")
    log.info("시도 21 complete")


if __name__ == "__main__":
    main()
