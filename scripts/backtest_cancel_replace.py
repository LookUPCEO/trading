"""Phase 3: Cancel/Replace Policy Backtest - Maker fill rate optimization."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
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
    log.info("PHASE 3: CANCEL/REPLACE POLICY OPTIMIZATION")
    log.info("=" * 70)

    np.random.seed(42)

    log.info("\nBuilding datasets...")
    train_df = build_split(DATES_TRAIN, log)
    val_df = build_split(DATES_VAL, log)
    test_df = build_split(DATES_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=[vol_target, dir_target], inplace=True)

    feature_cols = get_feature_columns(train_df)
    X_train_raw = train_df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_medians = X_train_raw.median(numeric_only=True)

    def make_X(df):
        X = df.reindex(columns=feature_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(train_medians).fillna(0)

    X_train = make_X(train_df); X_test = make_X(test_df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_vol_median = train_df[vol_target].median()
    y_vol_train = (train_df[vol_target] > train_vol_median).astype(int).values
    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    X_test_vol_scaled = scaler_vol.transform(X_test)
    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)
    vol_proba_test = lr_vol.predict_proba(X_test_vol_scaled)[:, 1]

    T = 0.20
    train_dir_mask = train_df[dir_target].abs() > T
    X_train_dir = X_train[train_dir_mask].values
    y_dir_train = (train_df.loc[train_dir_mask, dir_target] > 0).astype(int).values
    scaler_dir = StandardScaler()
    X_train_dir_scaled = scaler_dir.fit_transform(X_train_dir)
    X_test_dir_scaled = scaler_dir.transform(X_test.values)
    lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_dir.fit(X_train_dir_scaled, y_dir_train)
    dir_proba_test = lr_dir.predict_proba(X_test_dir_scaled)[:, 1]

    log.info("Models trained")

    test_df = test_df.copy().reset_index(drop=True)
    test_df["vol_proba"] = vol_proba_test
    test_df["dir_proba"] = dir_proba_test
    test_df["actual_return"] = test_df[dir_target].values

    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in test_df.columns), None)
    price_col = next((c for c in ["ob_mid_price", "mid", "mid_price", "close"] if c in test_df.columns), None)
    test_df = test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)
    log.info(f"Price col: {price_col}")

    DIR_THRESH, VOL_THRESH = 0.65, 0.6
    LOCKOUT_ROWS = 60
    SL_PCT = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025

    POLICIES = [
        ("Static 1min",        "static",  1,   None),
        ("Static 3min",        "static",  3,   None),
        ("Static 5min (LIVE)", "static",  5,   None),
        ("Static 10min",       "static",  10,  None),
        ("Static 30min",       "static",  30,  None),
        ("Drift 0.05%",        "drift",   30,  0.05),
        ("Drift 0.10%",        "drift",   30,  0.10),
        ("Follow (1min)",      "follow",  30,  None),
    ]

    def simulate_limit_fill(date_df, idx, direction, max_hold_min, policy_type, drift_threshold):
        n = len(date_df)
        if idx >= n:
            return False, 0, None, "skip"
        entry_mid = date_df.iloc[idx][price_col]
        if pd.isna(entry_mid):
            return False, 0, None, "skip"
        if direction == 1:
            limit_price = entry_mid * 0.99995
        else:
            limit_price = entry_mid * 1.00005

        for t in range(1, max_hold_min + 1):
            if idx + t >= n:
                return False, t, None, "expired"
            intra_price = date_df.iloc[idx + t][price_col]
            if pd.isna(intra_price):
                continue
            if direction == 1 and intra_price <= limit_price:
                return True, t, limit_price, "maker"
            if direction == -1 and intra_price >= limit_price:
                return True, t, limit_price, "maker"
            if policy_type == "drift" and drift_threshold is not None:
                drift_pct = abs(intra_price - limit_price) / limit_price * 100
                if drift_pct > drift_threshold:
                    if direction == 1:
                        limit_price = intra_price * 0.99995
                    else:
                        limit_price = intra_price * 1.00005
            if policy_type == "follow":
                if direction == 1:
                    limit_price = intra_price * 0.99995
                else:
                    limit_price = intra_price * 1.00005
        return False, max_hold_min, None, "taker_fallback"

    def run_policy(policy_name, policy_type, max_hold_min, drift_threshold, n_runs=3):
        all_runs = []
        for run_id in range(n_runs):
            np.random.seed(42 + run_id)
            date_results = []
            for date_str in DATES_TEST:
                date_df = test_df[test_df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
                if len(date_df) < 100:
                    date_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                         "maker_rate": 0, "sl_rate": 0, "win_rate": 0})
                    continue
                trades = []
                i, n = 0, len(date_df)
                while i < n:
                    row = date_df.iloc[i]
                    if pd.isna(row["actual_return"]) or pd.isna(row[price_col]):
                        i += 1; continue
                    vol_proba, dir_proba = row["vol_proba"], row["dir_proba"]
                    direction = 0; trade = False
                    if vol_proba > VOL_THRESH:
                        if dir_proba > DIR_THRESH:
                            direction = 1; trade = True
                        elif dir_proba < (1 - DIR_THRESH):
                            direction = -1; trade = True
                    if trade:
                        entry_price = row[price_col]
                        fee_entry = FEE_TAKER
                        # SL check intra-bar over 60min hold
                        # Default exit: signed return after applying direction
                        sl_hit = False
                        actual_return = direction * row["actual_return"]
                        for t in range(1, LOCKOUT_ROWS + 1):
                            if i + t >= n:
                                break
                            intra = date_df.iloc[i + t][price_col]
                            if pd.isna(intra):
                                continue
                            pnl_pct = direction * (intra - entry_price) / entry_price * 100
                            if pnl_pct <= -SL_PCT:
                                actual_return = -SL_PCT
                                sl_hit = True
                                break
                        # Exit policy
                        if sl_hit:
                            fee_exit = FEE_TAKER
                            fill_type = "sl_taker"
                        else:
                            exit_idx = i + LOCKOUT_ROWS
                            filled, _, _, fill_type = simulate_limit_fill(
                                date_df, exit_idx, -direction,
                                max_hold_min, policy_type, drift_threshold,
                            )
                            fee_exit = FEE_MAKER if filled else FEE_TAKER
                            if not filled:
                                fill_type = "taker_fallback"
                        total_fee = fee_entry + fee_exit
                        net_pnl = actual_return - total_fee
                        trades.append({"net_pnl": net_pnl, "fill_type": fill_type, "sl_hit": sl_hit})
                        i += LOCKOUT_ROWS
                    else:
                        i += 1
                if trades:
                    pnl_sum = sum(t["net_pnl"] for t in trades)
                    maker_count = sum(1 for t in trades if t["fill_type"] == "maker")
                    sl_count = sum(1 for t in trades if t["sl_hit"])
                    wins = sum(1 for t in trades if t["net_pnl"] > 0)
                    date_results.append({
                        "date": date_str, "n_trades": len(trades), "pnl_sum": pnl_sum,
                        "maker_rate": maker_count / len(trades),
                        "sl_rate": sl_count / len(trades),
                        "win_rate": wins / len(trades),
                    })
                else:
                    date_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                         "maker_rate": 0, "sl_rate": 0, "win_rate": 0})
            all_runs.append(date_results)

        # Average across runs
        avg_pnls, avg_makers, avg_sls, avg_trades = [], [], [], []
        for date_idx in range(len(DATES_TEST)):
            day_data = [run[date_idx] for run in all_runs if date_idx < len(run)]
            if not day_data:
                continue
            avg_pnls.append(float(np.mean([d["pnl_sum"] for d in day_data])))
            avg_makers.append(float(np.mean([d["maker_rate"] for d in day_data])))
            avg_sls.append(float(np.mean([d["sl_rate"] for d in day_data])))
            avg_trades.append(float(np.mean([d["n_trades"] for d in day_data])))

        daily_avg = float(np.mean(avg_pnls)) if avg_pnls else 0.0
        daily_std = float(np.std(avg_pnls)) if len(avg_pnls) > 1 else 0.0
        sharpe = daily_avg / max(daily_std, 0.001)
        cum = np.cumsum(avg_pnls) if avg_pnls else np.array([0])
        peak = np.maximum.accumulate(cum)
        max_dd = float(np.max(peak - cum)) if len(cum) else 0.0

        return {
            "daily_avg": daily_avg, "daily_std": daily_std,
            "sharpe": sharpe, "max_dd": max_dd,
            "maker_rate": float(np.mean(avg_makers)) if avg_makers else 0.0,
            "sl_rate": float(np.mean(avg_sls)) if avg_sls else 0.0,
            "total_trades": float(sum(avg_trades)),
        }

    print()
    print("=" * 110)
    print("PHASE 3 RESULTS (8 policies, SL 1.5% LIVE config)")
    print("=" * 110)
    print(f"{'Policy':<28} {'Daily':<10} {'Std':<8} {'Sharpe':<8} {'MaxDD':<8} {'Maker%':<8} {'SL%':<6} {'Trades':<8}")
    print("-" * 110)

    results = {}
    for policy_name, policy_type, max_hold, drift_thresh in POLICIES:
        r = run_policy(policy_name, policy_type, max_hold, drift_thresh, n_runs=3)
        results[policy_name] = r
        print(f"{policy_name:<28} {r['daily_avg']:<+10.3f} {r['daily_std']:<8.3f} {r['sharpe']:<8.2f} {r['max_dd']:<8.3f} {r['maker_rate']*100:<8.1f} {r['sl_rate']*100:<6.1f} {r['total_trades']:<8.1f}")

    print()
    print("=" * 110)
    print("TOP 3 BY SHARPE")
    print("=" * 110)
    sorted_r = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print(f"{'Rank':<5} {'Policy':<28} {'Daily':<10} {'Sharpe':<8} {'Maker%':<8}")
    print("-" * 70)
    for rank, (name, r) in enumerate(sorted_r[:3], 1):
        print(f"{rank:<5} {name:<28} {r['daily_avg']:<+10.3f} {r['sharpe']:<8.2f} {r['maker_rate']*100:<8.1f}")

    print()
    print("=" * 110)
    print("BASELINE 비교 (Static 5min = LIVE)")
    print("=" * 110)
    baseline = results.get("Static 5min (LIVE)")
    if baseline:
        print(f"\nBaseline daily: {baseline['daily_avg']:+.3f}%, maker {baseline['maker_rate']*100:.1f}%")
        print()
        print(f"{'Policy':<28} {'Δ Daily':<12} {'Δ Maker':<12}")
        print("-" * 60)
        for name, r in sorted_r:
            if name == "Static 5min (LIVE)":
                continue
            d_daily = r["daily_avg"] - baseline["daily_avg"]
            d_maker = (r["maker_rate"] - baseline["maker_rate"]) * 100
            print(f"{name:<28} {d_daily:<+12.3f} {d_maker:<+12.1f}p")

    out = {name: {k: float(v) for k, v in r.items()} for name, r in results.items()}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/phase3_cancel_replace_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved: {out_path}")
    log.info("Phase 3 complete")


if __name__ == "__main__":
    main()
