"""시도 14 Phase 2: Queue Simulation Backtest with Stochastic Maker Fill + SL."""
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
    log.info("PHASE 2: QUEUE SIMULATION BACKTEST")
    log.info("Stochastic Maker Fill + SL 통합")
    log.info("=" * 70)

    np.random.seed(42)

    # ---- 1. Datasets + model (시도 17 reproduction) ----
    log.info("\nBuilding datasets...")
    train_df = build_split(DATES_TRAIN, log)
    val_df = build_split(DATES_VAL, log)
    test_df = build_split(DATES_TEST, log)

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
    X_test = make_X(test_df)

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

    log.info("Models trained (시도 17)")

    # ---- 2. Test data prep ----
    test_df = test_df.copy().reset_index(drop=True)
    test_df["vol_proba"] = vol_proba_test
    test_df["dir_proba"] = dir_proba_test
    test_df["actual_return"] = test_df[dir_target].values

    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in test_df.columns), None)
    test_df = test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    price_col = next((c for c in ["ob_mid_price", "mid", "mid_price", "close", "price", "ask_0_price"]
                      if c in test_df.columns), None)
    if price_col is None:
        log.error("Price column not found")
        sys.exit(1)
    log.info(f"Price column: {price_col}")

    # ---- 3. Strategy params ----
    DIR_THRESH = 0.65
    VOL_THRESH = 0.6
    LOCKOUT_ROWS = 60          # 1h lockout (1-min grid → 60 rows)
    FEE_TAKER = 0.055           # %
    FEE_MAKER = -0.025          # % (rebate)

    MAKER_FILL_SCENARIOS = [
        ("p=0.30 (보수)",            0.30),
        ("p=0.38 (LIVE 검증)",       0.38),
        ("p=0.50",                  0.50),
        ("p=0.70 (개선 목표)",       0.70),
        ("p=1.00 (이상, 시도 17)", 1.00),
    ]

    SL_CONFIGS = [
        ("No SL (시도 17 baseline)",          None, None),
        ("SL 1.5% (LIVE)",                   1.5,  None),
        ("SL 0.5% / TP 1.0% (시도 16 변형)", 0.5,  1.0),
    ]

    # ---- 4. Per-trade simulation ----
    def simulate_intra_bar(date_df, idx, direction, entry_price, sl_pct, tp_pct):
        """Intra-bar SL/TP check on minute grid. Returns (actual_return, exit_min, sl_hit, tp_hit)."""
        n = len(date_df)
        for t in range(1, LOCKOUT_ROWS + 1):
            j = idx + t
            if j >= n:
                break
            intra_price = date_df.iloc[j][price_col]
            if pd.isna(intra_price):
                continue
            intra_pnl_pct = direction * (intra_price - entry_price) / entry_price * 100
            if sl_pct is not None and intra_pnl_pct <= -sl_pct:
                return -sl_pct, t, True, False
            if tp_pct is not None and intra_pnl_pct >= tp_pct:
                return tp_pct, t, False, True

        # Normal 60-min exit: use the model's target_return_3600s at entry row
        # (interpreted as direction × actual_return where actual_return is signed)
        ar = date_df.iloc[idx]["actual_return"]
        if pd.isna(ar):
            return 0.0, LOCKOUT_ROWS, False, False
        return float(direction * ar), LOCKOUT_ROWS, False, False

    def run_simulation(maker_fill_p, sl_pct, tp_pct, n_runs=5):
        all_run_results = []
        for run_id in range(n_runs):
            np.random.seed(42 + run_id)
            date_results = []
            for date_str in DATES_TEST:
                date_df = test_df[test_df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
                if len(date_df) < 100:
                    date_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                         "win_rate": 0, "maker_rate": 0, "sl_rate": 0})
                    continue

                trades = []
                i = 0
                n = len(date_df)
                while i < n:
                    row = date_df.iloc[i]
                    vol_proba = row["vol_proba"]
                    dir_proba = row["dir_proba"]
                    if pd.isna(row["actual_return"]) or pd.isna(row[price_col]):
                        i += 1
                        continue

                    direction = 0
                    trade = False
                    if vol_proba > VOL_THRESH:
                        if dir_proba > DIR_THRESH:
                            direction = 1; trade = True
                        elif dir_proba < (1 - DIR_THRESH):
                            direction = -1; trade = True

                    if trade:
                        entry_price = row[price_col]
                        actual_return, exit_min, sl_hit, tp_hit = simulate_intra_bar(
                            date_df, i, direction, entry_price, sl_pct, tp_pct,
                        )
                        # Entry: always taker (matches LIVE)
                        fee_entry = FEE_TAKER
                        # Exit: SL/TP triggers force taker; otherwise stochastic maker
                        if sl_hit or tp_hit:
                            fee_exit = FEE_TAKER
                            fill_type = "sl_taker" if sl_hit else "tp_taker"
                        else:
                            maker_success = bool(np.random.binomial(1, maker_fill_p))
                            fee_exit = FEE_MAKER if maker_success else FEE_TAKER
                            fill_type = "maker" if maker_success else "taker_fallback"
                        total_fee = fee_entry + fee_exit
                        net_pnl = actual_return - total_fee
                        trades.append({
                            "net_pnl": net_pnl, "fill_type": fill_type,
                            "sl_hit": sl_hit, "tp_hit": tp_hit, "exit_min": exit_min,
                        })
                        # Lockout: 60min OR shorter if SL/TP triggered
                        i += min(exit_min if (sl_hit or tp_hit) else LOCKOUT_ROWS, LOCKOUT_ROWS)
                    else:
                        i += 1

                if trades:
                    pnl_sum = sum(t["net_pnl"] for t in trades)
                    wins = sum(1 for t in trades if t["net_pnl"] > 0)
                    maker_count = sum(1 for t in trades if t["fill_type"] == "maker")
                    sl_count = sum(1 for t in trades if t["sl_hit"])
                    date_results.append({
                        "date": date_str,
                        "n_trades": len(trades),
                        "pnl_sum": pnl_sum,
                        "win_rate": wins / max(len(trades), 1),
                        "maker_rate": maker_count / max(len(trades), 1),
                        "sl_rate": sl_count / max(len(trades), 1),
                    })
                else:
                    date_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                         "win_rate": 0, "maker_rate": 0, "sl_rate": 0})
            all_run_results.append(date_results)

        # Average across runs
        avg = []
        for date_idx, date_str in enumerate(DATES_TEST):
            day_data = [run[date_idx] for run in all_run_results if date_idx < len(run)]
            if not day_data:
                continue
            avg.append({
                "date": date_str,
                "n_trades": float(np.mean([d["n_trades"] for d in day_data])),
                "pnl_sum": float(np.mean([d["pnl_sum"] for d in day_data])),
                "win_rate": float(np.mean([d["win_rate"] for d in day_data])),
                "maker_rate": float(np.mean([d["maker_rate"] for d in day_data])),
                "sl_rate": float(np.mean([d["sl_rate"] for d in day_data])),
            })

        daily_pnls = [d["pnl_sum"] for d in avg]
        daily_avg = float(np.mean(daily_pnls)) if daily_pnls else 0.0
        daily_std = float(np.std(daily_pnls)) if len(daily_pnls) > 1 else 0.0
        sharpe = daily_avg / max(daily_std, 0.001)
        cum = np.cumsum(daily_pnls) if daily_pnls else np.array([0])
        peak = np.maximum.accumulate(cum) if len(cum) else np.array([0])
        max_dd = float(np.max(peak - cum)) if len(cum) else 0.0

        return {
            "daily_pnls": avg, "daily_avg": daily_avg, "daily_std": daily_std,
            "sharpe": sharpe, "max_dd": max_dd,
            "total_trades": float(sum(d["n_trades"] for d in avg)),
            "avg_maker_rate": float(np.mean([d["maker_rate"] for d in avg if d["n_trades"] > 0]) if any(d["n_trades"] > 0 for d in avg) else 0),
            "avg_sl_rate": float(np.mean([d["sl_rate"] for d in avg if d["n_trades"] > 0]) if any(d["n_trades"] > 0 for d in avg) else 0),
        }

    # ---- 5. Run all combinations ----
    print()
    print("=" * 100)
    print("PHASE 2 RESULTS  (5 maker scenarios × 3 SL configs = 15 combos × 5 runs)")
    print("=" * 100)

    results = {}
    for sl_name, sl_pct, tp_pct in SL_CONFIGS:
        print(f"\n{'-'*100}")
        print(f"SL: {sl_name}")
        print(f"{'-'*100}")
        print(f"{'Maker fill':<26} {'Daily':<10} {'Std':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'Mkr%':<6} {'SL%':<6}")
        print("-" * 100)
        for maker_name, maker_p in MAKER_FILL_SCENARIOS:
            r = run_simulation(maker_p, sl_pct, tp_pct, n_runs=5)
            results[(sl_name, maker_name)] = r
            achieve = " ⭐" if r["daily_avg"] >= 1.0 else ""
            print(f"{maker_name:<26} {r['daily_avg']:<+10.3f} {r['daily_std']:<10.3f} {r['sharpe']:<8.2f} {r['max_dd']:<8.3f} {r['total_trades']:<8.1f} {r['avg_maker_rate']*100:<5.1f} {r['avg_sl_rate']*100:<5.1f}{achieve}")

    # ---- 6. Reference comparisons ----
    print()
    print("=" * 100)
    print("REFERENCES")
    print("=" * 100)
    bk = ("No SL (시도 17 baseline)", "p=1.00 (이상, 시도 17)")
    if bk in results:
        b = results[bk]
        print(f"\n시도 17 reproduction (No SL + p=1.00):")
        print(f"  Daily {b['daily_avg']:+.3f}%  Sharpe {b['sharpe']:.2f}  (BASECAMP +2.73% / 1.53)")
    lv = ("SL 1.5% (LIVE)", "p=0.38 (LIVE 검증)")
    if lv in results:
        l = results[lv]
        print(f"\nLIVE replica (SL 1.5% + p=0.38):")
        print(f"  Daily {l['daily_avg']:+.3f}%  Sharpe {l['sharpe']:.2f}  (LIVE 4-day actual: -0.09%/day)")

    # ---- 7. Top configs by Sharpe ----
    print()
    print("=" * 100)
    print("TOP 5 BY SHARPE")
    print("=" * 100)
    print(f"{'Rank':<5} {'SL config':<35} {'Maker fill':<26} {'Daily':<10} {'Sharpe':<8}")
    print("-" * 100)
    sorted_r = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for rank, ((sl, mf), r) in enumerate(sorted_r[:5], 1):
        print(f"{rank:<5} {sl:<35} {mf:<26} {r['daily_avg']:<+10.3f} {r['sharpe']:<8.2f}")

    print()
    print("TOP 5 BY DAILY AVG")
    print(f"{'Rank':<5} {'SL config':<35} {'Maker fill':<26} {'Daily':<10} {'Sharpe':<8}")
    print("-" * 100)
    sorted_d = sorted(results.items(), key=lambda x: x[1]["daily_avg"], reverse=True)
    for rank, ((sl, mf), r) in enumerate(sorted_d[:5], 1):
        print(f"{rank:<5} {sl:<35} {mf:<26} {r['daily_avg']:<+10.3f} {r['sharpe']:<8.2f}")

    # ---- 8. Save ----
    out = {f"{sl} | {mf}": {
        "daily_avg": r["daily_avg"], "daily_std": r["daily_std"],
        "sharpe": r["sharpe"], "max_dd": r["max_dd"],
        "total_trades": r["total_trades"],
        "avg_maker_rate": r["avg_maker_rate"], "avg_sl_rate": r["avg_sl_rate"],
    } for (sl, mf), r in results.items()}

    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/phase2_queue_sim_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved: {out_path}")
    log.info("Phase 2 complete")


if __name__ == "__main__":
    main()
