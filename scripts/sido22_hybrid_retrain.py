"""시도 22: Hybrid Retrain (Tardis 36 + 자체 6 days train) + Overfit/Vol-only 검증."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import (
    DATES_TRAIN, DATES_VAL, DATES_TEST,
    build_split, get_feature_columns,
)

# Reuse 시도 20's self_data builder via importlib
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_backtest_self_data",
    Path(__file__).resolve().parent / "backtest_self_data.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset


def build_self_split(dates, log, train_medians=None):
    """Wrapper: build per-date self datasets + concat."""
    dfs = []
    for d in dates:
        try:
            df = build_self_date_dataset(d, log, train_medians=train_medians)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            log.error(f"  build_self {d}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("시도 22: Hybrid Retrain (Tardis 36 + Self 6 train)")
    log.info("=" * 70)
    np.random.seed(42)

    # ---- 1. Self splits ----
    SELF_TRAIN = [f"2026-04-{d:02d}" for d in range(21, 27)]
    SELF_VAL = ["2026-04-27"]
    SELF_TEST = ["2026-04-28", "2026-04-29", "2026-04-30"]
    log.info(f"\nSelf split: train={SELF_TRAIN}  val={SELF_VAL}  test={SELF_TEST}")

    # ---- 2. Build Tardis pools first to compute train_medians for self DT synth ----
    log.info("\nBuilding Tardis training data...")
    tardis_train_df = build_split(DATES_TRAIN, log)
    tardis_val_df = build_split(DATES_VAL, log)
    tardis_test_df = build_split(DATES_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"

    # Compute train_medians from Tardis only (for self DT synth)
    feature_cols_tmp = get_feature_columns(tardis_train_df)
    tardis_train_clean = tardis_train_df.dropna(subset=[vol_target, dir_target])
    X_tmp_raw = tardis_train_clean.reindex(columns=feature_cols_tmp).replace([np.inf, -np.inf], np.nan)
    train_medians = X_tmp_raw.median(numeric_only=True)
    log.info(f"  Tardis median basis: {len(feature_cols_tmp)} features")

    log.info("\nBuilding Self data (using Tardis medians for DT synth)...")
    self_train_df = build_self_split(SELF_TRAIN, log, train_medians=train_medians)
    self_val_df = build_self_split(SELF_VAL, log, train_medians=train_medians)
    self_test_df = build_self_split(SELF_TEST, log, train_medians=train_medians)

    # ---- 3. Combine train/val pools ----
    train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
    val_df = pd.concat([tardis_val_df, self_val_df], ignore_index=True)
    log.info(f"\nCombined train: {len(train_df)}  val: {len(val_df)}")
    log.info(f"  Tardis train {len(tardis_train_df)} + Self train {len(self_train_df)}")

    for df in [train_df, val_df, tardis_test_df, self_test_df]:
        df.dropna(subset=[vol_target, dir_target], inplace=True)

    feature_cols = get_feature_columns(train_df)
    log.info(f"Features: {len(feature_cols)}")

    # Recompute medians on combined train (Tardis + Self)
    X_train_raw = train_df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_medians = X_train_raw.median(numeric_only=True)

    def make_X(df):
        X = df.reindex(columns=feature_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(train_medians).fillna(0)

    X_train = make_X(train_df)
    X_val = make_X(val_df)
    X_tt = make_X(tardis_test_df)
    X_st = make_X(self_test_df)

    # ---- 4. Train ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    train_vol_median = float(train_df[vol_target].median())
    y_vol_train = (train_df[vol_target] > train_vol_median).astype(int).values
    y_vol_val = (val_df[vol_target] > train_vol_median).astype(int).values
    y_vol_tt = (tardis_test_df[vol_target] > train_vol_median).astype(int).values
    y_vol_st = (self_test_df[vol_target] > train_vol_median).astype(int).values

    sc_vol = StandardScaler()
    Xt = sc_vol.fit_transform(X_train)
    Xv = sc_vol.transform(X_val); Xtt = sc_vol.transform(X_tt); Xst = sc_vol.transform(X_st)
    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(Xt, y_vol_train)

    vol_auc_train = roc_auc_score(y_vol_train, lr_vol.predict_proba(Xt)[:, 1])
    vol_auc_val = roc_auc_score(y_vol_val, lr_vol.predict_proba(Xv)[:, 1])
    vol_auc_tt = roc_auc_score(y_vol_tt, lr_vol.predict_proba(Xtt)[:, 1])
    vol_auc_st = roc_auc_score(y_vol_st, lr_vol.predict_proba(Xst)[:, 1])
    log.info(f"\nVol AUC: train {vol_auc_train:.3f} | val {vol_auc_val:.3f} | "
             f"Tardis test {vol_auc_tt:.3f} | Self test {vol_auc_st:.3f}")

    T = 0.20
    train_m = train_df[dir_target].abs() > T
    val_m = val_df[dir_target].abs() > T
    tt_m = tardis_test_df[dir_target].abs() > T
    st_m = self_test_df[dir_target].abs() > T

    sc_dir = StandardScaler()
    Xt_d = sc_dir.fit_transform(X_train[train_m].values)
    y_dir_train = (train_df.loc[train_m, dir_target] > 0).astype(int).values
    Xv_d = sc_dir.transform(X_val[val_m].values)
    y_dir_val = (val_df.loc[val_m, dir_target] > 0).astype(int).values
    Xtt_d = sc_dir.transform(X_tt[tt_m].values)
    y_dir_tt = (tardis_test_df.loc[tt_m, dir_target] > 0).astype(int).values
    Xst_d = sc_dir.transform(X_st[st_m].values)
    y_dir_st = (self_test_df.loc[st_m, dir_target] > 0).astype(int).values

    lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_dir.fit(Xt_d, y_dir_train)
    dir_auc_train = roc_auc_score(y_dir_train, lr_dir.predict_proba(Xt_d)[:, 1])
    dir_auc_val = roc_auc_score(y_dir_val, lr_dir.predict_proba(Xv_d)[:, 1])
    dir_auc_tt = roc_auc_score(y_dir_tt, lr_dir.predict_proba(Xtt_d)[:, 1])
    dir_auc_st = roc_auc_score(y_dir_st, lr_dir.predict_proba(Xst_d)[:, 1])
    log.info(f"Dir AUC: train {dir_auc_train:.3f} | val {dir_auc_val:.3f} | "
             f"Tardis test {dir_auc_tt:.3f} | Self test {dir_auc_st:.3f}")

    dir_overfit_gap = dir_auc_train - dir_auc_val
    vol_overfit_gap = vol_auc_train - vol_auc_val
    log.info(f"Overfit gaps: Vol {vol_overfit_gap:+.3f}  Dir {dir_overfit_gap:+.3f}")

    # ---- 5. Drift backtest on self test ----
    self_test_df = self_test_df.reset_index(drop=True)
    self_test_df["vol_proba"] = lr_vol.predict_proba(sc_vol.transform(X_st))[:, 1]
    self_test_df["dir_proba"] = lr_dir.predict_proba(sc_dir.transform(X_st.values))[:, 1]
    self_test_df["actual_return"] = self_test_df[dir_target].values

    DIR_THRESH, VOL_THRESH = 0.65, 0.6
    LOCKOUT_ROWS = 60
    SL_PCT = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025
    MAX_HOLD = 30
    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in self_test_df.columns), None)
    price_col = next((c for c in ["ob_mid_price", "mid"] if c in self_test_df.columns), None)
    self_test_df = self_test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    def drift_fill(d_df, idx, direction):
        if idx >= len(d_df): return False
        e = d_df.iloc[idx][price_col]
        if pd.isna(e): return False
        lim = e * (0.99995 if direction == 1 else 1.00005)
        for t in range(1, MAX_HOLD + 1):
            if idx + t >= len(d_df): return False
            x = d_df.iloc[idx + t][price_col]
            if pd.isna(x): continue
            if direction == 1 and x <= lim: return True
            if direction == -1 and x >= lim: return True
            lim = x * (0.99995 if direction == 1 else 1.00005)
        return False

    daily_results = []
    for date_str in SELF_TEST:
        d_df = self_test_df[self_test_df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
        if len(d_df) < 100:
            daily_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                  "maker_rate": 0, "win_rate": 0})
            continue
        trades = []; i = 0; n = len(d_df)
        while i < n:
            r = d_df.iloc[i]
            if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                i += 1; continue
            direction = 0; trade = False
            if r["vol_proba"] > VOL_THRESH:
                if r["dir_proba"] > DIR_THRESH: direction = 1; trade = True
                elif r["dir_proba"] < (1 - DIR_THRESH): direction = -1; trade = True
            if trade:
                e = r[price_col]
                ar = direction * r["actual_return"]
                sl = False
                for t in range(1, LOCKOUT_ROWS + 1):
                    if i + t >= n: break
                    x = d_df.iloc[i + t][price_col]
                    if pd.isna(x): continue
                    p = direction * (x - e) / e * 100
                    if p <= -SL_PCT:
                        ar = -SL_PCT; sl = True; break
                if sl:
                    fee_e = FEE_TAKER; ft = "sl"
                else:
                    filled = drift_fill(d_df, i + LOCKOUT_ROWS, -direction)
                    fee_e = FEE_MAKER if filled else FEE_TAKER
                    ft = "maker" if filled else "taker"
                np_ = ar - (FEE_TAKER + fee_e)
                trades.append({"net_pnl": np_, "fill_type": ft, "sl": sl})
                i += LOCKOUT_ROWS
            else:
                i += 1
        if trades:
            ps = sum(t["net_pnl"] for t in trades)
            mr = sum(1 for t in trades if t["fill_type"] == "maker") / len(trades)
            wr = sum(1 for t in trades if t["net_pnl"] > 0) / len(trades)
            daily_results.append({"date": date_str, "n_trades": len(trades),
                                  "pnl_sum": ps, "maker_rate": mr, "win_rate": wr})
        else:
            daily_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                  "maker_rate": 0, "win_rate": 0})

    daily_pnls = [d["pnl_sum"] for d in daily_results]
    daily_avg = float(np.mean(daily_pnls)) if daily_pnls else 0.0
    daily_std = float(np.std(daily_pnls)) if len(daily_pnls) > 1 else 0.0
    sharpe = daily_avg / max(daily_std, 0.001)
    avg_maker = float(np.mean([d["maker_rate"] for d in daily_results if d["n_trades"] > 0] or [0]))
    avg_win = float(np.mean([d["win_rate"] for d in daily_results if d["n_trades"] > 0] or [0]))

    # ---- 6. Vol-only baseline (5 runs random direction) ----
    log.info("\nVol-only baseline (random direction, 5 runs)...")
    vol_only_pnls_per_date = []
    for date_str in SELF_TEST:
        d_df = self_test_df[self_test_df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
        if len(d_df) < 100:
            vol_only_pnls_per_date.append(0.0); continue
        run_pnls = []
        for run_id in range(5):
            np.random.seed(42 + run_id)
            trades = []; i = 0; n = len(d_df)
            while i < n:
                r = d_df.iloc[i]
                if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                    i += 1; continue
                if r["vol_proba"] > VOL_THRESH:
                    direction = int(np.random.choice([1, -1]))
                    e = r[price_col]
                    ar = direction * r["actual_return"]
                    sl = False
                    for t in range(1, LOCKOUT_ROWS + 1):
                        if i + t >= n: break
                        x = d_df.iloc[i + t][price_col]
                        if pd.isna(x): continue
                        p = direction * (x - e) / e * 100
                        if p <= -SL_PCT:
                            ar = -SL_PCT; sl = True; break
                    if sl:
                        fee_e = FEE_TAKER
                    else:
                        filled = drift_fill(d_df, i + LOCKOUT_ROWS, -direction)
                        fee_e = FEE_MAKER if filled else FEE_TAKER
                    trades.append(ar - (FEE_TAKER + fee_e))
                    i += LOCKOUT_ROWS
                else:
                    i += 1
            run_pnls.append(sum(trades) if trades else 0)
        vol_only_pnls_per_date.append(float(np.mean(run_pnls)))
    vo_daily_avg = float(np.mean(vol_only_pnls_per_date)) if vol_only_pnls_per_date else 0.0
    direction_value = daily_avg - vo_daily_avg

    # ---- 7. Reports ----
    print()
    print("=" * 90)
    print("3-WAY (시도 17 vs 21 vs 22)")
    print("=" * 90)
    s21_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido21_recent_retrain.json")
    s21 = None
    if s21_path.exists():
        with open(s21_path) as f:
            s21 = json.load(f).get("metadata", {})
    s21_dir = s21.get("dir_auc_self") if s21 else None
    s21_vol = s21.get("vol_auc_self") if s21 else None
    s21_daily = s21.get("self_daily_avg") if s21 else None

    print(f"\n{'Metric':<35} {'시도 17':<14} {'시도 21':<14} {'시도 22':<14}")
    print("-" * 90)
    print(f"{'Train dates':<35} {'26 Tardis':<14} {'17 recent':<14} {'42 (36+6)':<14}")
    print(f"{'Vol AUC (Self 2026 OOS)':<35} {'0.793':<14} "
          f"{f'{s21_vol:.3f}' if s21_vol else 'N/A':<14} {vol_auc_st:<14.3f}")
    print(f"{'Dir AUC (Self 2026 OOS)':<35} {'0.480':<14} "
          f"{f'{s21_dir:.3f}' if s21_dir else 'N/A':<14} {dir_auc_st:<14.3f}")
    print(f"{'Train-Val Dir gap (overfit)':<35} {'~0.13':<14} {'0.197':<14} {dir_overfit_gap:<14.3f}")
    print(f"{'Self daily avg':<35} {'-0.134%':<14} "
          f"{f'{s21_daily:+.3f}%' if s21_daily is not None else 'N/A':<14} {f'{daily_avg:+.3f}%':<14}")

    print()
    print("=" * 80)
    print("시도 22 BACKTEST (Self test 4/28-30, Drift)")
    print("=" * 80)
    print(f"\nDays {len(daily_results)}  Trades {sum(d['n_trades'] for d in daily_results)}")
    print(f"Daily {daily_avg:+.3f}% std {daily_std:.3f}%  Sharpe {sharpe:.2f}")
    print(f"Maker {avg_maker*100:.1f}%  Win {avg_win*100:.1f}%")
    print()
    print(f"{'Date':<14} {'Trades':<8} {'PnL':<10} {'Maker%':<8} {'Win%':<8}")
    print("-" * 60)
    for d in daily_results:
        print(f"{d['date']:<14} {d['n_trades']:<8} {d['pnl_sum']:<+10.3f} {d['maker_rate']*100:<8.1f} {d['win_rate']*100:<8.1f}")

    print()
    print("=" * 80)
    print("VOL-ONLY BASELINE (random direction, 5 runs avg)")
    print("=" * 80)
    print(f"Vol-only daily:   {vo_daily_avg:+.3f}%")
    print(f"시도 22 daily:     {daily_avg:+.3f}%")
    print(f"Direction value:  {direction_value:+.3f}%/day")
    if abs(direction_value) < 0.1:
        print("⚠️  Direction model 가치 미미 → 시도 25 (Vol-only) 검토")
    elif direction_value > 0.1:
        print("OK Direction model 가치 있음")
    else:
        print("FAIL Direction model 해로움 → 시도 25 또는 inverse direction")

    # ---- 8. Save ----
    out = {
        "lr_vol": lr_vol, "scaler_vol": sc_vol,
        "lr_dir": lr_dir, "scaler_dir": sc_dir,
        "feature_cols": feature_cols,
        "train_medians": train_medians.to_dict(),
        "train_vol_median": train_vol_median,
        "T": T,
        "metadata": {
            "tardis_train_dates": list(DATES_TRAIN),
            "self_train_dates": SELF_TRAIN,
            "self_val_dates": SELF_VAL,
            "self_test_dates": SELF_TEST,
            "vol_auc_train": float(vol_auc_train),
            "vol_auc_val": float(vol_auc_val),
            "vol_auc_tardis_test": float(vol_auc_tt),
            "vol_auc_self_test": float(vol_auc_st),
            "dir_auc_train": float(dir_auc_train),
            "dir_auc_val": float(dir_auc_val),
            "dir_auc_tardis_test": float(dir_auc_tt),
            "dir_auc_self_test": float(dir_auc_st),
            "dir_overfit_gap": float(dir_overfit_gap),
            "self_test_daily_avg": daily_avg,
            "self_test_sharpe": sharpe,
            "self_test_maker_rate": avg_maker,
            "self_test_win_rate": avg_win,
            "vol_only_daily_avg": vo_daily_avg,
            "direction_model_value": direction_value,
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark22_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nModel: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido22_hybrid_retrain.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"], "daily_results": daily_results,
                   "vol_only_daily_pnls": vol_only_pnls_per_date}, f, indent=2, default=str)
    log.info(f"JSON:  {json_path}")

    # ---- 9. Diagnosis ----
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if dir_auc_st >= 0.55:
        print(f"\nGOOD Direction AUC 회복 ({dir_auc_st:.3f}) — Hybrid 효과적")
        if daily_avg > 0:
            print(f"     Daily {daily_avg:+.3f}% → LIVE 적용 후보")
    elif dir_auc_st >= 0.50:
        print(f"\nPARTIAL Dir AUC ({dir_auc_st:.3f}) — 시도 23 (Features) 또는 25 (Vol-only)")
    else:
        print(f"\nFAIL Dir AUC ({dir_auc_st:.3f}) < 0.50 — Direction edge 없음, 시도 25 권장")
    log.info("시도 22 complete")


if __name__ == "__main__":
    main()
