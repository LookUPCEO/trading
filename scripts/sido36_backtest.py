"""시도 36 backtest: walk-forward 9 days × DIR_TH sweep, daily PnL."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns

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
    log.info("시도 36 backtest: walk-forward 9 days × DIR_TH sweep")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = {}
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    tardis_train.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val.dropna(subset=[vol_target, dir_target], inplace=True)

    tardis_train = add_normalized_features(tardis_train, log)
    tardis_val = add_normalized_features(tardis_val, log)
    for d in self_dfs:
        self_dfs[d] = add_normalized_features(self_dfs[d], log)

    norm_features = [c for c in tardis_train.columns if c.endswith("_norm")]
    canonical = get_feature_columns(tardis_train)
    feat_norm_only = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
    log.info(f"\nFeature set (norm_only): {len(feat_norm_only)}")

    test_dates = SELF_ALL[1:]
    T = 0.20

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    DIR_THS = [0.50, 0.52, 0.55, 0.58]
    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30

    walk_results = []
    for step_idx, test_date in enumerate(test_dates, 1):
        test_dt_idx = SELF_ALL.index(test_date)
        train_self_dates = SELF_ALL[:test_dt_idx]
        log.info(f"\n=== STEP {step_idx}/9: train Self {len(train_self_dates)}d, test {test_date} ===")

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_test_df = self_dfs[test_date].copy()
        train_df = pd.concat([tardis_train, self_train_df], ignore_index=True)
        val_df = tardis_val

        meds = train_df.reindex(columns=feat_norm_only).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=feat_norm_only).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df)

        # Vol LR (combined)
        vol_med = float(train_df[vol_target].median())
        y_vt = (train_df[vol_target] > vol_med).astype(int).values
        sv = StandardScaler(); X_tv = sv.fit_transform(Xt); X_sv = sv.transform(Xst)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_tv, y_vt)

        # Direction filter
        tm = train_df[dir_target].abs() > T
        vm = val_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xv_f = Xv[vm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values

        # XGB n1000 d6 mcw100
        clf = xgb_lib.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03,
                                      min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                      reg_alpha=1.0, reg_lambda=5.0,
                                      random_state=42, n_jobs=4, eval_metric="auc",
                                      early_stopping_rounds=30)
        clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        auc_self = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
        log.info(f"  AUC self: {auc_self:.3f}")

        # Backtest
        bt = self_test_df.copy().reset_index(drop=True)
        Xbt = mx(bt)
        bt["vol_proba"] = lrv.predict_proba(sv.transform(Xbt))[:, 1]
        bt["dir_proba"] = clf.predict_proba(Xbt.values)[:, 1]
        bt["actual_return"] = bt[dir_target].values

        ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in bt.columns), None)
        price_col = next((c for c in ["ob_mid_price", "mid"] if c in bt.columns), None)
        bt = bt.sort_values(ts_col).reset_index(drop=True)

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

        per_th = {}
        for dir_th in DIR_THS:
            d_df = bt
            trades = []; i, n = 0, len(d_df)
            n_sl = 0; n_maker = 0
            while i < n:
                r = d_df.iloc[i]
                if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                    i += 1; continue
                direction = 0; trade = False
                if r["vol_proba"] > VOL_TH:
                    if r["dir_proba"] > dir_th: direction = 1; trade = True
                    elif r["dir_proba"] < (1 - dir_th): direction = -1; trade = True
                if trade:
                    e = r[price_col]; ar = direction * r["actual_return"]; sl = False
                    for t in range(1, LOCKOUT + 1):
                        if i + t >= n: break
                        x = d_df.iloc[i + t][price_col]
                        if pd.isna(x): continue
                        p = direction * (x - e) / e * 100
                        if p <= -SL: ar = -SL; sl = True; break
                    if sl:
                        fee_e = FEE_TAKER; n_sl += 1
                    else:
                        filled = drift_fill(d_df, i + LOCKOUT, -direction)
                        fee_e = FEE_MAKER if filled else FEE_TAKER
                        if filled: n_maker += 1
                    trades.append({"net_pnl": ar - (FEE_TAKER + fee_e)})
                    i += LOCKOUT
                else:
                    i += 1
            n_total = len(trades)
            ps = sum(t["net_pnl"] for t in trades) if trades else 0
            wr = (sum(1 for t in trades if t["net_pnl"] > 0) / n_total) if n_total else 0
            maker_rate = (n_maker / max(n_total - n_sl, 1)) if n_total else 0
            per_th[str(dir_th)] = {"pnl": ps, "n_trades": n_total, "win_rate": wr,
                                    "n_sl": n_sl, "maker_rate": maker_rate}
            log.info(f"    TH {dir_th}: {ps:+.3f}% ({n_total}t, win {wr*100:.1f}%, SL {n_sl}, maker {maker_rate*100:.0f}%)")

        walk_results.append({"step": step_idx, "test_date": test_date,
                             "n_train_self": len(train_self_dates),
                             "auc_self": float(auc_self), "per_th": per_th})

    # Aggregate
    print()
    print("=" * 100)
    print("WALK-FORWARD BACKTEST (norm_only, XGB n1000 d6 mcw100)")
    print("=" * 100)
    for th in DIR_THS:
        print(f"\nDIR_TH = {th}")
        print(f"{'Step':<6} {'Date':<14} {'AUC':<8} {'PnL':<10} {'Trades':<8} {'Win':<8} {'SL':<5} {'Maker':<8}")
        print("-" * 70)
        pnls = []
        for r in walk_results:
            t = r["per_th"][str(th)]
            print(f"{r['step']:<6} {r['test_date']:<14} {r['auc_self']:<8.3f} {t['pnl']:<+10.3f}% {t['n_trades']:<8} {t['win_rate']*100:<8.1f}% {t['n_sl']:<5} {t['maker_rate']*100:<8.0f}%")
            pnls.append(t["pnl"])
        v = np.array(pnls); pos = (v > 0).sum()
        print(f"  Mean {v.mean():+.3f}%/day  Std {v.std():.3f}  Min {v.min():+.3f}  Max {v.max():+.3f}  Positive {pos}/9  Total {v.sum():+.3f}%")

    out = {"approach": "시도 36 norm_only walk-forward backtest", "steps": walk_results}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido36_backtest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # Best TH
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    best_th = None; best_mean = -999
    for th in DIR_THS:
        v = np.array([r["per_th"][str(th)]["pnl"] for r in walk_results])
        if v.mean() > best_mean:
            best_mean = v.mean(); best_th = th
    print(f"\n  Best TH: {best_th}  mean PnL {best_mean:+.3f}%/day")
    print(f"  vs 시도 29f (best): +0.541%/day (4 days)")
    print(f"  vs 시도 29g (9 days walk-forward): -0.046%/day")
    if best_mean >= 0.5:
        print(f"\n  ✅ 시도 36 backtest robust (≥0.5%/day, 9 days)")
    elif best_mean >= 0.0:
        print(f"\n  🟡 marginal positive")
    else:
        print(f"\n  ❌ negative — AUC 향상이 backtest로 translate 안됨")
    log.info("\n시도 36 backtest complete")


if __name__ == "__main__":
    main()
