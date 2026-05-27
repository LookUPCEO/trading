"""Track D: BTC and SOL OOS walk-forward.

Train: tardis_trial 4/29-5/4 (6 days)
Test:  tardis_trial 5/5-5/7 (3 days OOS)
Same mark36_v2 config (n=100 fixed, max_depth=6, mcw=100, DIR_TH 0.58, norm features).
"""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import build_date_dataset, get_feature_columns

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec36 = importlib.util.spec_from_file_location("_sido36", _HERE / "sido36_regime_invariant.py")
_mod36 = importlib.util.module_from_spec(_spec36)
_spec36.loader.exec_module(_mod36)
add_normalized_features = _mod36.add_normalized_features
HIGH_SHIFT_FEATURES = _mod36.HIGH_SHIFT_FEATURES


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("Track D: BTC + SOL OOS walk-forward (mark36_v2 config)")
    log.info("  Train: tardis_trial 4/29-5/4 (6d)")
    log.info("  Test:  tardis_trial 5/5-5/7 (3d OOS)")
    log.info("=" * 70)
    np.random.seed(42)

    TRAIN_DATES = ["2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"]
    TEST_DATES = ["2026-05-05", "2026-05-06", "2026-05-07"]

    summary = {}

    for symbol in ["BTCUSDT", "SOLUSDT"]:
        log.info(f"\n{'=' * 70}")
        log.info(f"=== {symbol} OOS ===")
        log.info(f"{'=' * 70}")

        # Build train
        train_dfs = []
        for d in TRAIN_DATES:
            df = build_date_dataset(d, log, exchange="bybit_tardis_trial", symbol=symbol)
            if len(df) > 0:
                df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
                train_dfs.append(df)
        if not train_dfs:
            log.error(f"  {symbol}: no train data"); continue
        train_df = pd.concat(train_dfs, ignore_index=True)
        log.info(f"\n  Train: {len(train_df)} rows ({len(train_dfs)} days)")

        # Build test
        test_dfs = {}
        for d in TEST_DATES:
            df = build_date_dataset(d, log, exchange="bybit_tardis_trial", symbol=symbol)
            if len(df) > 0:
                df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
                test_dfs[d] = df
                log.info(f"  Test {d}: {len(df)} rows")
        if not test_dfs:
            log.error(f"  {symbol}: no test data"); continue

        # Norm features
        train_df = add_normalized_features(train_df, log)
        for d in test_dfs:
            test_dfs[d] = add_normalized_features(test_dfs[d], log)

        norm_features = [c for c in train_df.columns if c.endswith("_norm")]
        canonical = get_feature_columns(train_df)
        feat_set = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
        log.info(f"  Feature set: {len(feat_set)}")

        vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
        T = 0.20

        meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb_lib

        # Vol LR
        vol_med = float(train_df[vol_target].median())
        y_vt = (train_df[vol_target] > vol_med).astype(int).values
        sv = StandardScaler(); X_tv = sv.fit_transform(Xt)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_tv, y_vt)

        # Direction filter + train
        tm = train_df[dir_target].abs() > T
        Xt_f = Xt[tm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        if len(set(y_dt)) < 2:
            log.warning(f"  {symbol}: train dir single class, skip"); continue
        log.info(f"  Dir train: {len(y_dt)} (long {(y_dt==1).sum()}, short {(y_dt==0).sum()})")

        clf = xgb_lib.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.03,
                                      min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                      reg_alpha=1.0, reg_lambda=5.0,
                                      random_state=42, n_jobs=4, eval_metric="auc")
        clf.fit(Xt_f, y_dt, verbose=False)
        log.info(f"  Train AUC: {roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1]):.3f}")

        # OOS evaluation
        DIR_THS = [0.50, 0.55, 0.58]
        VOL_TH = 0.6
        LOCKOUT = 60; SL = 1.5
        FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30

        daily_results = []
        for d in sorted(test_dfs.keys()):
            df_test = test_dfs[d]
            Xst = mx(df_test)
            sm = df_test[dir_target].abs() > T
            Xst_f = Xst[sm].values
            y_ds = (df_test.loc[sm, dir_target] > 0).astype(int).values
            if len(set(y_ds)) < 2:
                log.warning(f"  {d}: skip (single class)"); continue
            auc = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1])

            bt = df_test.copy().reset_index(drop=True)
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
                n_long = 0; n_short = 0
                while i < n:
                    r = d_df.iloc[i]
                    if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                        i += 1; continue
                    direction = 0; trade = False
                    if r["vol_proba"] > VOL_TH:
                        if r["dir_proba"] > dir_th: direction = 1; trade = True; n_long += 1
                        elif r["dir_proba"] < (1 - dir_th): direction = -1; trade = True; n_short += 1
                    if trade:
                        e = r[price_col]; ar = direction * r["actual_return"]; sl = False
                        for t in range(1, LOCKOUT + 1):
                            if i + t >= n: break
                            x = d_df.iloc[i + t][price_col]
                            if pd.isna(x): continue
                            p = direction * (x - e) / e * 100
                            if p <= -SL: ar = -SL; sl = True; break
                        if sl: fee_e = FEE_TAKER
                        else:
                            filled = drift_fill(d_df, i + LOCKOUT, -direction)
                            fee_e = FEE_MAKER if filled else FEE_TAKER
                        trades.append({"net_pnl": ar - (FEE_TAKER + fee_e)})
                        i += LOCKOUT
                    else:
                        i += 1
                n_total = len(trades)
                ps = sum(t["net_pnl"] for t in trades) if trades else 0
                wr = (sum(1 for t in trades if t["net_pnl"] > 0) / n_total) if n_total else 0
                per_th[str(dir_th)] = {"pnl": ps, "n_trades": n_total, "win_rate": wr,
                                        "n_long": n_long, "n_short": n_short}

            daily_results.append({"date": d, "auc": float(auc), "per_th": per_th})
            log.info(f"  {d}: AUC {auc:.3f}  TH0.58 {per_th['0.58']['pnl']:+.3f}% ({per_th['0.58']['n_trades']}t L{per_th['0.58']['n_long']}/S{per_th['0.58']['n_short']})")

        # Per-symbol summary
        print()
        print(f"=== {symbol} OOS Summary ===")
        for th in DIR_THS:
            print(f"  TH {th}:")
            pnls = []
            for r in daily_results:
                t = r["per_th"][str(th)]
                print(f"    {r['date']}: AUC {r['auc']:.3f}  PnL {t['pnl']:+.3f}%  ({t['n_trades']}t L{t['n_long']}/S{t['n_short']}, win {t['win_rate']*100:.1f}%)")
                pnls.append(t["pnl"])
            v = np.array(pnls); pos = (v > 0).sum()
            print(f"    Mean {v.mean():+.3f}%/day  Total {v.sum():+.3f}%  Positive {pos}/{len(pnls)}")

        summary[symbol] = {
            "daily": daily_results,
            "th_0.58_mean": float(np.mean([r["per_th"]["0.58"]["pnl"] for r in daily_results])),
            "th_0.58_positive": int(sum(1 for r in daily_results if r["per_th"]["0.58"]["pnl"] > 0)),
            "th_0.58_total": float(sum(r["per_th"]["0.58"]["pnl"] for r in daily_results)),
        }

    # Cross-asset summary
    print()
    print("=" * 70)
    print("CROSS-ASSET SUMMARY (TH 0.58, OOS 5/5-5/7)")
    print("=" * 70)
    print(f"{'Symbol':<14} {'Mean/day':<14} {'Total':<12} {'Positive':<10}")
    print("-" * 60)
    print(f"{'ETHUSDT (Track C)':<14} {'+1.333%':<14} {'+8.000%':<12} {'6/6':<10}  (note: 6 days)")
    for sym, s in summary.items():
        print(f"{sym:<14} {s['th_0.58_mean']:+.3f}%       {s['th_0.58_total']:+.3f}%      {s['th_0.58_positive']}/3")

    out = {"approach": "Track D BTC + SOL OOS", "summary": summary}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/track_d_btc_sol_oos.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # Diagnosis
    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    pos_assets = sum(1 for s in summary.values() if s["th_0.58_mean"] > 0)
    if pos_assets == len(summary):
        print(f"\n  ✅ Multi-asset: 모두 양수 → mark36_v2 framework cross-asset 가치 확인")
    elif pos_assets > 0:
        print(f"\n  🟡 Mixed: {pos_assets}/{len(summary)} positive → asset-specific 효과")
    else:
        print(f"\n  ❌ Multi-asset 모두 음수 → ETH-specific 가능성")
    log.info("\nTrack D complete")


if __name__ == "__main__":
    main()
