"""시도 20: 자체 collector 데이터 OOS 검증 (시도 17 모델)."""
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from mark19.storage import read_range
from mark19.features.orderbook import compute_all_pointwise
from mark19.features.orderbook_timeseries import compute_rolling_stats, compute_obi_persistence
from mark19.features.trades import aggregate_to_1s, compute_rolling_features as compute_trades_rolling
from mark19.features.liquidation import compute_liquidation_features
from mark19.features.lagged import add_lagged_features
from mark19.features.cross import add_cross_features
from mark19.features.adaptive import add_adaptive_features
from mark19.ml.data_prep import (
    DATES_TRAIN, build_split, get_feature_columns, LAG_FEATURES, LAGS,
)

from live_bot.dt_adapter import load_funding_current, synthesize_dt_dataframe


SYMBOL = "ETHUSDT"


def build_self_date_dataset(date_str, log, train_medians=None):
    """build_date_dataset mirror, exchange='bybit'. DT synth via dt_adapter."""
    y, m, d = map(int, date_str.split("-"))
    start = datetime(y, m, d, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    log.info(f"  Building self-date {date_str}")

    features = {}
    ob_raw = read_range("orderbook", "bybit", SYMBOL, start, end)
    if len(ob_raw) > 100:
        ob_pw = compute_all_pointwise(ob_raw)
        ob_rs = compute_rolling_stats(ob_pw, "mid_price", [60, 300, 900])
        ob_op = compute_obi_persistence(ob_pw, "obi_top5", [60, 300])
        ob_pw_idx = ob_pw.set_index("timestamp") if "timestamp" in ob_pw.columns else ob_pw
        features["orderbook"] = pd.concat([ob_pw_idx, ob_rs, ob_op], axis=1).reset_index()

    tr_raw = read_range("trades", "bybit", SYMBOL, start, end)
    if len(tr_raw) > 1000:
        tr_agg = aggregate_to_1s(tr_raw)
        tr_rolling = compute_trades_rolling(tr_agg, [60, 300, 900])
        features["trades"] = pd.merge(tr_agg, tr_rolling, on="timestamp", how="outer")

    liq_raw = read_range("liquidation", "bybit", SYMBOL, start, end)
    if len(liq_raw) > 5:
        features["liquidation"] = compute_liquidation_features(liq_raw, [60, 300, 3600])

    if "orderbook" not in features:
        return pd.DataFrame()

    base = features["orderbook"].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True).dt.floor("1s")
    base = base.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
    full_idx = pd.date_range(base.index.min(), base.index.max(), freq="1s", tz="UTC")
    combined = base.reindex(full_idx)
    combined.columns = [f"ob_{c}" for c in combined.columns]

    if "trades" in features:
        t = features["trades"].copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True).dt.floor("1s")
        t = t.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        t = t.reindex(full_idx)
        t.columns = [f"tr_{c}" for c in t.columns]
        combined = combined.join(t)

    if "liquidation" in features:
        l = features["liquidation"].copy()
        l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True).dt.floor("1s")
        l = l.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        l = l.reindex(full_idx, fill_value=0)
        l.columns = [f"liq_{c}" for c in l.columns]
        combined = combined.join(l)

    # DT synth (live_bot.dt_adapter)
    funding_df = load_funding_current(start, end)
    medians_dict = train_medians.to_dict() if train_medians is not None else {}
    dt_df = synthesize_dt_dataframe(funding_df, full_idx, medians_dict)
    combined = combined.join(dt_df)

    if "ob_mid_price" not in combined.columns:
        return pd.DataFrame()

    mid = combined["ob_mid_price"]
    for N in [300, 900, 3600]:
        min_p = max(N // 2, 1)
        future_mid = mid.shift(-N)
        combined[f"target_return_{N}s"] = (future_mid - mid) / mid * 100
        combined[f"target_volatility_{N}s"] = mid.rolling(N, min_periods=min_p).std().shift(-(N-1))
        combined[f"target_max_drawdown_{N}s"] = (mid.rolling(N, min_periods=min_p).min().shift(-(N-1)) - mid) / mid * 100
        combined[f"target_max_runup_{N}s"] = (mid.rolling(N, min_periods=min_p).max().shift(-(N-1)) - mid) / mid * 100

    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    df_1min = combined.iloc[::60].copy().reset_index(drop=True)

    available_lag = [f for f in LAG_FEATURES if f in df_1min.columns]
    df_1min = add_lagged_features(df_1min, available_lag, LAGS)
    df_1min = add_cross_features(df_1min)
    df_1min = add_adaptive_features(df_1min)

    df_1min["_source_date"] = date_str
    return df_1min


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("시도 20: Self-Collected Data OOS Validation")
    log.info("=" * 70)
    np.random.seed(42)

    # ---- 1. Train Tardis 모델 (시도 17 reproduction) ----
    log.info("\nBuilding Tardis training data...")
    train_df = build_split(DATES_TRAIN, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    train_df.dropna(subset=[vol_target, dir_target], inplace=True)

    feature_cols = get_feature_columns(train_df)
    log.info(f"Features: {len(feature_cols)}")

    X_train_raw = train_df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_medians = X_train_raw.median(numeric_only=True)

    def make_X(df, feat_cols, medians):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(medians).fillna(0)

    X_train = make_X(train_df, feature_cols, train_medians)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    train_vol_median = train_df[vol_target].median()
    y_vol_train = (train_df[vol_target] > train_vol_median).astype(int).values
    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)

    T = 0.20
    train_dir_mask = train_df[dir_target].abs() > T
    X_train_dir = X_train[train_dir_mask].values
    y_dir_train = (train_df.loc[train_dir_mask, dir_target] > 0).astype(int).values
    scaler_dir = StandardScaler()
    X_train_dir_scaled = scaler_dir.fit_transform(X_train_dir)
    lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_dir.fit(X_train_dir_scaled, y_dir_train)

    log.info(f"시도 17 모델 학습 완료 (Tardis 26 train dates)")

    # ---- 2. Build self data ----
    log.info("\nBuilding self-collected data...")
    SELF_DATES = [f"2026-04-{d:02d}" for d in range(21, 31)]

    self_dfs = []
    for date_str in SELF_DATES:
        try:
            df = build_self_date_dataset(date_str, log, train_medians=train_medians)
            if len(df) > 0:
                self_dfs.append(df)
                log.info(f"    {date_str}: {len(df)} rows")
            else:
                log.warning(f"    {date_str}: empty")
        except Exception as e:
            log.error(f"    {date_str}: build error: {e}")

    if not self_dfs:
        log.error("Self data empty, abort")
        return

    self_df = pd.concat(self_dfs, ignore_index=True)
    self_df.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"Self total: {len(self_df)} rows  dates={sorted(self_df['_source_date'].unique())}")

    # ---- 3. Predict + AUC ----
    X_self = make_X(self_df, feature_cols, train_medians)
    X_self_vol_scaled = scaler_vol.transform(X_self)
    X_self_dir_scaled = scaler_dir.transform(X_self.values)

    self_df["vol_proba"] = lr_vol.predict_proba(X_self_vol_scaled)[:, 1]
    self_df["dir_proba"] = lr_dir.predict_proba(X_self_dir_scaled)[:, 1]
    self_df["actual_return"] = self_df[dir_target].values

    y_vol_self = (self_df[vol_target] > train_vol_median).astype(int).values
    vol_auc_self = roc_auc_score(y_vol_self, self_df["vol_proba"])

    self_dir_mask = self_df[dir_target].abs() > T
    if self_dir_mask.sum() > 10:
        y_dir_self = (self_df.loc[self_dir_mask, dir_target] > 0).astype(int).values
        dir_auc_self = roc_auc_score(y_dir_self, self_df.loc[self_dir_mask, "dir_proba"])
    else:
        dir_auc_self = float("nan")

    log.info(f"\nSelf-data AUC: Vol {vol_auc_self:.3f} | Direction {dir_auc_self:.3f}")
    log.info(f"  (Tardis 36-date baseline: Vol 0.762, Dir 0.545)")

    # ---- 4. Drift backtest on self data ----
    log.info("\nDrift policy backtest on self data...")
    DIR_THRESH, VOL_THRESH = 0.65, 0.6
    LOCKOUT_ROWS = 60
    SL_PCT = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025
    MAX_HOLD_MIN = 30

    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in self_df.columns), None)
    price_col = next((c for c in ["ob_mid_price", "mid", "mid_price"] if c in self_df.columns), None)
    self_df = self_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    def simulate_drift_fill(date_df, idx, direction):
        if idx >= len(date_df):
            return False, 0
        entry_mid = date_df.iloc[idx][price_col]
        if pd.isna(entry_mid):
            return False, 0
        limit_price = entry_mid * (0.99995 if direction == 1 else 1.00005)
        for t in range(1, MAX_HOLD_MIN + 1):
            if idx + t >= len(date_df):
                return False, t
            intra = date_df.iloc[idx + t][price_col]
            if pd.isna(intra):
                continue
            if direction == 1 and intra <= limit_price:
                return True, t
            if direction == -1 and intra >= limit_price:
                return True, t
            limit_price = intra * (0.99995 if direction == 1 else 1.00005)
        return False, MAX_HOLD_MIN

    date_results = []
    for date_str in SELF_DATES:
        date_df = self_df[self_df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
        if len(date_df) < 100:
            date_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                 "maker_rate": 0, "win_rate": 0, "sl_rate": 0})
            continue
        trades = []
        i, n = 0, len(date_df)
        while i < n:
            row = date_df.iloc[i]
            if pd.isna(row["actual_return"]) or pd.isna(row[price_col]):
                i += 1; continue
            vp, dp = row["vol_proba"], row["dir_proba"]
            direction = 0; trade = False
            if vp > VOL_THRESH:
                if dp > DIR_THRESH:
                    direction = 1; trade = True
                elif dp < (1 - DIR_THRESH):
                    direction = -1; trade = True
            if trade:
                entry_price = row[price_col]
                fee_entry = FEE_TAKER
                sl_hit = False
                actual_return = direction * row["actual_return"]
                for t in range(1, LOCKOUT_ROWS + 1):
                    if i + t >= n: break
                    intra = date_df.iloc[i + t][price_col]
                    if pd.isna(intra): continue
                    pnl_pct = direction * (intra - entry_price) / entry_price * 100
                    if pnl_pct <= -SL_PCT:
                        actual_return = -SL_PCT; sl_hit = True; break
                if sl_hit:
                    fee_exit = FEE_TAKER; fill_type = "sl_taker"
                else:
                    exit_idx = i + LOCKOUT_ROWS
                    filled, _ = simulate_drift_fill(date_df, exit_idx, -direction)
                    fee_exit = FEE_MAKER if filled else FEE_TAKER
                    fill_type = "maker" if filled else "taker_fallback"
                total_fee = fee_entry + fee_exit
                net_pnl = actual_return - total_fee
                trades.append({"net_pnl": net_pnl, "fill_type": fill_type, "sl_hit": sl_hit})
                i += LOCKOUT_ROWS
            else:
                i += 1
        if trades:
            pnl_sum = sum(t["net_pnl"] for t in trades)
            mc = sum(1 for t in trades if t["fill_type"] == "maker")
            wins = sum(1 for t in trades if t["net_pnl"] > 0)
            sl = sum(1 for t in trades if t["sl_hit"])
            date_results.append({
                "date": date_str, "n_trades": len(trades), "pnl_sum": pnl_sum,
                "maker_rate": mc/len(trades), "win_rate": wins/len(trades), "sl_rate": sl/len(trades),
            })
        else:
            date_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0,
                                 "maker_rate": 0, "win_rate": 0, "sl_rate": 0})

    daily_pnls = [d["pnl_sum"] for d in date_results]
    daily_avg = float(np.mean(daily_pnls)) if daily_pnls else 0.0
    daily_std = float(np.std(daily_pnls)) if len(daily_pnls) > 1 else 0.0
    sharpe = daily_avg / max(daily_std, 0.001)
    avg_maker = float(np.mean([d["maker_rate"] for d in date_results if d["n_trades"] > 0] or [0]))
    avg_win = float(np.mean([d["win_rate"] for d in date_results if d["n_trades"] > 0] or [0]))

    print()
    print("=" * 80)
    print("시도 20 RESULTS: Self-Data OOS (Drift policy + SL 1.5%)")
    print("=" * 80)
    print(f"\nDays tested: {sum(1 for d in date_results if d['n_trades'] > 0)} / {len(date_results)}")
    print(f"Total trades: {sum(d['n_trades'] for d in date_results)}")
    print(f"Daily avg: {daily_avg:+.3f}%  std {daily_std:.3f}%  Sharpe {sharpe:.2f}")
    print(f"Maker rate {avg_maker*100:.1f}%  Win rate {avg_win*100:.1f}%")
    print()
    print(f"{'Date':<14} {'Trades':<8} {'PnL':<12} {'Maker%':<8} {'Win%':<8} {'SL%':<6}")
    print("-" * 70)
    for d in date_results:
        print(f"{d['date']:<14} {d['n_trades']:<8} {d['pnl_sum']:<+12.3f} {d['maker_rate']*100:<8.1f} {d['win_rate']*100:<8.1f} {d['sl_rate']*100:<6.1f}")

    print()
    print("=" * 80)
    print("Tardis (Phase 3) vs Self-Data (시도 20)")
    print("=" * 80)
    print(f"{'Metric':<25} {'Tardis baseline':<18} {'Self':<18} {'Δ':<10}")
    print("-" * 75)
    print(f"{'Vol AUC':<25} {'0.762':<18} {vol_auc_self:<18.3f} {vol_auc_self-0.762:<+10.3f}")
    print(f"{'Direction AUC':<25} {'0.545':<18} {dir_auc_self:<18.3f} {dir_auc_self-0.545:<+10.3f}")
    print(f"{'Daily avg':<25} {'+1.105%':<18} {f'{daily_avg:+.3f}%':<18} {daily_avg-1.105:<+10.3f}p")
    print(f"{'Sharpe':<25} {'0.57':<18} {sharpe:<18.2f} {sharpe-0.57:<+10.2f}")
    print(f"{'Maker rate':<25} {'94.0%':<18} {f'{avg_maker*100:.1f}%':<18} {(avg_maker-0.94)*100:<+10.1f}p")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    auc_drop = abs(dir_auc_self - 0.545) if not np.isnan(dir_auc_self) else 99
    daily_gap = daily_avg - 1.105
    if auc_drop < 0.02 and abs(daily_gap) < 0.3:
        print("OK 모델 robust + Phase 3 신뢰 강화")
    elif auc_drop < 0.02 and daily_gap < -0.3:
        print("WARN AUC 유지 but Daily 하락 → maker fill model over-optimism")
    elif auc_drop > 0.02 and daily_avg < 0:
        print("FAIL AUC 하락 + Daily 음수 → 2026 covariate shift, 재학습 필요")
    else:
        print("PARTIAL degradation, 추가 진단 필요")

    out = {
        "self_data_dates": SELF_DATES,
        "vol_auc_self": float(vol_auc_self),
        "dir_auc_self": float(dir_auc_self) if not np.isnan(dir_auc_self) else None,
        "daily_avg": daily_avg, "daily_std": daily_std, "sharpe": sharpe,
        "maker_rate": avg_maker, "win_rate": avg_win,
        "per_date": date_results,
        "tardis_baseline": {"vol_auc": 0.762, "dir_auc": 0.545,
                            "daily_avg": 1.105, "sharpe": 0.57, "maker_rate": 0.94},
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido20_self_oos_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")
    log.info("시도 20 complete")


if __name__ == "__main__":
    main()
