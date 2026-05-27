"""Combined audit (1-4): data ceiling, market regime, sido17 attribution, LIVE infra."""
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import build_split, get_feature_columns, DATES_TEST

# Reuse self_data builder via importlib
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_backtest_self_data",
    Path(__file__).resolve().parent / "backtest_self_data.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset


def build_self_split(dates, log, train_medians=None):
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


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
np.random.seed(42)

# ---- Load model + datasets once ----
log.info("Loading mark22_v1 model + datasets...")
m = joblib.load("/Users/dohun/Desktop/Mark/mark19/models/mark22_v1.joblib")
lr_vol, sc_vol = m["lr_vol"], m["scaler_vol"]
lr_dir, sc_dir = m["lr_dir"], m["scaler_dir"]
feature_cols = m["feature_cols"]
train_medians = pd.Series(m["train_medians"])

vol_target = "target_volatility_300s"
dir_target = "target_return_3600s"

log.info("Building Tardis test (2024-2025)...")
tardis_df = build_split(DATES_TEST, log)
tardis_df.dropna(subset=[vol_target, dir_target], inplace=True)

log.info("Building Self 2026 (4/21-4/30)...")
self_df = build_self_split([f"2026-04-{d:02d}" for d in range(21, 31)], log,
                           train_medians=train_medians)
self_df.dropna(subset=[vol_target, dir_target], inplace=True)


def make_X(df):
    X = df.reindex(columns=feature_cols).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(train_medians).fillna(0)


# Predict
for df, name in [(tardis_df, "Tardis"), (self_df, "Self")]:
    Xx = make_X(df)
    df["vol_proba"] = lr_vol.predict_proba(sc_vol.transform(Xx))[:, 1]
    df["dir_proba"] = lr_dir.predict_proba(sc_dir.transform(Xx.values))[:, 1]
    df["actual_return"] = df[dir_target].values

ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in self_df.columns), None)
price_col = next((c for c in ["ob_mid_price", "mid"] if c in self_df.columns), None)
self_df = self_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)
tardis_df = tardis_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)


# ============================================================
# AUDIT 01: Perfect Foresight Maximum
# ============================================================
print()
print("=" * 80)
print("AUDIT 01: Perfect Foresight Max (Self 2026)")
print("=" * 80)

LOCKOUT = 60
FEE_TAKER, FEE_MAKER = 0.055, -0.025

def perfect_backtest(df, vol_thresh=None):
    daily = {}
    for date_str in df["_source_date"].unique():
        d_df = df[df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
        if len(d_df) < 100: continue
        trades = []; i = 0; n = len(d_df)
        while i < n:
            row = d_df.iloc[i]
            if pd.isna(row["actual_return"]):
                i += 1; continue
            if vol_thresh is None or row["vol_proba"] > vol_thresh:
                ret_abs = abs(row["actual_return"])
                net = ret_abs - FEE_TAKER - FEE_MAKER  # round-trip
                trades.append(net)
                i += LOCKOUT
            else:
                i += 1
        daily[date_str] = sum(trades) if trades else 0
    return daily

print("\n--- 1a. All entries (no vol filter) + perfect direction ---")
d1 = perfect_backtest(self_df, None)
for k, v in sorted(d1.items()):
    print(f"  {k}: {v:+.2f}%")
avg1 = np.mean(list(d1.values())) if d1 else 0
print(f"  Avg daily: {avg1:+.2f}%")

print("\n--- 1b. Vol filter (vol_proba > 0.6) + perfect direction ---")
d2 = perfect_backtest(self_df, 0.6)
for k, v in sorted(d2.items()):
    print(f"  {k}: {v:+.2f}%")
avg2 = np.mean(list(d2.values())) if d2 else 0
print(f"  Avg daily: {avg2:+.2f}%")

print(f"\n--- 1c. 일 1% 달성 가능성 ---")
print(f"  No filter (perfect):  {avg1:+.2f}%/day")
print(f"  Vol-filtered (perfect): {avg2:+.2f}%/day")
print(f"  Headroom over +1%:    {avg2 - 1.0:+.2f}%/day")
if avg2 >= 5.0:
    print("  ✅ Plenty of headroom — Direction edge needed")
elif avg2 >= 2.0:
    print("  ⚠️  Tight headroom — model + execution must both be good")
elif avg2 >= 1.0:
    print("  ❌ Marginal — near-perfect model required")
else:
    print("  ❌ Data ceiling below target — different timeframe/market needed")


# ============================================================
# AUDIT 02: Market Regime (Tardis vs Self)
# ============================================================
def regime_stats(df, name):
    rets = df[dir_target].values
    s = pd.Series(rets)
    print(f"\n--- {name} ---")
    print(f"  N: {len(rets)}  mean {np.mean(rets):+.4f}%  std {np.std(rets):.3f}%")
    print(f"  |R|>0.2%: {np.mean(np.abs(rets)>0.2):.1%}   |R|>0.5%: {np.mean(np.abs(rets)>0.5):.1%}   |R|>1.0%: {np.mean(np.abs(rets)>1.0):.1%}")
    print(f"  skew {s.skew():.3f}  kurt {s.kurtosis():.3f}")
    print(f"  autocorr lag1 {s.autocorr(1):.3f}  lag5 {s.autocorr(5):.3f}")
    direction = np.sign(rets)
    print(f"  long ratio {np.mean(direction>0):.1%}   same-dir-next {np.mean(direction[1:]==direction[:-1]):.1%}")

print()
print("=" * 80)
print("AUDIT 02: Market Regime (Tardis 2024-2025 vs Self 2026-04)")
print("=" * 80)
regime_stats(tardis_df, "Tardis test (2024-2025)")
regime_stats(self_df, "Self 2026-04 (9 days)")

t_a1 = pd.Series(tardis_df[dir_target].values).autocorr(1)
s_a1 = pd.Series(self_df[dir_target].values).autocorr(1)
print(f"\nLag-1 autocorr: Tardis {t_a1:+.3f} | Self {s_a1:+.3f}")
if abs(t_a1) > abs(s_a1) + 0.05:
    print("→ 2026 less serial-correlated. Direction harder to predict.")
elif abs(t_a1) < abs(s_a1) - 0.05:
    print("→ 2026 MORE serial-correlated, but model trained on different regime.")
else:
    print("→ 비슷한 predictability. Model 자체 limit 추정.")


# ============================================================
# AUDIT 03: 시도 17 Attribution (Tardis test)
# ============================================================
print()
print("=" * 80)
print("AUDIT 03: 시도 17 Attribution (model edge vs random vs perfect)")
print("=" * 80)

def strategy_backtest(df, dir_fn, vol_thresh=0.6, dir_thresh=0.65):
    daily = {}
    for date_str in df["_source_date"].unique():
        d_df = df[df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
        if len(d_df) < 100: continue
        trades = []; i = 0; n = len(d_df)
        while i < n:
            row = d_df.iloc[i]
            if pd.isna(row["actual_return"]):
                i += 1; continue
            if row["vol_proba"] > vol_thresh:
                direction = dir_fn(row, dir_thresh)
                if direction != 0:
                    net = direction * row["actual_return"] - FEE_TAKER - FEE_MAKER
                    trades.append(net)
                    i += LOCKOUT
                else:
                    i += 1
            else:
                i += 1
        daily[date_str] = sum(trades) if trades else 0
    return daily

def f_model(r, t):
    if r["dir_proba"] > t: return 1
    if r["dir_proba"] < (1 - t): return -1
    return 0
def f_random(r, t):
    return int(np.random.choice([1, -1]))
def f_perfect(r, t):
    return 1 if r["actual_return"] > 0 else -1

print("\n--- Tardis test (2024-2025) ---")
print(f"{'Strategy':<25} {'Daily':<10} {'vs Random':<10}")
print("-" * 50)

np.random.seed(42)
d_rand = strategy_backtest(tardis_df, f_random)
avg_rand = np.mean(list(d_rand.values())) if d_rand else 0
print(f"{'Random direction':<25} {avg_rand:<+10.3f} {'baseline':<10}")

d_model = strategy_backtest(tardis_df, f_model)
avg_model = np.mean(list(d_model.values())) if d_model else 0
print(f"{'Direction model':<25} {avg_model:<+10.3f} {avg_model-avg_rand:<+10.3f}")

d_perf = strategy_backtest(tardis_df, f_perfect)
avg_perf = np.mean(list(d_perf.values())) if d_perf else 0
print(f"{'Perfect oracle':<25} {avg_perf:<+10.3f} {avg_perf-avg_rand:<+10.3f}")

edge_t = avg_model - avg_rand
captured_t = (edge_t / max(avg_perf - avg_rand, 0.001)) * 100
print(f"\nTardis: model edge {edge_t:+.3f}%/day  ({captured_t:.1f}% of perfect)")

print("\n--- Self 2026 ---")
np.random.seed(42)
d_rand_s = strategy_backtest(self_df, f_random)
avg_rand_s = np.mean(list(d_rand_s.values())) if d_rand_s else 0
print(f"{'Random direction':<25} {avg_rand_s:<+10.3f} {'baseline':<10}")

d_model_s = strategy_backtest(self_df, f_model)
avg_model_s = np.mean(list(d_model_s.values())) if d_model_s else 0
print(f"{'Direction model':<25} {avg_model_s:<+10.3f} {avg_model_s-avg_rand_s:<+10.3f}")

d_perf_s = strategy_backtest(self_df, f_perfect)
avg_perf_s = np.mean(list(d_perf_s.values())) if d_perf_s else 0
print(f"{'Perfect oracle':<25} {avg_perf_s:<+10.3f} {avg_perf_s-avg_rand_s:<+10.3f}")

edge_s = avg_model_s - avg_rand_s
captured_s = (edge_s / max(avg_perf_s - avg_rand_s, 0.001)) * 100
print(f"\nSelf: model edge {edge_s:+.3f}%/day  ({captured_s:.1f}% of perfect)")


# ============================================================
# AUDIT 04: LIVE Infrastructure (log analysis)
# ============================================================
print()
print("=" * 80)
print("AUDIT 04: LIVE Infrastructure")
print("=" * 80)

import subprocess, glob, os
log_dir = "/Users/dohun/Desktop/Mark/mark19/logs"
logs = sorted(glob.glob(f"{log_dir}/live_small_*.log"), key=os.path.getmtime, reverse=True)
if logs:
    log_path = logs[0]
    print(f"\nAnalyzing: {Path(log_path).name}")
    with open(log_path) as f:
        content = f.read()
    print(f"  size: {len(content)/1024:.1f} KB")
    print(f"  cycles (vol=): {content.count('vol=')}")
    print(f"  OPENING POSITION: {content.count('OPENING POSITION')}")
    print(f"  TRADE FINALIZED: {content.count('TRADE FINALIZED')}")
    print(f"  Submitted MARKET: {content.count('Submitted MARKET')}")
    print(f"  Submitted LIMIT: {content.count('Submitted LIMIT')}")
    print(f"  Order rejected: {content.count('order rejected') + content.count('FAILED')}")
    print(f"  cancel_all calls: {content.count('cancel_all')}")
    print(f"  drift replace: {content.count('drift:')}")
    print(f"  Native SL set: {content.count('Native SL set')}")
    print(f"  Verify FAIL: {content.count('VERIFY')}")
    print(f"  Reconcile: {content.count('RECONCILE')}")
    print(f"  Parquet errors: {content.count('Parquet error counter')}")
    print(f"  ERROR lines: {content.count('[ERROR]')}")
    print(f"  WARNING lines: {content.count('[WARNING]')}")
else:
    print("\n  No LIVE logs found")


# ============================================================
# Summary
# ============================================================
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Data ceiling (Self 2026, vol-filtered + perfect direction): {avg2:+.2f}%/day
Tardis attribution: model captures {captured_t:.1f}% of perfect edge
Self attribution:   model captures {captured_s:.1f}% of perfect edge
Tardis lag-1 autocorr: {t_a1:+.3f}
Self   lag-1 autocorr: {s_a1:+.3f}

Layers:
  Data:     {'OK ceiling >>1%' if avg2 >= 2 else 'BORDERLINE' if avg2 >= 1 else 'TIGHT'}
  Market:   {'2026 different from training' if abs(t_a1-s_a1) > 0.05 else 'similar'}
  Strategy: {'Tardis edge OK' if captured_t > 30 else 'Tardis edge weak'},  Self {'edge present' if captured_s > 10 else 'edge absent'}
""")
