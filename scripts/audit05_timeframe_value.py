"""Audit 05: Timeframe 단축 가치 검증."""
import sys, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from mark19.storage import read_range


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    print("=" * 80)
    print("AUDIT 05: Timeframe 단축 가치 검증")
    print("=" * 80)

    log.info("\nLoading orderbook 9 days...")
    end = datetime(2026, 4, 30, tzinfo=timezone.utc) + timedelta(days=1)
    start = datetime(2026, 4, 21, tzinfo=timezone.utc)
    ob_df = read_range("orderbook", "bybit", "ETHUSDT", start, end)
    log.info(f"Orderbook: {len(ob_df)} rows")

    if "ob_mid_price" in ob_df.columns:
        ob_df["mid"] = ob_df["ob_mid_price"]
    elif "ob_bid_0_price" in ob_df.columns and "ob_ask_0_price" in ob_df.columns:
        ob_df["mid"] = (ob_df["ob_bid_0_price"] + ob_df["ob_ask_0_price"]) / 2
    elif "bid_0_price" in ob_df.columns and "ask_0_price" in ob_df.columns:
        ob_df["mid"] = (ob_df["bid_0_price"] + ob_df["ask_0_price"]) / 2
    else:
        log.error("mid price column not found")
        log.info(f"Columns: {list(ob_df.columns)[:30]}")
        return

    if not isinstance(ob_df.index, pd.DatetimeIndex):
        ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in ob_df.columns), None)
        if ts_col:
            ob_df = ob_df.set_index(pd.to_datetime(ob_df[ts_col], utc=True))
        else:
            log.error("no timestamp column")
            return

    mid_1min = ob_df["mid"].resample("1min").last().dropna()
    log.info(f"1-min mid: {len(mid_1min)} rows ({mid_1min.index[0]} → {mid_1min.index[-1]})")

    print()
    print("=" * 100)
    print("FORWARD RETURN BY TIMEFRAME")
    print("=" * 100)

    timeframes = [5, 10, 15, 30, 60, 120]
    print(f"\n{'TF (min)':<10} {'N samples':<12} {'Mean ret':<13} {'Std ret':<13} {'AC lag1':<10} {'|R|>0.1%':<11} {'|R|>0.2%':<11} {'|R|>0.5%':<11}")
    print("-" * 100)

    tf_stats = {}
    for tf in timeframes:
        forward_ret = (mid_1min.shift(-tf) - mid_1min) / mid_1min * 100
        forward_ret = forward_ret.dropna()
        n = len(forward_ret)
        mean_r = float(forward_ret.mean())
        std_r = float(forward_ret.std())
        autocorr = float(forward_ret.autocorr(1)) if n > 2 else float("nan")
        ratio_01 = float((forward_ret.abs() > 0.1).mean())
        ratio_02 = float((forward_ret.abs() > 0.2).mean())
        ratio_05 = float((forward_ret.abs() > 0.5).mean())
        avg_abs = float(forward_ret.abs().mean())
        tf_stats[tf] = {"n": n, "mean": mean_r, "std": std_r, "autocorr": autocorr,
                        "ratio_01": ratio_01, "ratio_02": ratio_02, "ratio_05": ratio_05,
                        "avg_abs": avg_abs}
        print(f"{tf:<10} {n:<12} {mean_r:<+13.5f} {std_r:<13.5f} {autocorr:<10.3f} {ratio_01:<11.1%} {ratio_02:<11.1%} {ratio_05:<11.1%}")

    print()
    print("=" * 100)
    print("PERFECT FORESIGHT BY TIMEFRAME (이론값, vol filter 없음, entry taker + exit maker)")
    print("=" * 100)
    FEE_TAKER = 0.055
    FEE_MAKER = -0.025
    fee_round = FEE_TAKER + FEE_MAKER  # 0.030%

    print(f"\n{'TF (min)':<10} {'Trades/day':<12} {'Avg |R|':<12} {'Daily Max gross':<18} {'Daily after fee':<18}")
    print("-" * 80)
    for tf in timeframes:
        n_per_day = 1440 // tf
        avg_abs = tf_stats[tf]["avg_abs"]
        daily_max = avg_abs * n_per_day
        daily_after = daily_max - fee_round * n_per_day
        print(f"{tf:<10} {n_per_day:<12} {avg_abs:<12.5f}% {daily_max:<+18.2f}% {daily_after:<+18.2f}%")

    print()
    print("=" * 100)
    print("BREAK-EVEN ANALYSIS (일 1% target, vol filter 60% 가정)")
    print("=" * 100)
    print(f"\n{'TF (min)':<10} {'N filtered':<12} {'Edge needed/trade':<22} {'AUC needed':<14}")
    print("-" * 70)
    target_daily = 1.0
    for tf in timeframes:
        n_per_day = 1440 // tf
        n_filtered = max(int(n_per_day * 0.6), 1)
        edge_needed = target_daily / n_filtered + fee_round
        avg_abs = tf_stats[tf]["avg_abs"]
        if avg_abs > 0:
            auc_needed = 0.5 + edge_needed / (2 * avg_abs)
        else:
            auc_needed = 1.0
        auc_str = f"AUC {auc_needed:.3f}" + (" (불가능)" if auc_needed > 0.95 else "")
        print(f"{tf:<10} {n_filtered:<12} {edge_needed:<+22.5f}% {auc_str:<14}")

    print()
    print("=" * 80)
    print("DATA PREP TARGET 생성 코드")
    print("=" * 80)
    import subprocess
    grep_paths = ["mark19/ml/data_prep.py", "mark19/ml"]
    for p in grep_paths:
        if Path(p).exists():
            result = subprocess.run(
                ["grep", "-rn", "-E",
                 r"target_return_|target_volatility_|forward_return|3600s|300s",
                 p],
                capture_output=True, text=True
            )
            if result.stdout:
                print(f"\n[{p}]")
                print(result.stdout[:3000])
                break

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    auc_60_actual = 0.561
    print(f"\n현재 (60min hold):")
    print(f"  Dir AUC actual:        {auc_60_actual:.3f}")
    print(f"  Autocorr (60min fwd):  {tf_stats[60]['autocorr']:+.3f}")
    print(f"  Avg |R|:               {tf_stats[60]['avg_abs']:.4f}%")
    print(f"  |R|>0.2% ratio:        {tf_stats[60]['ratio_02']:.1%}")

    print(f"\n10min hold candidate:")
    print(f"  Trades/day potential:  {1440//10}")
    print(f"  Autocorr (10min fwd):  {tf_stats[10]['autocorr']:+.3f}")
    print(f"  Avg |R|:               {tf_stats[10]['avg_abs']:.4f}%")
    print(f"  |R|>0.1% ratio:        {tf_stats[10]['ratio_01']:.1%}")
    print(f"  |R|>0.2% ratio:        {tf_stats[10]['ratio_02']:.1%}")

    ac10 = abs(tf_stats[10]["autocorr"])
    ac60 = abs(tf_stats[60]["autocorr"])
    ratio_ac = ac10 / max(ac60, 0.01)
    print(f"\n  Autocorr ratio 10/60:  {ratio_ac:.2f}x")

    if ac10 > 0.10:
        print(f"\n[+] 10min predictability 강함 (|AC| {ac10:.3f} > 0.10)")
        sig_verdict = "strong"
    elif ac10 > 0.05:
        print(f"\n[~] 10min predictability 약함 (|AC| {ac10:.3f}, 0.05~0.10)")
        sig_verdict = "weak"
    else:
        print(f"\n[-] 10min forward 거의 random walk (|AC| {ac10:.3f} < 0.05)")
        sig_verdict = "none"

    avg_abs_10 = tf_stats[10]["avg_abs"]
    daily_max_10 = avg_abs_10 * (1440 // 10) * 0.6
    print(f"\n  10min hold ceiling (vol-filter 60%, perfect): {daily_max_10:.1f}%")
    if daily_max_10 > 5:
        print(f"  [+] Ceiling 충분 (>5%)")
    elif daily_max_10 > 2:
        print(f"  [~] Ceiling 보통 (2~5%)")
    else:
        print(f"  [-] Ceiling 작음 (<2%)")

    print()
    print("--- 시도 27 진행 권장 ---")
    if sig_verdict == "strong" and daily_max_10 > 5:
        print("[GO] 시도 27 강력 권장 (predictability + ceiling 둘 다 OK)")
    elif sig_verdict in ("strong", "weak") and daily_max_10 > 2:
        print("[CAUTION] 시도 27 진행 가능 (확률 50:50)")
    elif sig_verdict == "none":
        print("[STOP] 시도 27 효과 의문 — 시도 28 (다른 시장) 또는 시도 29 (funding harvesting)")
    else:
        print("[STOP] 단축만으론 부족 — 다른 방향 검토")

    out = {
        "timeframes": {str(tf): {
            "n_samples": int(s["n"]),
            "mean_return": s["mean"],
            "std_return": s["std"],
            "autocorr_lag1": s["autocorr"],
            "avg_abs_return": s["avg_abs"],
            "abs_above_0_1pct": s["ratio_01"],
            "abs_above_0_2pct": s["ratio_02"],
            "abs_above_0_5pct": s["ratio_05"],
        } for tf, s in tf_stats.items()},
        "diagnosis": {
            "10min_autocorr": tf_stats[10]["autocorr"],
            "10min_ceiling_perfect_pct": daily_max_10,
            "verdict": sig_verdict,
        },
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/audit05_timeframe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")
    log.info("Audit 05 complete")


if __name__ == "__main__":
    main()
