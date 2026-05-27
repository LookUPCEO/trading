"""Audit 06: Orderbook deep-level usage 검증."""
import sys, logging, json, re
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range
from mark19.ml.data_prep import DATES_TRAIN, build_split, get_feature_columns

# Reuse self builder via importlib
import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    print("=" * 80)
    print("AUDIT 06: Orderbook Deep-level Usage")
    print("=" * 80)

    # ---- 1. Raw orderbook columns (Self & Tardis) ----
    log.info("\n[1] Raw orderbook columns")
    start = datetime(2026, 4, 28, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    self_ob = read_range("orderbook", "bybit", "ETHUSDT", start, end)
    log.info(f"  Self  bybit       : {len(self_ob)} rows, {len(self_ob.columns)} cols")

    tar_start = datetime(2025, 4, 1, tzinfo=timezone.utc)
    tar_end = tar_start + timedelta(hours=1)
    try:
        tar_ob = read_range("orderbook", "bybit_tardis", "ETHUSDT", tar_start, tar_end)
        log.info(f"  Tardis bybit_tardis: {len(tar_ob)} rows, {len(tar_ob.columns)} cols")
    except Exception as e:
        log.warning(f"  Tardis read failed: {e}")
        tar_ob = pd.DataFrame()

    def levels_of(df, prefix):
        cols = [c for c in df.columns if c.startswith(prefix)]
        nums = []
        for c in cols:
            m = re.match(rf"{prefix}(\d+)_", c)
            if m: nums.append(int(m.group(1)))
        return sorted(set(nums)), cols

    print()
    print(f"{'Source':<14} {'bid_size lvls':<16} {'ask_size lvls':<16} {'bid_price lvls':<16} {'ask_price lvls':<16}")
    print("-" * 80)
    for label, df in [("Self bybit", self_ob), ("Tardis", tar_ob)]:
        if len(df) == 0:
            print(f"{label:<14} (no data)")
            continue
        bs, _ = levels_of(df, "bid_")
        bs_size = [n for n in bs if f"bid_{n}_size" in df.columns]
        bs_price = [n for n in bs if f"bid_{n}_price" in df.columns]
        as_size = [n for n in bs if f"ask_{n}_size" in df.columns]
        as_price = [n for n in bs if f"ask_{n}_price" in df.columns]
        print(f"{label:<14} {f'{len(bs_size)} ({min(bs_size)}-{max(bs_size)})':<16} "
              f"{f'{len(as_size)} ({min(as_size)}-{max(as_size)})':<16} "
              f"{f'{len(bs_price)} ({min(bs_price)}-{max(bs_price)})':<16} "
              f"{f'{len(as_price)} ({min(as_price)}-{max(as_price)})':<16}")

    # ---- 2. Sample values L0-L4 ----
    if len(self_ob) > 0:
        print()
        print("Sample values (Self, first row):")
        r = self_ob.iloc[0]
        for n in range(5):
            bp = r.get(f"bid_{n}_price"); bs = r.get(f"bid_{n}_size")
            ap = r.get(f"ask_{n}_price"); as_ = r.get(f"ask_{n}_size")
            print(f"  L{n}: bid {bp} x {bs}  |  ask {ap} x {as_}")

    # ---- 3. After feature build: how many deep-level features ----
    print()
    print("=" * 80)
    print("[3] Features used after build (Self test 4/28)")
    print("=" * 80)

    # Need Tardis medians for self DT synth
    log.info("Building Tardis (small subset for medians)...")
    tardis_train_df = build_split(DATES_TRAIN[:5], log)
    tardis_train_df.dropna(subset=["target_volatility_300s", "target_return_3600s"], inplace=True)
    feat_pre = get_feature_columns(tardis_train_df)
    medians = tardis_train_df.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self 4/28...")
    self_df = build_self_date_dataset("2026-04-28", log, train_medians=medians)
    self_df.dropna(subset=["target_volatility_300s", "target_return_3600s"], inplace=True)
    log.info(f"  Self 4/28: {len(self_df)} rows, {len(self_df.columns)} cols")

    feat_self = get_feature_columns(self_df)
    feat_tardis = get_feature_columns(tardis_train_df)
    log.info(f"  feature_columns Tardis: {len(feat_tardis)}")
    log.info(f"  feature_columns Self  : {len(feat_self)}")

    # Categorize features by orderbook level
    def cat_level(name):
        m = re.search(r"(?:bid|ask)_?(\d+)", name)
        if m:
            return int(m.group(1))
        m = re.search(r"top(\d+)", name)
        if m:
            return int(m.group(1))
        return None

    def bucket(features):
        b = {"L0": 0, "L1-4": 0, "L5-9": 0, "L10-24": 0, "L25+": 0, "no-level": 0}
        examples = {k: [] for k in b}
        for f in features:
            lvl = cat_level(f)
            if lvl is None:
                b["no-level"] += 1; examples["no-level"].append(f) if len(examples["no-level"]) < 3 else None
            elif lvl == 0:
                b["L0"] += 1; examples["L0"].append(f) if len(examples["L0"]) < 3 else None
            elif 1 <= lvl <= 4:
                b["L1-4"] += 1; examples["L1-4"].append(f) if len(examples["L1-4"]) < 3 else None
            elif 5 <= lvl <= 9:
                b["L5-9"] += 1; examples["L5-9"].append(f) if len(examples["L5-9"]) < 3 else None
            elif 10 <= lvl <= 24:
                b["L10-24"] += 1; examples["L10-24"].append(f) if len(examples["L10-24"]) < 3 else None
            else:
                b["L25+"] += 1; examples["L25+"].append(f) if len(examples["L25+"]) < 3 else None
        return b, examples

    print()
    print(f"{'Bucket':<12} {'Tardis':<10} {'Self':<10} {'Examples (Self)':<60}")
    print("-" * 100)
    bt, exT = bucket(feat_tardis)
    bs, exS = bucket(feat_self)
    for k in ["L0", "L1-4", "L5-9", "L10-24", "L25+", "no-level"]:
        ex = ", ".join(exS.get(k, [])[:3])
        print(f"{k:<12} {bt[k]:<10} {bs[k]:<10} {ex[:58]:<60}")

    # Deep-level columns in raw orderbook NOT in features
    raw_levels_self = [n for n in range(50) if f"bid_{n}_size" in self_ob.columns]
    deep_unused = [n for n in raw_levels_self if n >= 5 and not any(f"_{n}_" in f or f"top{n}" in f for f in feat_self)]
    print(f"\nRaw orderbook levels available (Self): L0-L{max(raw_levels_self)} = {len(raw_levels_self)} levels")
    print(f"Deep levels (L5+) NOT directly used in features: {len(deep_unused)} of {len([n for n in raw_levels_self if n >= 5])}")

    # ---- 4. Correlation L0 vs L5 vs L10 (size sums) ----
    print()
    print("=" * 80)
    print("[4] L0 vs L5 vs L10 size correlation (Self 4/28, 1-min resampled)")
    print("=" * 80)

    if len(self_ob) > 0:
        ts_col = "timestamp" if "timestamp" in self_ob.columns else self_ob.index.name
        if ts_col != self_ob.index.name:
            self_ob_t = self_ob.set_index(pd.to_datetime(self_ob[ts_col], utc=True))
        else:
            self_ob_t = self_ob.copy()

        # Re-load full day
        log.info("Loading full Self 4/28 OB for correlation...")
        d_start = datetime(2026, 4, 28, tzinfo=timezone.utc)
        d_end = d_start + timedelta(days=1)
        ob_day = read_range("orderbook", "bybit", "ETHUSDT", d_start, d_end)
        ts_col = "timestamp" if "timestamp" in ob_day.columns else None
        if ts_col:
            ob_day = ob_day.set_index(pd.to_datetime(ob_day[ts_col], utc=True))
        ob_day = ob_day.sort_index()

        bid_imb_L0 = ob_day["bid_0_size"] / (ob_day["bid_0_size"] + ob_day["ask_0_size"]) - 0.5

        for top_n in [5, 10, 25]:
            bid_cols = [f"bid_{i}_size" for i in range(top_n) if f"bid_{i}_size" in ob_day.columns]
            ask_cols = [f"ask_{i}_size" for i in range(top_n) if f"ask_{i}_size" in ob_day.columns]
            if not bid_cols or not ask_cols: continue
            bid_sum = ob_day[bid_cols].sum(axis=1)
            ask_sum = ob_day[ask_cols].sum(axis=1)
            imb = bid_sum / (bid_sum + ask_sum) - 0.5
            c = bid_imb_L0.corr(imb)
            print(f"  imbL0 vs imbTop{top_n}: corr {c:+.3f}")

        # Test directional informativeness: imb at L10 vs forward 10-min return
        ob_1min = ob_day[["bid_0_price", "ask_0_price"]].resample("1min").last()
        ob_1min["mid"] = (ob_1min["bid_0_price"] + ob_1min["ask_0_price"]) / 2
        ret_10 = (ob_1min["mid"].shift(-10) - ob_1min["mid"]) / ob_1min["mid"] * 100

        print()
        print("Forward 10-min return correlation with deep-imb (Self 4/28):")
        for top_n in [1, 5, 10, 25]:
            bid_cols = [f"bid_{i}_size" for i in range(top_n) if f"bid_{i}_size" in ob_day.columns]
            ask_cols = [f"ask_{i}_size" for i in range(top_n) if f"ask_{i}_size" in ob_day.columns]
            if not bid_cols or not ask_cols: continue
            bid_sum = ob_day[bid_cols].sum(axis=1).resample("1min").last()
            ask_sum = ob_day[ask_cols].sum(axis=1).resample("1min").last()
            imb = (bid_sum / (bid_sum + ask_sum) - 0.5)
            c = imb.corr(ret_10)
            print(f"  imbTop{top_n} vs ret_10min: corr {c:+.4f}")

    # ---- 5. Diagnosis ----
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    deep_self = bs["L5-9"] + bs["L10-24"] + bs["L25+"]
    deep_tardis = bt["L5-9"] + bt["L10-24"] + bt["L25+"]
    print(f"\nDeep level features (L5+):")
    print(f"  Tardis: {deep_tardis}")
    print(f"  Self  : {deep_self}")

    if deep_self < 3 and len(raw_levels_self) >= 25:
        print(f"\n[+] OPPORTUNITY: Self OB has {len(raw_levels_self)} levels but only {deep_self} deep features used")
        print(f"    → 시도 23c (Deep OB features) 가치 있음")
    elif deep_self >= 5:
        print(f"\n[~] Deep features 이미 {deep_self}개 사용 중 — 추가 효과 제한적")
    else:
        print(f"\n[?] Self OB level {len(raw_levels_self) if raw_levels_self else 0}개, deep features {deep_self}개")

    # ---- 6. Save ----
    out = {
        "raw_levels_self": len(raw_levels_self) if raw_levels_self else 0,
        "feat_count_tardis": len(feat_tardis),
        "feat_count_self": len(feat_self),
        "buckets_tardis": bt,
        "buckets_self": bs,
        "deep_features_unused_levels": deep_unused if 'deep_unused' in dir() else [],
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/audit06_orderbook.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")
    log.info("Audit 06 complete")


if __name__ == "__main__":
    main()
