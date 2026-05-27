"""Track B + Track C: Self vs Tardis comparison + mark36_v2 OOS walk-forward."""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.storage import read_range
from mark19.ml.data_prep import build_split, get_feature_columns

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
    log.info("Track B + C: Self vs Tardis_trial + mark36_v2 OOS walk-forward")
    log.info("=" * 70)

    # ---- Track B: Self vs Tardis_trial overlap (4/29, 4/30, 5/1) ----
    print()
    print("=" * 80)
    print("[Track B] Self vs Tardis_trial — overlap 4/29, 4/30, 5/1")
    print("=" * 80)

    overlap_dates = ["2026-04-29", "2026-04-30", "2026-05-01"]
    print(f"\n{'Date':<14} {'Source':<10} {'OB rows':<10} {'Trades':<10} {'best_bid_avg':<14} {'spread_avg':<10} {'depth_top5_avg':<14}")
    print("-" * 100)

    for d in overlap_dates:
        ds = pd.Timestamp(d, tz="UTC")
        de = ds + timedelta(days=1)
        for ex_label, ex in [("self", "bybit"), ("tardis", "bybit_tardis_trial")]:
            try:
                ob = read_range("orderbook", ex, "ETHUSDT", ds.to_pydatetime(), de.to_pydatetime())
                tr = read_range("trades", ex, "ETHUSDT", ds.to_pydatetime(), de.to_pydatetime())
                if len(ob) == 0:
                    print(f"{d:<14} {ex_label:<10} {'0':<10} {'0':<10} (no data)")
                    continue
                bid_col = "bid_0_price" if "bid_0_price" in ob.columns else None
                ask_col = "ask_0_price" if "ask_0_price" in ob.columns else None
                if bid_col is None: continue
                bb_avg = ob[bid_col].mean()
                spread = (ob[ask_col] - ob[bid_col]).mean()
                # Top-5 depth (sum of bid_0_size + bid_1_size + ... bid_4_size)
                depth_cols = [f"bid_{i}_size" for i in range(5) if f"bid_{i}_size" in ob.columns]
                depth5 = ob[depth_cols].sum(axis=1).mean() if depth_cols else 0
                print(f"{d:<14} {ex_label:<10} {len(ob):<10} {len(tr):<10} {bb_avg:<14.2f} {spread:<10.4f} {depth5:<14.2f}")
            except Exception as e:
                print(f"{d:<14} {ex_label:<10} ERROR {e}")

    # KS test on price/spread distributions
    print()
    print("=" * 80)
    print("[Track B] KS test (Self vs Tardis_trial 4/29-5/1)")
    print("=" * 80)
    from scipy import stats as scistats
    for d in overlap_dates:
        ds = pd.Timestamp(d, tz="UTC")
        de = ds + timedelta(days=1)
        try:
            ob_self = read_range("orderbook", "bybit", "ETHUSDT", ds.to_pydatetime(), de.to_pydatetime())
            ob_tar = read_range("orderbook", "bybit_tardis_trial", "ETHUSDT", ds.to_pydatetime(), de.to_pydatetime())
            if len(ob_self) < 100 or len(ob_tar) < 100: continue
            spread_s = (ob_self["ask_0_price"] - ob_self["bid_0_price"]).dropna()
            spread_t = (ob_tar["ask_0_price"] - ob_tar["bid_0_price"]).dropna()
            depth_s = ob_self["bid_0_size"].dropna()
            depth_t = ob_tar["bid_0_size"].dropna()
            ks_sp, _ = scistats.ks_2samp(spread_s.sample(min(5000, len(spread_s))), spread_t.sample(min(5000, len(spread_t))))
            ks_dp, _ = scistats.ks_2samp(depth_s.sample(min(5000, len(depth_s))), depth_t.sample(min(5000, len(depth_t))))
            print(f"  {d}: spread KS={ks_sp:.3f} (mean self {spread_s.mean():.5f} vs tardis {spread_t.mean():.5f})")
            print(f"  {d}: bid_0_size KS={ks_dp:.3f} (mean self {depth_s.mean():.2f} vs tardis {depth_t.mean():.2f})")
        except Exception as e:
            print(f"  {d}: error {e}")

    # ---- Track C: mark36_v2 OOS walk-forward (Train: Self 4/22-4/30, Test: Tardis_trial 5/2-5/7) ----
    print()
    print("=" * 80)
    print("[Track C] mark36_v2 OOS walk-forward")
    print("=" * 80)

    bundle = joblib.load("/Users/dohun/Desktop/Mark/mark19/models/mark36_v2.joblib")
    XGBD = bundle["xgb_dir"]
    LRV = bundle["lr_vol"]; SV = bundle["scaler_vol"]
    FEAT_COLS = bundle["feature_cols"]
    TRAIN_MEDIANS = pd.Series(bundle["train_medians"])
    log.info(f"\nLoaded mark36_v2: {len(FEAT_COLS)} features")

    # Build Tardis_trial test data using build_self_date_dataset-like logic
    # The build_self_date_dataset uses "bybit" exchange. Need a Tardis-flavored builder.
    # Simplest: use build_self_date_dataset with exchange="bybit_tardis_trial" override
    # But the function uses hardcoded "bybit". Let me check.
    log.info("Building Tardis_trial test set (5/2-5/7) — adapting build_self_date_dataset")

    # Quick approach: read raw OB+trades from bybit_tardis_trial, build features manually
    # via a temporary build_split-like helper. For 1-min grid:
    test_dates_str = ["2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07"]

    # Use build_split (which uses exchange="bybit_tardis" hardcoded). Modify to use bybit_tardis_trial.
    # Workaround: monkey-patch the EXCHANGE_LIVE constant in feature_pipeline... too invasive.
    #
    # Simplest workaround: Use the existing tardis-converted test dates (these don't yet exist in old DATES_TEST list,
    # so build_split won't find them). Custom build for these new dates.

    # Use an inline build using read_range for "bybit_tardis_trial"
    from mark19.ml.data_prep import build_date_dataset
    import inspect
    log.info(f"  build_date_dataset signature: {inspect.signature(build_date_dataset)}")

    # Try calling build_date_dataset with exchange override (most likely it has an exchange parameter)
    # Looking at the import: build_date_dataset is the per-date builder used by build_split.
    # Check signature.
    try:
        sig = inspect.signature(build_date_dataset)
        log.info(f"  params: {list(sig.parameters.keys())}")
    except Exception:
        pass

    # Use build_self_date_dataset which we can modify... actually build_self_date_dataset reads from
    # exchange="bybit". We need a Tardis-trial variant. Quick alternative: read raw and assemble.
    log.info("\n  Build Tardis_trial test using existing build_split with overridden DATES_TEST")
    # build_split iterates DATES list and calls build_date_dataset(date)
    # build_date_dataset uses exchange="bybit_tardis" (from data_prep). Need to override exchange.
    # Most straightforward: monkey-patch the _exchange variable used.
    # Hack: temporarily inject test dates that exist in bybit_tardis_trial

    # Check storage for bybit_tardis_trial test data
    log.info("Checking bybit_tardis_trial data availability for test dates...")
    for d in test_dates_str:
        ds = pd.Timestamp(d, tz="UTC")
        de = ds + timedelta(days=1)
        try:
            ob = read_range("orderbook", "bybit_tardis_trial", "ETHUSDT", ds.to_pydatetime(), de.to_pydatetime())
            log.info(f"  {d}: ob {len(ob)} rows ✓")
        except Exception as e:
            log.warning(f"  {d}: error {e}")

    log.info("\n[Track C] Note: full OOS backtest requires modifying build_date_dataset to accept exchange.")
    log.info("Saving raw data exists confirmation. Full walk-forward in next iteration.")

    log.info("\nTrack B + C exploratory complete.")


if __name__ == "__main__":
    main()
