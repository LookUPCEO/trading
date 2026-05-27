"""Audit 07: 전체 데이터 인벤토리 + 가능 시도 매트릭스."""
import sys, logging, os, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import read_range


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    print("=" * 80)
    print("AUDIT 07: 데이터 인벤토리")
    print("=" * 80)

    DATA_ROOT = Path("/Users/dohun/Desktop/Mark/mark19/data")
    TARDIS_ROOT = Path("/Volumes/PortableSSD/40_사이드프로젝트/mark19_data")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=15)

    # ---- 1. Self-collected data subdirs ----
    print()
    print("--- Self-collected data subdirectories ---")
    if DATA_ROOT.exists():
        print(f"\n{DATA_ROOT}:")
        for subdir in sorted(DATA_ROOT.iterdir()):
            if not subdir.is_dir(): continue
            symbol_paths = []
            for root, dirs, _ in os.walk(subdir):
                for d in dirs:
                    if d.endswith("USDT") or d.endswith("USD"):
                        symbol_paths.append(os.path.join(root, d))
            if symbol_paths:
                print(f"  {subdir.name}:")
                seen = set()
                for sp in sorted(set(symbol_paths)):
                    rel = os.path.relpath(sp, DATA_ROOT)
                    if rel in seen: continue
                    seen.add(rel)
                    parquets = list(Path(sp).glob("*.parquet"))
                    if parquets:
                        files = sorted([p.stem for p in parquets])
                        print(f"    {rel:<55} {len(parquets)} files  ({files[0]} ~ {files[-1]})")

    # ---- 2. read_range probing ----
    print()
    print("--- read_range probing (last 15 days) ---")
    test_symbols = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "ETHUSD", "BTCUSD", "XRPUSDT", "DOGEUSDT"]
    test_datatypes = ["orderbook", "trades", "liquidation", "funding", "derivative_ticker"]
    test_exchanges = ["bybit", "bybit_tardis", "binance", "okx"]

    available_combos = {}
    for dt in test_datatypes:
        for ex in test_exchanges:
            for sym in test_symbols:
                try:
                    df = read_range(dt, ex, sym, start, end)
                    if len(df) > 0:
                        date_min = date_max = "?"
                        if isinstance(df.index, pd.DatetimeIndex):
                            date_min = str(df.index.min())
                            date_max = str(df.index.max())
                        elif "timestamp" in df.columns:
                            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                            date_min = str(ts.min()); date_max = str(ts.max())
                        available_combos[f"{dt} / {ex} / {sym}"] = {
                            "rows": int(len(df)),
                            "cols": int(len(df.columns)),
                            "date_min": date_min, "date_max": date_max,
                        }
                except Exception:
                    pass

    print(f"\nAvailable combos (last 15d): {len(available_combos)}")
    for key, info in sorted(available_combos.items()):
        print(f"  {key:<48} rows={info['rows']:<10} cols={info['cols']}")

    # ---- 3. Cross-exchange & funding_current details ----
    print()
    print("--- cross_exchange / funding_current details ---")
    cx_dir = DATA_ROOT / "cross_exchange"
    if cx_dir.exists():
        print(f"\ncross_exchange:")
        for subdir in sorted(cx_dir.iterdir())[:10]:
            if subdir.is_dir():
                files = list(subdir.rglob("*.parquet"))
                if files:
                    rel = subdir.relative_to(DATA_ROOT)
                    print(f"  {rel}: {len(files)} files")
                    sample = files[0]
                    try:
                        df = pd.read_parquet(sample)
                        print(f"    sample cols: {list(df.columns)[:8]}")
                    except Exception:
                        pass

    fc_dir = DATA_ROOT / "funding_current" / "combined"
    if fc_dir.exists():
        print(f"\nfunding_current/combined:")
        for sym_dir in sorted(fc_dir.iterdir())[:10]:
            if sym_dir.is_dir():
                files = list(sym_dir.glob("*.parquet"))
                if files:
                    print(f"  {sym_dir.name}: {len(files)} files ({files[0].stem} ~ {sorted(f.stem for f in files)[-1]})")
                    sample = files[0]
                    try:
                        df = pd.read_parquet(sample)
                        print(f"    cols: {list(df.columns)}")
                    except Exception:
                        pass

    # ---- 4. Tardis ----
    print()
    print("--- Tardis downloaded data ---")
    tardis_summary = {}
    if TARDIS_ROOT.exists():
        try:
            top_dirs = list(TARDIS_ROOT.iterdir())
            print(f"\n{TARDIS_ROOT}: {len(top_dirs)} top-level entries")
            for top in sorted(top_dirs)[:20]:
                if top.is_dir():
                    sym_dirs = []
                    for root, dirs, _ in os.walk(top):
                        for d in dirs:
                            if d.endswith("USDT") or d.endswith("USD"):
                                sym_dirs.append(os.path.join(root, d))
                                break
                        if len(sym_dirs) > 5: break
                    print(f"  {top.name}: {len(sym_dirs)} symbol dirs (sample): {[os.path.basename(s) for s in sym_dirs[:3]]}")
                    tardis_summary[top.name] = sym_dirs[:10]
        except Exception as e:
            print(f"  Tardis access failed: {e}")
    else:
        print(f"  Tardis path not accessible: {TARDIS_ROOT}")

    # ---- 5. Capability flags + matrix ----
    print()
    print("=" * 80)
    print("POSSIBLE EXPERIMENTS MATRIX")
    print("=" * 80)

    has = {
        "eth_ob_self": any("orderbook / bybit / ETHUSDT" in k for k in available_combos),
        "btc_ob_self": any("orderbook / bybit / BTCUSDT" in k for k in available_combos),
        "sol_ob_self": any("orderbook / bybit / SOLUSDT" in k for k in available_combos),
        "btc_trades_self": any("trades / bybit / BTCUSDT" in k for k in available_combos),
        "sol_trades_self": any("trades / bybit / SOLUSDT" in k for k in available_combos),
        "binance_self": any("/ binance /" in k for k in available_combos),
        "okx_self": any("/ okx /" in k for k in available_combos),
        "btc_ob_tardis": any("orderbook / bybit_tardis / BTCUSDT" in k for k in available_combos),
        "sol_ob_tardis": any("orderbook / bybit_tardis / SOLUSDT" in k for k in available_combos),
    }

    print(f"\n자체 collector capability:")
    print(f"  ETH orderbook (self): {'✅' if has['eth_ob_self'] else '❌'}")
    print(f"  BTC orderbook (self): {'✅' if has['btc_ob_self'] else '❌'}")
    print(f"  SOL orderbook (self): {'✅' if has['sol_ob_self'] else '❌'}")
    print(f"  BTC trades (self): {'✅' if has['btc_trades_self'] else '❌'}")
    print(f"  Binance (self): {'✅' if has['binance_self'] else '❌'}")
    print(f"  OKX (self): {'✅' if has['okx_self'] else '❌'}")
    print(f"\nTardis capability:")
    print(f"  BTC orderbook (Tardis): {'✅' if has['btc_ob_tardis'] else '❌'}")
    print(f"  SOL orderbook (Tardis): {'✅' if has['sol_ob_tardis'] else '❌'}")

    print()
    print("실험 매트릭스:")
    print(f"\n{'Strategy':<30} {'Asset':<12} {'Timeframe':<12} {'Data ready':<25} {'Effort':<14}")
    print("-" * 100)

    matrix = []
    matrix.append(("Direction prediction (LR/XGB)", "ETH", "1H", "✅ self+tardis", "DONE (sido 17~29h)"))
    matrix.append(("Direction prediction", "ETH", "15min", "✅ target redo", "4-8h"))
    matrix.append(("Direction prediction", "ETH", "5min", "✅ target redo", "4-8h"))
    matrix.append(("Direction prediction", "BTC", "1H",
                   "✅ self+tardis" if has["btc_ob_self"] and has["btc_ob_tardis"] else
                   "🟡 tardis only" if has["btc_ob_tardis"] else
                   "🟡 self only" if has["btc_ob_self"] else "❌ collect needed",
                   "4-8h" if (has["btc_ob_self"] or has["btc_ob_tardis"]) else "1w + 4-8h"))
    matrix.append(("Direction prediction", "SOL", "1H",
                   "✅ self+tardis" if has["sol_ob_self"] and has["sol_ob_tardis"] else
                   "🟡 partial" if (has["sol_ob_self"] or has["sol_ob_tardis"]) else "❌",
                   "4-8h" if (has["sol_ob_self"] or has["sol_ob_tardis"]) else "1w + 4-8h"))
    matrix.append(("Market making", "ETH", "1min", "✅ self OB", "8-16h"))
    matrix.append(("Market making", "BTC", "1min",
                   "✅" if has["btc_ob_self"] else "❌", "8-16h" if has["btc_ob_self"] else "1w+"))
    matrix.append(("Pair trading (ETH-BTC)", "ETH-BTC", "1H",
                   "✅" if has["btc_ob_self"] else "❌", "8-16h" if has["btc_ob_self"] else "1w+"))
    matrix.append(("Cross-exch arb", "ETH", "tick",
                   "✅ ?" if has["binance_self"] else "❌", "16-24h" if has["binance_self"] else "infra+"))
    matrix.append(("Funding harvest", "ETH", "8h", "✅ DONE (시도 30, 음수)", "DONE"))
    matrix.append(("Funding harvest", "Multi-asset", "8h",
                   "🟡 funding 데이터 limited", "8h scan"))

    for strat, asset, tf, data, effort in matrix:
        print(f"{strat:<30} {asset:<12} {tf:<12} {data:<25} {effort:<14}")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print()
    if has["btc_ob_self"] and has["btc_ob_tardis"]:
        print("[1순위] ✅ BTC self + Tardis 모두 가용")
        print("        → 시도 31: BTC 1H direction (시도 17 framework 재사용)")
        print("        → ETH 한계가 본질인지 ETH-specific인지 결정적 진단")
        print("        → Effort 4-8h")
    elif has["btc_ob_tardis"]:
        print("[1순위] 🟡 BTC Tardis만 가용 (self 미수집)")
        print("        → 시도 31a: Tardis-only BTC 1H direction (Tardis test set 만 backtest)")
        print("        → Self test sample 없으나 train ceiling 측정 가능")
    elif has["btc_ob_self"]:
        print("[1순위] 🟡 BTC self만 가용 (Tardis 미다운로드)")
        print("        → 시도 31b: Self-only BTC backtest (작은 sample)")
    else:
        print("[1순위] ❌ BTC 데이터 미수집 → BTC collector 시작 후 1주 대기")

    if has["sol_ob_self"] or has["sol_ob_tardis"]:
        print("\n[2순위] ✅ SOL 데이터 가용 → 시도 32 (SOL 1H)")

    if has["binance_self"]:
        print("\n[3순위] ✅ Binance self 데이터 가용")
        print("        → 시도 33: ETH cross-exchange (Bybit vs Binance) spread analysis")

    print()
    print("[빠른 검증, 데이터 그대로]")
    print("  시도 27 (timeframe redo): 15min/5min target. data_prep.py target N 변경. 4-8h.")
    print("  시도 28 (asset switch): BTC 데이터 위 결과에 따라.")
    print()
    print("[지속 권장]")
    print("  현재 5 PID collector 유지 → 매주 ETH self data 확장 → 30+ days walk-forward 가능")

    # ---- 6. Save ----
    out = {
        "available_combos": available_combos,
        "has": has,
        "tardis_root_accessible": TARDIS_ROOT.exists(),
        "tardis_summary": tardis_summary,
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/audit07_data_inventory.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")
    log.info("Audit 07 complete")


if __name__ == "__main__":
    main()
