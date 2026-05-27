"""
Market Making backtest on mark19_data (1Hz orderbook) + trades_perp.

Based on sido28b_strict_market_making.py — but:
  - Path-fixed for this environment (~/mark19_data, no live_bot dependency).
  - Configurable fee scenarios: maker rebate (-2.5bp), maker standard (+2bp),
    taker fallback (+5.5bp) per side.
  - Parallel across days (multiprocessing.Pool).
  - Designed for 1198 ETH days = large-n honest verification.

Realistic constraints kept from sido28b:
  - Queue depletion factor 0.5 (only 50% of incoming volume hits our queue)
  - Cancel rate cap 600/min (Bybit retail)
  - 1-sec latency after cancel/place
  - Adverse selection tracked (60s lookahead, 0.05% drift = toxic)

CLI:
  python mm_backtest.py --symbols ETHUSDT --spread-bps 0,0.5,1 --sizes 0.01 \
    --maker-rate-bp -2.5 --workers 4
"""
from __future__ import annotations
import argparse, json, logging, re, sys, time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OB_ROOT = Path("/Users/mark/mark19_data")
DEFAULT_TR_ROOT = Path("/Users/mark/mark19_data/trades_perp")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")

# Strict realism params
CANCEL_RATE_PER_MIN_CAP = 600
LATENCY_SEC = 1
QUEUE_DEPLETION_FACTOR = 0.5
ADVERSE_DRIFT_THRESHOLD = 0.0005  # 0.05%
ADVERSE_LOOKAHEAD_SEC = 60


def simulate_day(ob_path: Path, tr_path: Path, date_str: str,
                  spread_bp: float, size_eth: float,
                  maker_rate: float, taker_rate: float) -> dict | None:
    """One-day MM sim. Rates as fractions (e.g. maker -0.00025 = -2.5bp rebate)."""
    if not ob_path.exists() or not tr_path.exists():
        return None
    try:
        ob = pd.read_parquet(ob_path, columns=["timestamp", "bid_0_price", "bid_0_size",
                                                "ask_0_price", "ask_0_size"])
        tr = pd.read_parquet(tr_path, columns=["timestamp", "side", "size", "price"])
    except Exception as e:
        return {"date": date_str, "error": str(e)}

    ob["timestamp"] = pd.to_datetime(ob["timestamp"], utc=True)
    # tr timestamp is epoch seconds float (Bybit perp archive)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], unit="s", utc=True)

    ob = ob.set_index("timestamp").sort_index()
    ob_1s = ob[["bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size"]].resample("1s").last().ffill()

    tr = tr.set_index("timestamp").sort_index()
    tr["sec"] = tr.index.floor("1s")
    tr_g = tr.groupby(["sec", "side"]).agg(total_size=("size", "sum"),
                                            max_price=("price", "max"),
                                            min_price=("price", "min")).reset_index()
    buy_g = tr_g[tr_g["side"] == "Buy"].set_index("sec")
    sell_g = tr_g[tr_g["side"] == "Sell"].set_index("sec")
    buy_g = buy_g.rename(columns={"total_size":"buy_size","max_price":"buy_max_p","min_price":"buy_min_p"})
    sell_g = sell_g.rename(columns={"total_size":"sell_size","max_price":"sell_max_p","min_price":"sell_min_p"})
    df = ob_1s.join(buy_g[["buy_size","buy_max_p","buy_min_p"]], how="left")
    df = df.join(sell_g[["sell_size","sell_max_p","sell_min_p"]], how="left")
    df[["buy_size","sell_size"]] = df[["buy_size","sell_size"]].fillna(0)
    df["mid"] = (df["bid_0_price"] + df["ask_0_price"]) / 2

    df_index = df.index
    df_arr = df[["bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size",
                 "mid", "buy_size", "sell_size", "buy_max_p", "sell_min_p"]].values

    inventory = 0.0
    bid_state = None; ask_state = None
    fills = []
    cash = 0.0; fee_paid = 0.0
    cancel_log = []; cancel_rate_limited_count = 0
    cooldown_bid = None; cooldown_ask = None
    spread_factor = spread_bp / 10000.0
    MAX_INVENTORY = max(0.5, size_eth * 100)

    for i in range(len(df)):
        bb, bsz, ba, asz, mid, b_vol, s_vol, b_maxp, s_minp = df_arr[i]
        if pd.isna(bb) or pd.isna(ba): continue

        if cooldown_bid is not None and i >= cooldown_bid:
            cooldown_bid = None
        if cooldown_ask is not None and i >= cooldown_ask:
            cooldown_ask = None

        target_bid = bb * (1 - spread_factor)
        target_ask = ba * (1 + spread_factor)

        # Cancel if drifted
        if bid_state is not None and abs(bid_state["price"] - target_bid) / target_bid > 0.0001:
            cancel_log.append(i); bid_state = None; cooldown_bid = i + LATENCY_SEC
        if ask_state is not None and abs(ask_state["price"] - target_ask) / target_ask > 0.0001:
            cancel_log.append(i); ask_state = None; cooldown_ask = i + LATENCY_SEC

        # Process queue depletion → fill
        if bid_state is not None and s_vol > 0 and not pd.isna(s_minp) and s_minp <= bid_state["price"] + 1e-9:
            bid_state["queue"] -= s_vol * QUEUE_DEPLETION_FACTOR
            if bid_state["queue"] <= 0:
                fp = bid_state["price"]
                fee = size_eth * fp * maker_rate
                cash += -size_eth * fp - fee
                fee_paid += fee
                inventory += size_eth
                fills.append({"side":"bid","price":fp,"mid":mid,"i":i})
                bid_state = None; cooldown_bid = i + LATENCY_SEC

        if ask_state is not None and b_vol > 0 and not pd.isna(b_maxp) and b_maxp >= ask_state["price"] - 1e-9:
            ask_state["queue"] -= b_vol * QUEUE_DEPLETION_FACTOR
            if ask_state["queue"] <= 0:
                fp = ask_state["price"]
                fee = size_eth * fp * maker_rate
                cash += size_eth * fp - fee
                fee_paid += fee
                inventory -= size_eth
                fills.append({"side":"ask","price":fp,"mid":mid,"i":i})
                ask_state = None; cooldown_ask = i + LATENCY_SEC

        # Place new
        recent_cancels = sum(1 for c in cancel_log[-CANCEL_RATE_PER_MIN_CAP-1:] if c > i - 60)
        rate_limited = recent_cancels >= CANCEL_RATE_PER_MIN_CAP
        if (bid_state is None and inventory < MAX_INVENTORY
            and (cooldown_bid is None or i >= cooldown_bid) and not rate_limited):
            initial_q = float(bsz) if spread_bp <= 0 else float(bsz) * 0.5
            bid_state = {"price": target_bid, "queue": initial_q}
        if (ask_state is None and inventory > -MAX_INVENTORY
            and (cooldown_ask is None or i >= cooldown_ask) and not rate_limited):
            initial_q = float(asz) if spread_bp <= 0 else float(asz) * 0.5
            ask_state = {"price": target_ask, "queue": initial_q}
        if rate_limited:
            cancel_rate_limited_count += 1

    # EoD flatten at taker
    if abs(inventory) > 1e-9 and len(df) > 0:
        last = df_arr[-1]
        if inventory > 0:
            fp = last[0]; fee = inventory * fp * taker_rate
            cash += inventory * fp - fee; fee_paid += fee
        else:
            fp = last[2]; fee = abs(inventory) * fp * taker_rate
            cash += -abs(inventory) * fp - fee; fee_paid += fee
        inventory = 0

    # Adverse selection
    n_toxic = n_favorable = 0
    for f in fills:
        future_i = f["i"] + ADVERSE_LOOKAHEAD_SEC
        if future_i < len(df):
            future_mid = df_arr[future_i][4]
            if not pd.isna(future_mid):
                drift = (future_mid - f["mid"]) / f["mid"]
                if f["side"] == "bid":
                    if drift < -ADVERSE_DRIFT_THRESHOLD: n_toxic += 1
                    elif drift > ADVERSE_DRIFT_THRESHOLD: n_favorable += 1
                else:
                    if drift > ADVERSE_DRIFT_THRESHOLD: n_toxic += 1
                    elif drift < -ADVERSE_DRIFT_THRESHOLD: n_favorable += 1

    avg_mid = float(np.nanmean(df_arr[:, 4]))
    capital = MAX_INVENTORY * avg_mid
    return {
        "date": date_str, "spread_bp": spread_bp, "size_eth": size_eth,
        "pnl_usd": float(cash),
        "pct_per_cap": float(cash / capital * 100) if capital > 0 else 0,
        "fee_total": float(fee_paid),
        "n_fills": int(len(fills)),
        "n_pairs": int(min(sum(1 for f in fills if f["side"]=="bid"),
                            sum(1 for f in fills if f["side"]=="ask"))),
        "n_cancels": int(len(cancel_log)),
        "rate_limited_secs": int(cancel_rate_limited_count),
        "inventory_end": float(inventory),
        "avg_mid": avg_mid, "capital_used": float(capital),
        "n_toxic": int(n_toxic), "n_favorable": int(n_favorable),
        "toxic_rate": float(n_toxic / max(n_toxic + n_favorable, 1)),
        "n_seconds": int(len(df)),
    }


def worker(task):
    sym, ds, ob_path, tr_path, spread_bp, size_eth, mkr, tkr = task
    t0 = time.time()
    try:
        r = simulate_day(ob_path, tr_path, ds, spread_bp, size_eth, mkr, tkr)
        if r is None:
            return (sym, ds, time.time()-t0, None, "missing")
        return (sym, ds, time.time()-t0, r, "ok")
    except Exception as e:
        return (sym, ds, time.time()-t0, None, f"FAIL: {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--ob-root", type=Path, default=DEFAULT_OB_ROOT)
    p.add_argument("--tr-root", type=Path, default=DEFAULT_TR_ROOT)
    p.add_argument("--spread-bps", default="0,0.5,1,2", help="comma-separated bp offsets")
    p.add_argument("--sizes", default="0.01", help="comma-separated ETH sizes")
    p.add_argument("--maker-rate-bp", type=float, default=-2.5,
                   help="maker rebate/fee in bp/side (negative = rebate, +2 = standard maker)")
    p.add_argument("--taker-rate-bp", type=float, default=5.5)
    p.add_argument("--start", default=None); p.add_argument("--end", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("/Users/mark/mark19_data/mm_results.json"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger()

    spread_bps = [float(x) for x in args.spread_bps.split(",")]
    sizes = [float(x) for x in args.sizes.split(",")]
    mkr = args.maker_rate_bp / 10000
    tkr = args.taker_rate_bp / 10000

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tasks = []
    for sym in symbols:
        ob_dir = args.ob_root / sym
        tr_dir = args.tr_root / sym
        if not ob_dir.exists() or not tr_dir.exists(): continue
        for f in sorted(ob_dir.iterdir()):
            m = DATE_RE.match(f.name)
            if not m: continue
            ds = m.group(1)
            if args.start and ds < args.start: continue
            if args.end and ds > args.end: continue
            tr_path = tr_dir / f.name
            if not tr_path.exists(): continue
            for sb in spread_bps:
                for sz in sizes:
                    tasks.append((sym, ds, f, tr_path, sb, sz, mkr, tkr))
    if args.limit > 0:
        tasks = tasks[:args.limit]
    log.info(f"Discovered {len(tasks)} day-combo tasks. spreads={spread_bps} sizes={sizes} maker={args.maker_rate_bp}bp taker={args.taker_rate_bp}bp")
    if not tasks: return

    t0 = time.time(); done = ok = failed = 0
    all_results = []
    with Pool(processes=args.workers, maxtasksperchild=20) as pool:
        for sym, ds, dt, r, status in pool.imap_unordered(worker, tasks):
            done += 1
            if status == "ok":
                ok += 1; all_results.append(r)
            else:
                failed += 1
                if failed <= 5: log.warning(f"  {ds} {status}")
            if done % 200 == 0 or done == len(tasks):
                rate = done / (time.time()-t0)
                log.info(f"  [{done}/{len(tasks)}] ok={ok} fail={failed} | rate {rate*60:.0f}/min | ETA {(len(tasks)-done)/rate/60:.1f}min")

    log.info(f"DONE. ok={ok}/{len(tasks)} | elapsed {(time.time()-t0)/60:.1f}min")

    # Aggregate by combo
    rdf = pd.DataFrame(all_results)
    print()
    print("=" * 120)
    print("MM BACKTEST SUMMARY")
    print(f"  maker_rate={args.maker_rate_bp:+.2f}bp/side   taker_rate={args.taker_rate_bp:+.2f}bp/side")
    print("=" * 120)
    print(f"{'spread_bp':<10}{'size':<8}{'days':<6}{'mean_pct':<12}{'std_pct':<10}{'Sharpe':<10}{'pos%':<8}{'mean_fills':<12}{'toxic%':<8}")
    print("-"*120)
    for (sb, sz), g in rdf.groupby(["spread_bp","size_eth"]):
        m_pct = g["pct_per_cap"].mean(); s_pct = g["pct_per_cap"].std()
        sh = m_pct/s_pct*np.sqrt(365) if s_pct>0 else 0
        from scipy import stats as st
        t,pv = st.ttest_1samp(g["pct_per_cap"].values, 0)
        print(f"{sb:<10.2f}{sz:<8.3f}{len(g):<6}{m_pct:<+12.4f}{s_pct:<10.4f}{sh:<+10.2f}{(g['pct_per_cap']>0).mean()*100:<8.1f}{g['n_fills'].mean():<12.0f}{g['toxic_rate'].mean()*100:<8.1f} (t={t:+.2f} p={pv:.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args":vars(args), "results":all_results}, default=str, indent=1))
    log.info(f"Saved results → {args.out}")


if __name__ == "__main__":
    main()
