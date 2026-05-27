"""시도 28b: Stricter MM backtest with realistic queue + cancel limit + latency + adverse selection.

Sweep: 4 spread offsets × 3 sizes = 12 combinations.
"""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

FEE_TAKER = 0.00055
FEE_MAKER = -0.00025

# Strict params
CANCEL_RATE_PER_MIN_CAP = 600   # Bybit retail cap (10/sec)
LATENCY_SEC = 1                 # 1-sec gap after cancel/replace (200-300ms in practice)
QUEUE_DEPLETION_FACTOR = 0.5    # only 50% of trade volume hits our queue (others go to better-priced makers)
ADVERSE_DRIFT_THRESHOLD = 0.0005  # 0.05% drift in 60s = toxic
ADVERSE_LOOKAHEAD_SEC = 60

DATES = [f"2026-04-{d:02d}" for d in range(21, 31)] + ["2026-05-01"]


def simulate_day(date_str, spread_bp, size_eth, log):
    """Run one day with given spread offset (bp from best) and size."""
    ob_path = Path(f"/Users/dohun/Desktop/Mark/mark19/data/orderbook/bybit/ETHUSDT/{date_str}.parquet")
    tr_path = Path(f"/Users/dohun/Desktop/Mark/mark19/data/trades/bybit/ETHUSDT/{date_str}.parquet")
    if not ob_path.exists() or not tr_path.exists():
        return None

    from live_bot.parquet_retry import read_parquet_with_retry
    try:
        ob = read_parquet_with_retry(ob_path, retries=10, wait_ms=500)
        tr = read_parquet_with_retry(tr_path, retries=10, wait_ms=500)
    except Exception as e:
        log.warning(f"  {date_str}: skipping after retries exhausted: {e}")
        return None
    ob["timestamp"] = pd.to_datetime(ob["timestamp"], utc=True)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    ob = ob.set_index("timestamp").sort_index()
    ob_1s = ob[["bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size"]].resample("1s").last().ffill()

    tr = tr.set_index("timestamp").sort_index()
    tr["sec"] = tr.index.floor("1s")
    tr_g = tr.groupby(["sec", "side"]).agg(total_size=("size", "sum"),
                                            max_price=("price", "max"),
                                            min_price=("price", "min")).reset_index()
    buy_g = tr_g[tr_g["side"] == "Buy"].set_index("sec")
    sell_g = tr_g[tr_g["side"] == "Sell"].set_index("sec")
    buy_g.columns = ["side_b", "buy_size", "buy_max_p", "buy_min_p"]
    sell_g.columns = ["side_s", "sell_size", "sell_max_p", "sell_min_p"]
    df = ob_1s.join(buy_g[["buy_size", "buy_max_p", "buy_min_p"]], how="left")
    df = df.join(sell_g[["sell_size", "sell_max_p", "sell_min_p"]], how="left")
    df[["buy_size", "sell_size"]] = df[["buy_size", "sell_size"]].fillna(0)
    df["mid"] = (df["bid_0_price"] + df["ask_0_price"]) / 2

    inventory = 0.0
    bid_state = None  # {price, queue, place_ts, cooldown_until}
    ask_state = None
    fills = []
    cash = 0.0
    fee_paid = 0.0

    cancel_log = []  # timestamps of cancels
    cancel_rate_limited_count = 0

    # Adverse selection: store fills with their fill_ts and entry_mid
    fills_for_adv = []  # (fill_ts, side, fill_price, entry_mid)

    df_index = df.index
    df_arr = df[["bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size",
                 "mid", "buy_size", "sell_size", "buy_max_p", "sell_min_p"]].values

    cooldown_bid = None
    cooldown_ask = None

    spread_factor = spread_bp / 10000.0  # bp to fraction
    MAX_INVENTORY = max(0.5, size_eth * 100)  # scale max inv with size

    for i in range(len(df)):
        sec_ts = df_index[i]
        bb, bsz, ba, asz, mid, b_vol, s_vol, b_maxp, s_minp = df_arr[i]
        if pd.isna(bb) or pd.isna(ba): continue

        # --- Latency cooldown check ---
        if cooldown_bid is not None and i < cooldown_bid: pass
        else:
            cooldown_bid = None
        if cooldown_ask is not None and i < cooldown_ask: pass
        else:
            cooldown_ask = None

        # --- 1. Cancel if level no longer at best ---
        # We placed at "best ± spread_offset". Spread = best - spread_factor (for bid better)
        # If best moves up significantly past us, cancel
        target_bid = bb * (1 - spread_factor)  # our bid goes BELOW current best by spread_bp
        target_ask = ba * (1 + spread_factor)  # our ask goes ABOVE current best by spread_bp
        # spread_bp=0 → at best; spread_bp>0 → outside best (passive)

        if bid_state is not None:
            # Cancel if our price too far from current best (>1bp away from target)
            if abs(bid_state["price"] - target_bid) / target_bid > 0.0001:
                # Need to cancel
                cancel_log.append(i)
                bid_state = None
                cooldown_bid = i + LATENCY_SEC

        if ask_state is not None:
            if abs(ask_state["price"] - target_ask) / target_ask > 0.0001:
                cancel_log.append(i)
                ask_state = None
                cooldown_ask = i + LATENCY_SEC

        # --- 2. Process trades to deplete queue (with QUEUE_DEPLETION_FACTOR) ---
        if bid_state is not None and s_vol > 0:
            # Only count sell trades whose price hit our level
            if not pd.isna(s_minp) and s_minp <= bid_state["price"] + 1e-9:
                bid_state["queue"] -= s_vol * QUEUE_DEPLETION_FACTOR
                if bid_state["queue"] <= 0:
                    fp = bid_state["price"]
                    fee = size_eth * fp * FEE_MAKER
                    cash += -size_eth * fp - fee
                    fee_paid += fee
                    inventory += size_eth
                    fills.append({"ts": sec_ts, "side": "bid", "price": fp,
                                  "size": size_eth, "inventory": inventory, "mid": mid, "i": i})
                    bid_state = None
                    cooldown_bid = i + LATENCY_SEC

        if ask_state is not None and b_vol > 0:
            if not pd.isna(b_maxp) and b_maxp >= ask_state["price"] - 1e-9:
                ask_state["queue"] -= b_vol * QUEUE_DEPLETION_FACTOR
                if ask_state["queue"] <= 0:
                    fp = ask_state["price"]
                    fee = size_eth * fp * FEE_MAKER
                    cash += size_eth * fp - fee
                    fee_paid += fee
                    inventory -= size_eth
                    fills.append({"ts": sec_ts, "side": "ask", "price": fp,
                                  "size": size_eth, "inventory": inventory, "mid": mid, "i": i})
                    ask_state = None
                    cooldown_ask = i + LATENCY_SEC

        # --- 3. Place new orders (respecting cooldown + rate limit) ---
        # Cancel rate check (per minute)
        # Count cancels in last 60 sec
        recent_cancels = sum(1 for c in cancel_log[-CANCEL_RATE_PER_MIN_CAP-1:] if c > i - 60)
        rate_limited = recent_cancels >= CANCEL_RATE_PER_MIN_CAP

        if (bid_state is None and inventory < MAX_INVENTORY
            and (cooldown_bid is None or i >= cooldown_bid) and not rate_limited):
            # Place bid at target_bid. Initial queue = bsz if at best, less if behind
            initial_q = float(bsz) if spread_bp <= 0 else float(bsz) * 0.5  # halved if outside
            bid_state = {"price": target_bid, "queue": initial_q, "place_i": i}

        if (ask_state is None and inventory > -MAX_INVENTORY
            and (cooldown_ask is None or i >= cooldown_ask) and not rate_limited):
            initial_q = float(asz) if spread_bp <= 0 else float(asz) * 0.5
            ask_state = {"price": target_ask, "queue": initial_q, "place_i": i}

        if rate_limited:
            cancel_rate_limited_count += 1

    # End of day: flatten inventory at best (taker)
    if abs(inventory) > 1e-9 and len(df) > 0:
        last_idx = len(df) - 1
        last = df_arr[last_idx]
        if inventory > 0:
            sell_p = last[0]  # best_bid
            fee = inventory * sell_p * FEE_TAKER
            cash += inventory * sell_p - fee
            fee_paid += fee
        else:
            buy_p = last[2]  # best_ask
            fee = abs(inventory) * buy_p * FEE_TAKER
            cash += -abs(inventory) * buy_p - fee
            fee_paid += fee
        inventory = 0

    # Adverse selection analysis
    n_toxic = 0
    n_favorable = 0
    for f in fills:
        if f["side"] not in ("bid", "ask"): continue
        future_i = f["i"] + ADVERSE_LOOKAHEAD_SEC
        if future_i < len(df):
            future_mid = df_arr[future_i][4]
            if not pd.isna(future_mid):
                drift = (future_mid - f["mid"]) / f["mid"]
                if f["side"] == "bid":
                    # Bought: drift < 0 = adverse (price dropped after buy)
                    if drift < -ADVERSE_DRIFT_THRESHOLD: n_toxic += 1
                    elif drift > ADVERSE_DRIFT_THRESHOLD: n_favorable += 1
                else:
                    # Sold: drift > 0 = adverse
                    if drift > ADVERSE_DRIFT_THRESHOLD: n_toxic += 1
                    elif drift < -ADVERSE_DRIFT_THRESHOLD: n_favorable += 1

    avg_mid = np.nanmean(df_arr[:, 4])
    return {
        "date": date_str,
        "spread_bp": spread_bp, "size_eth": size_eth,
        "pnl_usd": float(cash),
        "fee_total": float(fee_paid),
        "n_fills": int(len(fills)),
        "n_pairs": int(min(sum(1 for f in fills if f["side"] == "bid"),
                           sum(1 for f in fills if f["side"] == "ask"))),
        "n_cancels": int(len(cancel_log)),
        "rate_limited_secs": int(cancel_rate_limited_count),
        "inventory_end": float(inventory),
        "avg_mid": float(avg_mid),
        "n_toxic": int(n_toxic),
        "n_favorable": int(n_favorable),
        "toxic_rate": float(n_toxic / max(n_toxic + n_favorable, 1)),
        "n_seconds": int(len(df)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 28b: Stricter MM with queue/cancel/latency/adverse")
    log.info("=" * 70)

    spread_bps = [0, 0.5, 1, 2]   # 0bp = at best, 2bp = 2bp better than best
    sizes = [0.005, 0.01, 0.05]   # ETH
    log.info(f"\nSweep: {len(spread_bps)} spreads × {len(sizes)} sizes = {len(spread_bps)*len(sizes)} combos")
    log.info(f"Spread bps: {spread_bps}")
    log.info(f"Sizes ETH: {sizes}")
    log.info(f"Strict params:")
    log.info(f"  Queue depletion factor: {QUEUE_DEPLETION_FACTOR} (50% of trades hit our queue)")
    log.info(f"  Cancel rate cap: {CANCEL_RATE_PER_MIN_CAP}/min")
    log.info(f"  Latency: {LATENCY_SEC} sec")
    log.info(f"  Adverse drift threshold: {ADVERSE_DRIFT_THRESHOLD*100:.3f}% in {ADVERSE_LOOKAHEAD_SEC} sec")

    all_results = {}
    for spread_bp in spread_bps:
        for size_eth in sizes:
            key = f"sp{spread_bp}_sz{size_eth}"
            log.info(f"\n--- combo {key}: spread {spread_bp}bp, size {size_eth} ETH ---")
            day_results = []
            for d in DATES:
                r = simulate_day(d, spread_bp, size_eth, log)
                if r is None: continue
                day_results.append(r)
                # Compute capital
                avg_mid = r["avg_mid"]
                max_inv = max(0.5, size_eth * 100)
                capital = max_inv * avg_mid
                pct = r["pnl_usd"] / capital * 100 if capital > 0 else 0
                log.info(f"  {d}: pnl ${r['pnl_usd']:+.4f} ({pct:+.3f}%/cap) fills {r['n_fills']} pairs {r['n_pairs']} toxic {r['toxic_rate']*100:.1f}% rate-lim {r['rate_limited_secs']}s")
            all_results[key] = {
                "spread_bp": spread_bp, "size_eth": size_eth,
                "days": day_results,
            }

    # ---- Aggregate ----
    print()
    print("=" * 110)
    print("12-COMBO COMPARISON (11 days each)")
    print("=" * 110)
    print(f"{'Combo':<18} {'Total $':<12} {'Mean $':<10} {'Mean %':<10} {'Std %':<10} {'Sharpe':<10} {'Pos':<8} {'AvgFills':<10} {'Toxic%':<8} {'Cancel/d':<10}")
    print("-" * 120)

    summary = {}
    for key, data in all_results.items():
        days = data["days"]
        if not days: continue
        pnls_usd = np.array([d["pnl_usd"] for d in days])
        # capital per day
        caps = np.array([max(0.5, data["size_eth"] * 100) * d["avg_mid"] for d in days])
        pcts = pnls_usd / caps * 100
        n_pairs_avg = np.mean([d["n_pairs"] for d in days])
        n_cancels_avg = np.mean([d["n_cancels"] for d in days])
        toxic_rate_avg = np.mean([d["toxic_rate"] for d in days])
        rate_limited_total = sum(d["rate_limited_secs"] for d in days)
        positive = (pnls_usd > 0).sum()
        sharpe = pcts.mean() / pcts.std() * np.sqrt(365) if pcts.std() > 0 else 0
        print(f"{key:<18} {pnls_usd.sum():<+12.2f} {pnls_usd.mean():<+10.3f} {pcts.mean():<+10.3f} {pcts.std():<10.3f} {sharpe:<10.2f} {positive}/{len(days):<6} {n_pairs_avg:<10.0f} {toxic_rate_avg*100:<8.1f} {n_cancels_avg:<10.0f}")
        summary[key] = {
            "spread_bp": data["spread_bp"], "size_eth": data["size_eth"],
            "total_pnl": float(pnls_usd.sum()),
            "mean_pnl": float(pnls_usd.mean()),
            "mean_pct": float(pcts.mean()),
            "std_pct": float(pcts.std()),
            "sharpe_annualized": float(sharpe),
            "positive_days": int(positive),
            "total_days": len(days),
            "avg_pairs_per_day": float(n_pairs_avg),
            "avg_cancels_per_day": float(n_cancels_avg),
            "avg_toxic_rate": float(toxic_rate_avg),
            "total_rate_limited_secs": int(rate_limited_total),
        }

    # ---- Best combo ----
    best_key = max(summary.keys(), key=lambda k: summary[k]["mean_pct"])
    best = summary[best_key]
    log.info(f"\nBest combo by mean%: {best_key}")
    log.info(f"  Total $: {best['total_pnl']:+.2f}")
    log.info(f"  Mean: {best['mean_pct']:+.3f}%/cap, Std {best['std_pct']:.3f}, Sharpe {best['sharpe_annualized']:.2f}")
    log.info(f"  Positive: {best['positive_days']}/{best['total_days']}")
    log.info(f"  Toxic rate: {best['avg_toxic_rate']*100:.1f}%, Cancels/day: {best['avg_cancels_per_day']:.0f}")

    # ---- Compare with 시도 28 ----
    print()
    print("=" * 80)
    print("VS 시도 28 (idealized 100% maker)")
    print("=" * 80)
    print(f"\n시도 28 (idealized): Mean +2.566%/cap, Std 2.06, Sharpe 23.80, Pos 10/11")
    print(f"시도 28b best ({best_key}): Mean {best['mean_pct']:+.3f}%/cap, Std {best['std_pct']:.3f}, Sharpe {best['sharpe_annualized']:.2f}, Pos {best['positive_days']}/{best['total_days']}")
    print(f"  Adjustment factor: {best['mean_pct']/2.566:.2f}× (시도 28 → 시도 28b realistic)")

    out = {
        "approach": "Strict MM with realistic queue + cancel cap + latency + adverse selection",
        "params": {
            "queue_depletion_factor": QUEUE_DEPLETION_FACTOR,
            "cancel_rate_per_min_cap": CANCEL_RATE_PER_MIN_CAP,
            "latency_sec": LATENCY_SEC,
            "adverse_drift_threshold": ADVERSE_DRIFT_THRESHOLD,
            "fee_maker": FEE_MAKER, "fee_taker": FEE_TAKER,
        },
        "summary": summary,
        "best_combo": best_key,
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido28b_strict_market_making.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if best["mean_pct"] > 0.5 and best["positive_days"] / best["total_days"] > 0.7:
        print(f"\n  STRONG: Best combo +{best['mean_pct']:.3f}%/cap, {best['positive_days']}/{best['total_days']} positive")
        print(f"  → SHADOW LIVE 검증 권장")
    elif best["mean_pct"] > 0.0:
        print(f"\n  MARGINAL: Best combo {best['mean_pct']:+.3f}%/cap, {best['positive_days']}/{best['total_days']}")
        print(f"  → Edge 좁음, SHADOW LIVE에서 진짜 측정 필요")
    else:
        print(f"\n  NEGATIVE: Best combo {best['mean_pct']:+.3f}%/cap")
        print(f"  → MM 단순 strategy로 부적합. Direction model + MM 결합 검토.")

    log.info("\n시도 28b complete")


if __name__ == "__main__":
    main()
