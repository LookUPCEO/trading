"""LIVE 24h log analysis."""
import re, sys, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


LOG = Path("/Users/dohun/Desktop/Mark/mark19/logs/live_small_20260430_2305.log")


def parse_ts(line):
    m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return None


def main():
    if not LOG.exists():
        print(f"Log not found: {LOG}")
        return

    with open(LOG) as f:
        lines = f.readlines()
    print(f"Lines: {len(lines)}")
    print(f"First: {lines[0].strip()[:120]}")
    print(f"Last:  {lines[-1].strip()[:120]}")

    # Extract events
    trades = []  # state machine: open → exit attempts → close
    current = None  # current trade dict
    wallet_history = []
    reconcile_events = []
    sl_events = []
    market_orders = []
    limit_orders = []
    fill_events = []  # if any
    daily_resets = []
    errors = []

    re_wallet = re.compile(r"_calc_qty: wallet=\$([\d.]+).*notional=\$([\d.]+) → qty=([\d.]+)")
    re_market = re.compile(r"Submitted MARKET (long|short) qty=([\d.]+) order_id=(\d+)")
    re_limit = re.compile(r"Submitted LIMIT (long|short) qty=([\d.]+) @ \$([\d.]+) order_id=(\d+)")
    re_reconcile = re.compile(r"RECONCILE: (.*)")
    re_filled = re.compile(r"(filled|FILLED|exit_filled)", re.IGNORECASE)
    re_sl = re.compile(r"(SL|stop[\s_]?loss|stoploss|trading-stop|trading_stop)", re.IGNORECASE)
    re_emergency = re.compile(r"(EMERGENCY|emergency)", re.IGNORECASE)
    re_error = re.compile(r"(ERROR|Error|exception|Exception)")
    re_drift = re.compile(r"_check_exit drift: t=([\d.]+)min")
    re_daily_reset = re.compile(r"DAILY|daily_reset")

    for line in lines:
        ts = parse_ts(line)

        m = re_wallet.search(line)
        if m and ts:
            wallet_history.append({"ts": ts, "wallet": float(m.group(1)),
                                    "notional": float(m.group(2)), "qty": float(m.group(3))})

        m = re_market.search(line)
        if m and ts:
            market_orders.append({"ts": ts, "side": m.group(1), "qty": float(m.group(2)),
                                  "order_id": m.group(3)})

        m = re_limit.search(line)
        if m and ts:
            limit_orders.append({"ts": ts, "side": m.group(1), "qty": float(m.group(2)),
                                 "price": float(m.group(3)), "order_id": m.group(4)})

        m = re_reconcile.search(line)
        if m and ts:
            reconcile_events.append({"ts": ts, "msg": m.group(1).strip()})

        if re_sl.search(line) and ts:
            sl_events.append({"ts": ts, "line": line.strip()[:200]})

        if re_emergency.search(line) and ts:
            errors.append({"ts": ts, "type": "emergency", "line": line.strip()[:200]})

        if re_error.search(line) and ts:
            errors.append({"ts": ts, "type": "error", "line": line.strip()[:200]})

    # Group market orders into trades (each MARKET open is a new trade)
    trades = []
    for mo in market_orders:
        trades.append({
            "open_ts": mo["ts"], "open_side": mo["side"], "qty": mo["qty"],
            "open_order_id": mo["order_id"],
            "exit_attempts": [], "reconcile": None, "duration_min": None,
        })

    # Attach limit exits and reconciles to trades by time window
    for limit in limit_orders:
        # find most recent trade whose open_ts < limit.ts
        for t in reversed(trades):
            if t["open_ts"] < limit["ts"]:
                t["exit_attempts"].append({
                    "ts": limit["ts"], "side": limit["side"], "price": limit["price"],
                })
                break

    for r in reconcile_events:
        for t in reversed(trades):
            if t["open_ts"] < r["ts"] and t["reconcile"] is None:
                t["reconcile"] = {"ts": r["ts"], "msg": r["msg"]}
                t["close_ts"] = r["ts"]
                t["duration_min"] = (r["ts"] - t["open_ts"]).total_seconds() / 60.0
                break

    print()
    print("=" * 80)
    print("LIVE 24h ANALYSIS")
    print("=" * 80)
    if wallet_history:
        first_w = wallet_history[0]["wallet"]
        last_w = wallet_history[-1]["wallet"]
        wallet_diff = last_w - first_w
        wallet_pct = (last_w - first_w) / first_w * 100
        print(f"\nSession: {wallet_history[0]['ts']} → {wallet_history[-1]['ts']}")
        print(f"Duration: {(wallet_history[-1]['ts'] - wallet_history[0]['ts']).total_seconds() / 3600:.1f} hours")
        print(f"\nWallet: ${first_w:.2f} → ${last_w:.2f}  ({wallet_diff:+.2f}, {wallet_pct:+.3f}%)")
        # Min, max
        ws = [w["wallet"] for w in wallet_history]
        print(f"Wallet range: ${min(ws):.2f} ~ ${max(ws):.2f}")

    print(f"\nMarket orders (entries): {len(market_orders)}")
    print(f"Limit orders (exit attempts): {len(limit_orders)}")
    print(f"Reconcile events: {len(reconcile_events)}")
    print(f"SL-related lines: {len(sl_events)}")
    print(f"Errors/exceptions: {len(errors)}")

    # Trades summary
    print()
    print("=" * 80)
    print("TRADE LIST")
    print("=" * 80)
    closed = [t for t in trades if t["reconcile"]]
    open_unclosed = [t for t in trades if not t["reconcile"]]
    print(f"\nClosed trades: {len(closed)}")
    print(f"Unclosed (open at session end): {len(open_unclosed)}")

    print(f"\n{'#':<4} {'Open ts':<22} {'Side':<6} {'Qty':<6} {'Exits':<7} {'Duration':<11} {'Close reason':<60}")
    print("-" * 120)
    for i, t in enumerate(trades):
        dur = f"{t['duration_min']:.1f}min" if t.get("duration_min") else "OPEN"
        reason = t["reconcile"]["msg"][:58] if t["reconcile"] else "(still open at session end)"
        print(f"{i+1:<4} {t['open_ts'].strftime('%Y-%m-%d %H:%M:%S'):<22} {t['open_side']:<6} {t['qty']:<6.2f} {len(t['exit_attempts']):<7} {dur:<11} {reason:<60}")

    # Closed trade duration distribution
    if closed:
        durs = [t["duration_min"] for t in closed]
        print()
        print(f"Closed trade duration: min {min(durs):.1f}  median {sorted(durs)[len(durs)//2]:.1f}  max {max(durs):.1f}  mean {sum(durs)/len(durs):.1f} min")

    # Reconcile messages
    print()
    print("=" * 80)
    print("RECONCILE BREAKDOWN")
    print("=" * 80)
    msg_counts = defaultdict(int)
    for r in reconcile_events:
        # Bucket by first 80 chars
        msg_counts[r["msg"][:80]] += 1
    for msg, count in sorted(msg_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}× {msg}")

    # SL events
    print()
    print("=" * 80)
    print("SL/STOP-LOSS LINES (sample)")
    print("=" * 80)
    for sl in sl_events[:5]:
        print(f"  {sl['ts']}  {sl['line'][:140]}")
    if len(sl_events) > 5:
        print(f"  ... and {len(sl_events) - 5} more")

    # Errors
    if errors:
        print()
        print("=" * 80)
        print("ERRORS/EXCEPTIONS")
        print("=" * 80)
        for e in errors[:10]:
            print(f"  {e['ts']}  [{e['type']}]  {e['line'][:140]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Per-trade exit attempts (drift policy stats)
    n_attempts_total = len(limit_orders)
    closed_trades_with_attempts = [t for t in closed if len(t["exit_attempts"]) > 0]
    avg_attempts = sum(len(t["exit_attempts"]) for t in closed_trades_with_attempts) / max(len(closed_trades_with_attempts), 1)
    max_attempts = max((len(t["exit_attempts"]) for t in closed_trades_with_attempts), default=0)

    print()
    print("=" * 80)
    print("DRIFT POLICY STATS")
    print("=" * 80)
    print(f"  Total LIMIT order replacements: {n_attempts_total}")
    print(f"  Average per closed trade: {avg_attempts:.1f}")
    print(f"  Max for single trade: {max_attempts}")
    if open_unclosed:
        last_open = open_unclosed[-1]
        print(f"  Last unclosed trade had {len(last_open['exit_attempts'])} replacements (still attempting at session end)")

    # Save
    out = {
        "session": {
            "start": str(wallet_history[0]["ts"]) if wallet_history else None,
            "end": str(wallet_history[-1]["ts"]) if wallet_history else None,
            "wallet_first": wallet_history[0]["wallet"] if wallet_history else None,
            "wallet_last": wallet_history[-1]["wallet"] if wallet_history else None,
            "wallet_min": min(ws) if wallet_history else None,
            "wallet_max": max(ws) if wallet_history else None,
        },
        "trade_count_total": len(trades),
        "trade_count_closed": len(closed),
        "trade_count_open_at_end": len(open_unclosed),
        "limit_order_replacements_total": n_attempts_total,
        "reconcile_events": len(reconcile_events),
        "sl_lines": len(sl_events),
        "errors": len(errors),
        "trades": [{
            "open_ts": str(t["open_ts"]),
            "side": t["open_side"], "qty": t["qty"],
            "exits": len(t["exit_attempts"]),
            "duration_min": t.get("duration_min"),
            "close_reason": t["reconcile"]["msg"][:120] if t["reconcile"] else None,
        } for t in trades],
        "wallet_history": [{
            "ts": str(w["ts"]), "wallet": w["wallet"],
        } for w in wallet_history],
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/live_24h_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON saved: {out_path}")


if __name__ == "__main__":
    main()
