"""
Trade dashboard — Discord status snapshots.

Two modes:
  shadow → 📈 virtual trading, explicit "실거래 아님" label
  live   → 💰 real trading from Bybit V5 (position + execution + closed-pnl)

Functions:
  shadow_snapshot(runner_state) → dict for Discord
  live_snapshot()               → dict for Discord (calls Bybit V5)
  send_snapshot(mode, ...)      → sends to Discord
  on_trade_event(event, payload)→ immediate alert for entry/exit
"""
from __future__ import annotations
import json, logging, os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import discord_notify as dn
except Exception:
    dn = None

import bybit_v5


SYMBOL = "ETHUSDT"
SHADOW_DECISIONS_DIR = Path("/Users/mark/mark19_data/shadow_decisions")
SHADOW_FILLS_DIR = Path("/Users/mark/mark19_data/shadow_fills")
ASSUMED_MAKER_FILL_RATE = 0.38   # for shadow estimation only


def _load_env_keys() -> tuple[Optional[str], Optional[str]]:
    """Read BYBIT_API_KEY/SECRET from env or live_bot/.env. Returns (key, secret) or (None, None)."""
    key = os.environ.get("MARK19_BYBIT_KEY") or os.environ.get("BYBIT_API_KEY")
    secret = os.environ.get("MARK19_BYBIT_SECRET") or os.environ.get("BYBIT_API_SECRET")
    if key and secret:
        return key, secret
    env_path = Path("/Users/mark/Desktop/Mark/mark19/live_bot/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("#"): continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k == "BYBIT_API_KEY" and not key: key = v
            elif k == "BYBIT_API_SECRET" and not secret: secret = v
    return key, secret


def _utc_boundary(days_back: int = 0) -> int:
    """Returns ms timestamp for UTC midnight today minus N days."""
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((now - timedelta(days=days_back)).timestamp() * 1000)


def _sum_realized_pnl(pnl_list: list, since_ms: int) -> tuple[float, int]:
    """Sum closedPnl from raw list, filtered by createdTime >= since_ms. (sum, count)."""
    total = 0.0; n = 0
    for r in pnl_list:
        ts = int(r.get("createdTime", 0) or 0)
        if ts < since_ms: continue
        try:
            total += float(r.get("closedPnl", 0) or 0)
            n += 1
        except Exception: pass
    return total, n


# ========== LIVE snapshot ==========
def live_snapshot(log) -> Optional[dict]:
    """Fetch real position/balance/pnl from Bybit. Returns None if API unavailable."""
    key, secret = _load_env_keys()
    if not key or not secret:
        log.warning("[dashboard] no API keys — skipping live snapshot")
        return None
    try:
        st = bybit_v5.fetch_state_and_map(key, secret, SYMBOL)
        m = st["mapped"]
        # Closed PnL history (last 7 days for week aggregation)
        week_start = _utc_boundary(7)
        try:
            cp = bybit_v5.get_closed_pnl(key, secret, SYMBOL, start_ms=week_start)
            pnl_list = cp.get("result", {}).get("list", [])
        except Exception as e:
            log.warning(f"[dashboard] closed-pnl fetch failed: {e}")
            pnl_list = []

        # Position details
        positions = m.get("positions", [])
        position_summary = []
        unrealized_total = 0.0
        for p in positions:
            position_summary.append({
                "side": p.get("side"),
                "size": p.get("size"),
                "avgPrice": p.get("avgPrice"),
                "unrealisedPnl": p.get("unrealisedPnl"),
            })
            try: unrealized_total += float(p.get("unrealisedPnl", 0) or 0)
            except Exception: pass

        # Day/week/month realized
        today_realized, today_n = _sum_realized_pnl(pnl_list, _utc_boundary(0))
        week_realized, week_n = _sum_realized_pnl(pnl_list, _utc_boundary(7))
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_realized, month_n = _sum_realized_pnl(pnl_list, int(month_start.timestamp() * 1000))

        balance = float(m["balance_usdt"])
        equity = float(m["total_equity"])

        return {
            "mode": "live",
            "ts": datetime.now(timezone.utc).isoformat(),
            "balance_usdt": balance,
            "total_equity_usdt": equity,
            "positions": position_summary,
            "unrealized_pnl_usd": unrealized_total,
            "today_realized_usd": today_realized,
            "today_trades": today_n,
            "week_realized_usd": week_realized,
            "week_trades": week_n,
            "month_realized_usd": month_realized,
            "month_trades": month_n,
            "today_pct": (today_realized + unrealized_total) / equity * 100 if equity else 0,
            "week_pct": week_realized / equity * 100 if equity else 0,
            "month_pct": month_realized / equity * 100 if equity else 0,
        }
    except Exception as e:
        log.error(f"[dashboard] live snapshot failed: {type(e).__name__}: {e}")
        return None


# ========== SHADOW snapshot ==========
def shadow_snapshot(log) -> dict:
    """Aggregate today's shadow decisions + estimated PnL (no real trades)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dec_path = SHADOW_DECISIONS_DIR / f"{today}.jsonl"
    fill_path = SHADOW_FILLS_DIR / f"{today}.jsonl"
    decisions = []
    fills = []
    if dec_path.exists():
        for line in dec_path.read_text().splitlines():
            if line.strip():
                try: decisions.append(json.loads(line))
                except Exception: pass
    if fill_path.exists():
        for line in fill_path.read_text().splitlines():
            if line.strip():
                try: fills.append(json.loads(line))
                except Exception: pass

    n_dec = len(decisions)
    n_actions = sum(1 for d in decisions if d.get("action") != "SKIP")
    actions_today = [d for d in decisions if d.get("action") != "SKIP"]
    pending = [f for f in fills if f.get("filled") is None]
    resolved = [f for f in fills if f.get("filled") is not None]
    n_filled = sum(1 for f in resolved if f.get("filled") == "maker")
    actual_fill_rate = (n_filled / len(resolved)) if resolved else None

    # Try to get current mid for "what's our virtual P&L right now"
    current_mid = None
    try:
        live_dir = Path("/Users/mark/mark19_data/bars_5min_v3_live") / SYMBOL
        files = sorted(live_dir.glob("*.parquet"))
        if files:
            import pandas as pd
            recent = pd.read_parquet(files[-1])
            if len(recent):
                current_mid = float(recent["mid_close"].iloc[-1])
    except Exception:
        pass

    # Estimated PnL = sum over each (resolved + filled) of (sign × (current_mid - entry) - fee)
    est_pnl_bp = 0.0
    est_trade_count = 0
    for f in resolved:
        if f.get("filled") != "maker" or current_mid is None: continue
        try:
            limit = float(f.get("limit_price") or 0)
            if limit == 0: continue
            sign = 1 if f.get("action") == "LONG" else -1
            move_bp = (current_mid - limit) / limit * 10000 * sign
            est_pnl_bp += move_bp - 4.0  # Maker RT 4bp
            est_trade_count += 1
        except Exception: pass

    return {
        "mode": "shadow",
        "ts": datetime.now(timezone.utc).isoformat(),
        "decisions_today": n_dec,
        "actions_attempted": n_actions,
        "actions": [{"ts": d.get("decision_ts"), "side": d.get("action"),
                      "p_up": d.get("p_up"), "limit": d.get("limit_price")} for d in actions_today],
        "pending_count": len(pending),
        "resolved_count": len(resolved),
        "maker_filled": n_filled,
        "actual_fill_rate": actual_fill_rate,
        "assumed_fill_rate": ASSUMED_MAKER_FILL_RATE,
        "current_mid": current_mid,
        "estimated_pnl_bp_today": est_pnl_bp,
        "estimated_trades_resolved": est_trade_count,
    }


# ========== Discord formatting ==========
def _fmt_money(v: Optional[float]) -> str:
    if v is None: return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None: return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def send_snapshot(mode: str, log):
    """Compute snapshot for current mode and send to Discord."""
    if dn is None:
        log.info("[dashboard] Discord notify unavailable — snapshot skipped")
        return

    if mode == "shadow":
        s = shadow_snapshot(log)
        title = "📈 Shadow trading (가상 — 실거래 아님)"
        body_lines = [
            f"**모드**: SHADOW (실주문 X, 가상 시뮬레이션)",
            f"**오늘 decision**: {s['decisions_today']} (active {s['actions_attempted']})",
            f"**가상 진입 누적**: pending {s['pending_count']}, resolved {s['resolved_count']}",
            f"**추정 maker fill rate**: "
            + (f"{s['actual_fill_rate']*100:.1f}% (실측, vs 가정 {s['assumed_fill_rate']*100:.0f}%)"
                if s['actual_fill_rate'] is not None
                else f"미측정 (resolved=0, 가정 {s['assumed_fill_rate']*100:.0f}%)"),
        ]
        if s['current_mid'] is not None:
            body_lines.append(f"**현재 mid**: ${s['current_mid']:.2f}")
        body_lines.append(f"**추정 PnL (오늘, maker -4bp 가정)**: {s['estimated_pnl_bp_today']:+.1f} bp on {s['estimated_trades_resolved']} trades")
        if s['actions']:
            body_lines.append("\n**최근 가상 action** (최대 3개):")
            for a in s['actions'][-3:]:
                body_lines.append(f"  • {a.get('side')} @ ${a.get('limit'):.2f} (p_up={a.get('p_up'):.3f})"
                                    if a.get('limit') else f"  • {a.get('side')} (p_up={a.get('p_up'):.3f})")
        body_lines.append("\n⚠️ **실거래 아님** — Bybit 계좌에 어떤 변경도 없음.")
        return dn.send(title, "\n".join(body_lines), "info")

    elif mode == "live":
        s = live_snapshot(log)
        if s is None:
            return dn.warning("Live snapshot unavailable",
                              "API 키 또는 Bybit 응답 실패. ~/mark19_data/bybit_raw_logs/ 확인.")
        title = "💰 LIVE 거래 현황"
        body_lines = [
            f"**모드**: LIVE — 실거래",
            f"**잔고**: {_fmt_money(s['balance_usdt'])} (available) / 자본 {_fmt_money(s['total_equity_usdt'])}",
            f"**오픈 포지션**: {len(s['positions'])}",
        ]
        for p in s['positions']:
            body_lines.append(f"  • {p['side']} {p['size']} ETH @ ${p['avgPrice']}  "
                              f"미실현 {_fmt_money(float(p['unrealisedPnl'] or 0))}")
        body_lines.extend([
            "",
            f"**오늘** (UTC): {_fmt_money(s['today_realized_usd'])} 실현 + {_fmt_money(s['unrealized_pnl_usd'])} 미실현 "
            f"= {_fmt_pct(s['today_pct'])} ({s['today_trades']} 거래)",
            f"**이번 주**: {_fmt_money(s['week_realized_usd'])} 실현 ({s['week_trades']} 거래) {_fmt_pct(s['week_pct'])}",
            f"**이번 달**: {_fmt_money(s['month_realized_usd'])} 실현 ({s['month_trades']} 거래) {_fmt_pct(s['month_pct'])}",
        ])
        # Liquidation distance (1x = 99% away → safe)
        for p in s['positions']:
            try:
                avg = float(p['avgPrice'])
                side = p['side']
                # 1x leverage → liq at ~99% away (we don't fetch liqPrice directly here; mark19_live's RiskRail formula)
                liq_pct = 99.0  # at 1x
                body_lines.append(f"  liq distance ~{liq_pct:.0f}% (at 1x leverage = safe)")
            except Exception: pass
        return dn.send(title, "\n".join(body_lines), "info")

    else:
        log.warning(f"[dashboard] unknown mode {mode!r}")


def on_trade_event(event: str, payload: dict, mode: str = "live"):
    """Immediate alert for entry/fill/exit. mode='shadow' adds (가상) suffix."""
    if dn is None: return
    suffix = " (가상)" if mode == "shadow" else ""
    if event == "entry":
        title = f"🟢 Entry{suffix}"
        body = (f"**{payload.get('side')}** {payload.get('size')} ETH @ ${payload.get('price')}\n"
                f"P(up): {payload.get('p_up')}  conf: {payload.get('confidence')}")
        return dn.send(title, body, "info")
    elif event == "fill":
        title = f"✅ Fill{suffix}"
        body = (f"{payload.get('side')} {payload.get('qty')} @ ${payload.get('fill_price')}\n"
                f"OrderId: {payload.get('orderId')}")
        return dn.send(title, body, "info")
    elif event == "exit":
        title = f"🔴 Exit{suffix}"
        pnl = payload.get('pnl_usd', 0)
        emoji = "📈" if pnl > 0 else "📉"
        body = (f"{payload.get('side')} closed @ ${payload.get('exit_price')}\n"
                f"{emoji} PnL: {_fmt_money(pnl)}")
        return dn.send(title, body, "info" if pnl >= 0 else "warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="shadow", choices=["shadow", "live"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()
    send_snapshot(args.mode, log)
