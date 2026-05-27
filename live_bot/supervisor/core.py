"""Supervisor state machine.

Rules (from roadmap M3):

  * If rolling-180d PF < 0.8 AND n_trades_180d >= MIN_TRADES
        → status = OFF   (paused_by_auto = 1)

  * If status == OFF AND paused_by_auto == 1 AND rolling-180d PF > 1.15
        AND n_trades_180d >= MIN_TRADES AND reactivation cooldown passed
        → status = PAPER (paused_by_auto = 0)

  * Auto transitions NEVER promote to LIVE — manual CLI only.

  * If `state/HALT` file exists, every strategy is forced to OFF on the next tick,
    with gate_events reason='emergency_halt'.

  * Transitions that would flap (same side of threshold within `MIN_DWELL`)
    are blocked — we wait for genuine hysteresis to clear.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from live_bot.state_store import HALT_PATH
from live_bot.state_store.db import DB, open_db

log = logging.getLogger("supervisor")

# ---- rules (all public so they're testable) ----
ROLLING_WINDOW_DAYS = 180
PF_OFF_THRESHOLD = 0.8
PF_ON_THRESHOLD = 1.15
MIN_TRADES = 20
MIN_DWELL_DAYS = 7          # must stay in a state this long before flip-back


@dataclass
class TransitionDecision:
    strategy_id: int
    strategy_name: str
    from_status: str
    to_status: str
    reason: str
    pf_180d: Optional[float]
    n_trades_180d: int
    changed: bool


# =========================================================
# Rolling-PF computation
# =========================================================
def _pf_over_trades(pnls: List[float]) -> float:
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p <= 0]
    gl = sum(losses)
    if gl <= 0:
        return float("inf") if wins else 0.0
    return sum(wins) / gl


def rolling_pf(db: DB, strategy_id: int, now: datetime,
               window_days: int = ROLLING_WINDOW_DAYS) -> Dict[str, Any]:
    cutoff = (now - timedelta(days=window_days)).isoformat(timespec="seconds")
    # Paper + live count toward gating. Pure backtest trades do not.
    trades = db.trades_since(strategy_id, cutoff, modes=["paper", "live"])
    pnls = [float(t["pnl"]) for t in trades if t["pnl"] is not None]
    return {
        "pf": _pf_over_trades(pnls) if pnls else 0.0,
        "n": len(pnls),
    }


# =========================================================
# Decision logic
# =========================================================
def _last_auto_transition(db: DB, strategy_id: int) -> Optional[datetime]:
    events = db.recent_gate_events(strategy_id, limit=1)
    if not events:
        return None
    try:
        return datetime.fromisoformat(events[0]["ts"]).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def evaluate(db: DB, strategy: Dict[str, Any], now: datetime) -> TransitionDecision:
    sid = int(strategy["id"])
    name = strategy["name"]
    cur_status = strategy["status"]
    paused_by_auto = bool(strategy["paused_by_auto"])

    # Emergency halt.
    if HALT_PATH.exists():
        if cur_status != "OFF":
            return TransitionDecision(
                strategy_id=sid, strategy_name=name,
                from_status=cur_status, to_status="OFF",
                reason="emergency_halt",
                pf_180d=None, n_trades_180d=0, changed=True,
            )
        return TransitionDecision(sid, name, cur_status, cur_status,
                                  "already_off_under_halt", None, 0, False)

    stats = rolling_pf(db, sid, now)
    pf, n = stats["pf"], stats["n"]

    # Persist last-known eval metrics even if no status change.
    db.update_eval_metrics(sid, pf, last_change_point_at=None)

    # Not enough data — never transition.
    if n < MIN_TRADES:
        return TransitionDecision(sid, name, cur_status, cur_status,
                                  "insufficient_sample", pf, n, False)

    # Anti-flap dwell.
    last = _last_auto_transition(db, sid)
    if last is not None and (now - last) < timedelta(days=MIN_DWELL_DAYS):
        return TransitionDecision(sid, name, cur_status, cur_status,
                                  "min_dwell_not_satisfied", pf, n, False)

    # ---- PF below OFF threshold while active → pause ----
    if cur_status in ("PAPER", "LIVE_SHADOW", "LIVE_SMALL_CAPITAL") and pf < PF_OFF_THRESHOLD:
        return TransitionDecision(
            sid, name, cur_status, "OFF",
            reason=f"auto_pf_below_{PF_OFF_THRESHOLD}",
            pf_180d=pf, n_trades_180d=n, changed=True,
        )

    # ---- Auto-revive from OFF only if we paused it ourselves ----
    if cur_status == "OFF" and paused_by_auto and pf > PF_ON_THRESHOLD:
        return TransitionDecision(
            sid, name, "OFF", "PAPER",
            reason=f"auto_pf_above_{PF_ON_THRESHOLD}",
            pf_180d=pf, n_trades_180d=n, changed=True,
        )

    return TransitionDecision(sid, name, cur_status, cur_status,
                              "no_change", pf, n, False)


# =========================================================
# Orchestrator
# =========================================================
def run_once(now: Optional[datetime] = None) -> List[TransitionDecision]:
    now = now or datetime.now(timezone.utc)
    decisions: List[TransitionDecision] = []
    with open_db() as db:
        strategies = db.list_strategies()
        for s in strategies:
            try:
                d = evaluate(db, s, now)
            except Exception as e:
                log.exception("evaluate failed for %s: %s", s["name"], e)
                db.audit("supervisor_error",
                         {"strategy": s["name"], "error": str(e)})
                continue
            if d.changed:
                # paused_by_auto flag logic:
                #   going OFF from active → set paused_by_auto=True (only for auto rules)
                #   going ON from OFF     → set paused_by_auto=False
                paused_by_auto: Optional[bool] = None
                if d.to_status == "OFF" and d.reason.startswith("auto_pf_below"):
                    paused_by_auto = True
                elif d.to_status == "OFF" and d.reason == "emergency_halt":
                    paused_by_auto = False
                elif d.from_status == "OFF" and d.to_status != "OFF":
                    paused_by_auto = False
                db.set_status(
                    strategy_id=d.strategy_id,
                    new_status=d.to_status, reason=d.reason,
                    actor="supervisor",
                    pf_180d=d.pf_180d, n_trades_180d=d.n_trades_180d,
                    paused_by_auto=paused_by_auto,
                )
            decisions.append(d)
    return decisions


# =========================================================
# CLI
# =========================================================
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Strategy supervisor")
    ap.add_argument("--once", action="store_true",
                    help="run a single tick and exit (default: same as --once)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    decisions = run_once()
    for d in decisions:
        marker = "→" if d.changed else " "
        log.info("  %s  %-20s  %-16s %s %-16s  pf=%.3f  n=%d  (%s)",
                 marker, d.strategy_name, d.from_status,
                 "→" if d.changed else "=", d.to_status,
                 d.pf_180d or 0.0, d.n_trades_180d, d.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
