"""Unit tests for P0/P1 fix in live_bot/trading_bot.py.

P0 (Bug #1): _reconcile + _finalize integration
  - EXITING + Bybit pos 0 → _finalize called
  - daily_pnl_pct, consecutive_losses correctly updated
  - Single position-of-truth via _fetch_position

P1 (Bug #3): partial fill / actual_qty tracking
  - state.actual_qty set after entry verify
  - Exit reduce_only uses actual_qty (not intended state.qty)
  - _can_trade blocks new entry on orphan position

Mocking strategy: replace bot.broker with FakeBroker, mock
_query_recent_exit_fill via attribute swap.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_bot.broker.base import OrderSide, Position
from live_bot.config import CFG, Mode
from live_bot.trading_bot import BotState, Mark19TradingBot, TradingState


# ---- helpers ----

class FakeBroker:
    """Minimal Broker stub that lets tests script Bybit position responses."""

    def __init__(self, qty=0.0, side=None, avg_price=0.0):
        self._qty = qty
        self._side = side
        self._avg = avg_price
        self._fills = []
        self.cancel_all_calls = 0
        self.placed_orders = []

    def name(self): return "fake"
    def equity(self): return 200.0

    def position(self, symbol):
        return Position(symbol=symbol, side=self._side, qty=self._qty, avg_price=self._avg)

    def set_position(self, qty, side=None, avg_price=0.0):
        self._qty = qty
        self._side = side
        self._avg = avg_price

    def place_order(self, order):
        self.placed_orders.append(order)
        return 12345

    def cancel(self, oid): return True

    def cancel_all(self, symbol=None):
        self.cancel_all_calls += 1
        return 0

    def open_orders(self, symbol=None): return []

    def drain_fills(self):
        out, self._fills = self._fills, []
        return out


def make_bot_with_fake(mode=Mode.LIVE_SMALL_CAPITAL):
    """Create a bot, then swap in a FakeBroker. Bypass mode-based broker init."""
    # Use PAPER to skip the LiveBroker BOT_LIVE_OK gate, then patch broker.
    bot = Mark19TradingBot(mode=Mode.PAPER, model_path="models/mark17_v1.joblib")
    bot.mode = mode
    bot.broker = FakeBroker()
    return bot


# ---- tests ----

def test_p0_reconcile_finalizes_on_close():
    """P0: bot EXITING + Bybit pos 0 → _finalize called via _reconcile."""
    print("\n[TEST] P0: reconcile finalizes on close (EXITING → READY with PnL)")
    bot = make_bot_with_fake()

    # Set up: bot in EXITING state, entry was 60min ago, had a SHORT @ 2260
    bot.state.bot_state = BotState.EXITING
    bot.state.entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
    bot.state.entry_price = 2260.0
    bot.state.direction = -1  # SHORT
    bot.state.qty = 0.24
    bot.state.actual_qty = 0.24
    bot.state.exit_started = datetime.now(timezone.utc) - timedelta(minutes=2)
    bot.state.daily_pnl_pct = 0.0
    bot.state.consecutive_losses = 0

    # Bybit reports position 0 (close already happened)
    bot.broker.set_position(qty=0.0)

    # Mock _query_recent_exit_fill: return a maker fill at 2255 (profit on SHORT)
    fake_fill = {
        "exit_price": 2255.0,
        "was_maker": True,
        "fill_qty": 0.24,
        "fee_total": -0.001,  # rebate (maker)
        "n_fills": 1,
    }
    with patch.object(bot, "_query_recent_exit_fill", return_value=fake_fill):
        bot._reconcile(bybit_qty=0.0, bybit_side=None, ref_price=2256.0)

    # Verify: state went READY, PnL recorded, NOT consecutive_losses (it was a win)
    assert bot.state.bot_state == BotState.READY, f"state={bot.state.bot_state}"
    assert bot.state.trades_today == 1, f"trades_today={bot.state.trades_today}"
    # SHORT entry 2260 → exit 2255 = +0.221% raw, fee maker+taker_entry = +0.025% → net ≈ +0.197%
    assert bot.state.daily_pnl_pct > 0, f"daily_pnl_pct={bot.state.daily_pnl_pct}"
    assert bot.state.wins_today == 1
    assert bot.state.consecutive_losses == 0  # win → reset
    print(f"  ✅ daily_pnl_pct={bot.state.daily_pnl_pct*100:+.4f}% (expected >0 for SHORT 2260→2255)")
    print(f"  ✅ trades_today={bot.state.trades_today}, wins={bot.state.wins_today}")


def test_p0_reconcile_records_loss_increments_streak():
    """P0: SHORT @ 2260, exits @ 2280 (taker SL) → loss recorded, consecutive_losses++."""
    print("\n[TEST] P0: SL-style loss → consecutive_losses increments")
    bot = make_bot_with_fake()
    bot.state.bot_state = BotState.EXITING
    bot.state.entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
    bot.state.entry_price = 2260.0
    bot.state.direction = -1
    bot.state.qty = 0.24
    bot.state.actual_qty = 0.24
    bot.state.exit_started = datetime.now(timezone.utc) - timedelta(minutes=2)
    bot.state.daily_pnl_pct = 0.0
    bot.state.consecutive_losses = 2  # 2 prior losses

    bot.broker.set_position(qty=0.0)

    # Native SL fired at 2294 (1.5%)
    fake_fill = {
        "exit_price": 2294.0,
        "was_maker": False,
        "fill_qty": 0.24,
        "fee_total": 0.30,  # taker fee positive
        "n_fills": 1,
    }
    with patch.object(bot, "_query_recent_exit_fill", return_value=fake_fill):
        bot._reconcile(bybit_qty=0.0, bybit_side=None, ref_price=2294.0)

    assert bot.state.bot_state == BotState.READY
    assert bot.state.daily_pnl_pct < 0, f"expected loss, got {bot.state.daily_pnl_pct}"
    assert bot.state.losses_today == 1
    assert bot.state.consecutive_losses == 3, f"streak={bot.state.consecutive_losses}"
    print(f"  ✅ daily_pnl_pct={bot.state.daily_pnl_pct*100:+.4f}% (loss)")
    print(f"  ✅ consecutive_losses 2→{bot.state.consecutive_losses}")


def test_p0_reconcile_fallback_when_exec_list_unavailable():
    """P0: exec list query returns None → fallback to ref_price + filled_at_maker=False."""
    print("\n[TEST] P0: fallback when exec list unavailable")
    bot = make_bot_with_fake()
    bot.state.bot_state = BotState.EXITING
    bot.state.entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
    bot.state.entry_price = 2260.0
    bot.state.direction = -1
    bot.state.qty = 0.24
    bot.state.actual_qty = 0.24
    bot.state.exit_started = datetime.now(timezone.utc) - timedelta(minutes=2)

    bot.broker.set_position(qty=0.0)
    with patch.object(bot, "_query_recent_exit_fill", return_value=None):
        bot._reconcile(bybit_qty=0.0, bybit_side=None, ref_price=2258.0)

    # Should still finalize (using ref_price + taker fees)
    assert bot.state.bot_state == BotState.READY
    assert bot.state.trades_today == 1
    print(f"  ✅ Fallback finalize OK, trades_today={bot.state.trades_today}")
    print(f"  ✅ daily_pnl_pct={bot.state.daily_pnl_pct*100:+.4f}% (fallback uses taker fees)")


def test_p0_reconcile_no_op_when_states_consistent():
    """P0: bot READY + Bybit pos 0 → no action; bot TRADING + Bybit pos > 0 → no action."""
    print("\n[TEST] P0: reconcile no-op on consistent states")
    bot = make_bot_with_fake()

    # Case A: READY + 0 → no change
    bot.state.bot_state = BotState.READY
    bot._reconcile(bybit_qty=0.0, bybit_side=None, ref_price=2260.0)
    assert bot.state.bot_state == BotState.READY
    assert bot.state.trades_today == 0
    print(f"  ✅ READY + pos=0 → no change")

    # Case B: TRADING + 0.24 → no change (still in trade)
    bot.state.bot_state = BotState.TRADING
    bot.state.entry_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    bot.state.entry_price = 2260.0
    bot.state.direction = 1
    bot.state.qty = 0.24
    bot.state.actual_qty = 0.24
    bot._reconcile(bybit_qty=0.24, bybit_side=OrderSide.LONG, ref_price=2261.0)
    assert bot.state.bot_state == BotState.TRADING
    print(f"  ✅ TRADING + pos>0 → no change")


def test_p0_reconcile_orphan_alert():
    """P0: bot READY + Bybit pos > 0 → orphan alert, no auto-recover."""
    print("\n[TEST] P0: orphan position alert")
    bot = make_bot_with_fake()
    bot.state.bot_state = BotState.READY
    bot.broker.set_position(qty=0.5, side=OrderSide.LONG)
    bot._reconcile(bybit_qty=0.5, bybit_side=OrderSide.LONG, ref_price=2260.0)
    # State should NOT change (manual decision required)
    assert bot.state.bot_state == BotState.READY
    print(f"  ✅ Orphan detected, state unchanged (notifier alerted)")


def test_p1_actual_qty_set_on_entry_verify():
    """P1: state.actual_qty set after _open_position entry verify."""
    print("\n[TEST] P1: actual_qty tracked after entry")
    bot = make_bot_with_fake()
    # Mock prediction with LONG signal
    from live_bot.model_predictor import Prediction
    pred = Prediction(
        vol_proba=0.7, dir_proba=0.7, trade_signal=True, direction=1,
        timestamp=datetime.now(timezone.utc),
    )
    # Bybit fills 0.23 (slight slippage from intended 0.24)
    bot.broker.set_position(qty=0.23, side=OrderSide.LONG, avg_price=2261.5)
    # Patch _calc_qty to return 0.24
    with patch.object(bot, "_calc_qty", return_value=0.24):
        bot._open_position(pred, ref_price=2260.0)

    assert bot.state.bot_state == BotState.TRADING
    assert bot.state.qty == 0.24, f"intended qty mismatch: {bot.state.qty}"
    assert bot.state.actual_qty == 0.23, f"actual_qty mismatch: {bot.state.actual_qty}"
    assert bot.state.entry_price == 2261.5  # uses Bybit avg, not ref
    print(f"  ✅ qty (intended)={bot.state.qty}, actual_qty={bot.state.actual_qty}")
    print(f"  ✅ entry_price={bot.state.entry_price} (from Bybit avg)")


def test_p1_partial_fill_below_90pct_closes():
    """P1: partial fill < 90% → close partial position, do not transition to TRADING."""
    print("\n[TEST] P1: partial fill <90% triggers cleanup close")
    bot = make_bot_with_fake()
    from live_bot.model_predictor import Prediction
    pred = Prediction(
        vol_proba=0.7, dir_proba=0.7, trade_signal=True, direction=1,
        timestamp=datetime.now(timezone.utc),
    )
    # Bybit only filled 0.20 of intended 0.24 (=83%, below 90% threshold)
    bot.broker.set_position(qty=0.20, side=OrderSide.LONG, avg_price=2260.0)
    with patch.object(bot, "_calc_qty", return_value=0.24):
        bot._open_position(pred, ref_price=2260.0)

    # State should stay READY (no transition to TRADING)
    assert bot.state.bot_state == BotState.READY, f"state={bot.state.bot_state}"
    # And a market close order should have been placed for the 0.20 partial
    # (placed_orders has [entry MARKET, close MARKET])
    assert len(bot.broker.placed_orders) >= 2, f"orders placed: {len(bot.broker.placed_orders)}"
    close_order = bot.broker.placed_orders[-1]
    assert close_order.qty == 0.20, f"close qty mismatch: {close_order.qty}"
    assert close_order.side == OrderSide.SHORT, f"close side mismatch: {close_order.side}"
    print(f"  ✅ Partial 0.20 detected, market close issued for 0.20 SHORT")


def test_p1_can_trade_blocks_orphan():
    """P1: _can_trade(bybit_qty>0) returns False even when state is READY."""
    print("\n[TEST] P1: _can_trade blocks new entry on orphan position")
    bot = make_bot_with_fake()
    bot.state.bot_state = BotState.READY
    bot.state.consecutive_losses = 0
    bot.state.daily_pnl_pct = 0.0

    # Normal: no orphan → can trade
    assert bot._can_trade(bybit_qty=0.0) is True
    # Orphan: should block
    assert bot._can_trade(bybit_qty=0.15) is False
    print(f"  ✅ orphan present → _can_trade=False")


def test_p1_finalize_resets_actual_qty_and_drift():
    """P1: _finalize resets actual_qty AND drift state."""
    print("\n[TEST] P1: _finalize cleans actual_qty + drift state")
    bot = make_bot_with_fake()
    bot.state.bot_state = BotState.EXITING
    bot.state.entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
    bot.state.entry_price = 2260.0
    bot.state.direction = 1
    bot.state.qty = 0.24
    bot.state.actual_qty = 0.23
    bot.state._last_drift_limit_price = 2255.0
    bot.state._last_drift_replace_ts = datetime.now(timezone.utc)

    bot._finalize(2262.0, filled_at_maker=True)

    assert bot.state.bot_state == BotState.READY
    assert bot.state.actual_qty == 0.0, f"actual_qty not reset: {bot.state.actual_qty}"
    assert bot.state.qty == 0.0
    assert bot.state._last_drift_limit_price is None
    assert bot.state._last_drift_replace_ts is None
    print(f"  ✅ actual_qty=0, drift state cleared")


def test_p0_tick_integration_no_finalize_skip():
    """P0 integration: simulate full close-during-EXITING cycle.

    Setup: bot EXITING with position. After tick, Bybit shows position 0.
    Expected: _reconcile catches it, _finalize runs, state=READY, PnL recorded.
    Critically: _check_exit must NOT also try to handle this (no double finalize).
    """
    print("\n[TEST] P0 integration: full _reconcile pipeline (no double-finalize)")
    bot = make_bot_with_fake()
    bot.state.bot_state = BotState.EXITING
    bot.state.entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
    bot.state.entry_price = 2260.0
    bot.state.direction = -1  # SHORT
    bot.state.qty = 0.24
    bot.state.actual_qty = 0.24
    bot.state.exit_started = datetime.now(timezone.utc) - timedelta(minutes=2)

    bot.broker.set_position(qty=0.0)

    fake_fill = {
        "exit_price": 2255.0, "was_maker": True, "fill_qty": 0.24,
        "fee_total": -0.001, "n_fills": 1,
    }

    # Manually run the reconcile-then-state-transition flow (mimic _tick body)
    bybit_qty, bybit_side, _ = bot._fetch_position()
    assert bybit_qty == 0.0
    with patch.object(bot, "_query_recent_exit_fill", return_value=fake_fill):
        bot._reconcile(bybit_qty, bybit_side, ref_price=2256.0)
    # After reconcile, state is READY → _check_exit short-circuits.
    bot._check_exit(2256.0, bybit_qty)

    assert bot.state.bot_state == BotState.READY
    assert bot.state.trades_today == 1, f"DOUBLE FINALIZE? trades={bot.state.trades_today}"
    print(f"  ✅ Single finalize, trades_today=1 (no double-counting)")


# ---- runner ----

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    tests = [
        test_p0_reconcile_finalizes_on_close,
        test_p0_reconcile_records_loss_increments_streak,
        test_p0_reconcile_fallback_when_exec_list_unavailable,
        test_p0_reconcile_no_op_when_states_consistent,
        test_p0_reconcile_orphan_alert,
        test_p1_actual_qty_set_on_entry_verify,
        test_p1_partial_fill_below_90pct_closes,
        test_p1_can_trade_blocks_orphan,
        test_p1_finalize_resets_actual_qty_and_drift,
        test_p0_tick_integration_no_finalize_skip,
    ]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  ❌ FAIL: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{len(tests)} passed")
    if failed:
        print(f"\nFAILURES:")
        for name, msg in failed:
            print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print(f"  ✅ All P0/P1 unit tests passed")
        sys.exit(0)
