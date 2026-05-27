"""Mark19 Live Trading Bot Core.

State machine + 1-min cadence + 1h lockout + Risk management.

PAPER mode: predicts and tracks position internally — no broker calls.
LIVE modes: places orders via broker.place_order(Order(...)).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
from enum import Enum
from typing import Optional

import pandas as pd

from live_bot.config import CFG, Mode, get_leverage_for_mode, get_capital_for_mode
from live_bot.feature_pipeline import build_live_dataset
from live_bot.model_predictor import ModelPredictor, Prediction
import numpy as np
from live_bot.broker.base import Order, OrderSide, OrderType
from live_bot.state_store import heartbeat as heartbeat_mod
from live_bot.notifier import get_notifier
from live_bot.dashboard import get_dashboard_manager

log = logging.getLogger(__name__)


class BotState(str, Enum):
    READY = "READY"          # No position, ready to trade
    TRADING = "TRADING"      # Position open, in 1h lockout
    EXITING = "EXITING"      # Closing position (Maker exit attempt)
    COOLDOWN = "COOLDOWN"    # Risk-triggered, no trading


@dataclass
class TradingState:
    bot_state: BotState = BotState.READY
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    direction: int = 0          # +1 long, -1 short
    qty: float = 0.0            # ETH (intended)
    actual_qty: float = 0.0     # ETH (Bybit-confirmed after entry verify) — P1 fix Bug #3
    exit_started: Optional[datetime] = None

    trades_today: int = 0
    daily_pnl_pct: float = 0.0  # cumulative net % return
    consecutive_losses: int = 0

    cycles_today: int = 0
    signals_today: int = 0
    wins_today: int = 0
    losses_today: int = 0

    last_prediction: Optional[Prediction] = None
    last_check: Optional[datetime] = None
    wallet_equity_usdt: float = 0.0

    last_reset_date: Optional[date] = None  # KST date of last daily reset

    # Drift policy state (sido28b)
    _last_drift_limit_price: Optional[float] = None
    _last_drift_replace_ts: Optional[datetime] = None


class Mark19TradingBot:
    """Core loop:
      every 1 min: features → predict → decide
      if signal & no position: open
      if position & 60min elapsed: try maker exit (5min wait) → taker fallback
      if risk triggered: COOLDOWN
    """

    def __init__(self, mode: str = Mode.PAPER, model_path: str = "models/mark17_v1.joblib"):
        self.mode = mode
        self.predictor = ModelPredictor(model_path)
        self.state = TradingState()

        capital_krw = get_capital_for_mode(mode)
        self.virtual_equity_krw = capital_krw if capital_krw > 0 else 1_000_000
        self.leverage = get_leverage_for_mode(mode)
        self.strategy_name = f"mark17_v1_{mode}"

        # Broker selection per mode:
        #   PAPER         → no broker (internal dry-run, no API calls)
        #   LIVE_SHADOW   → real LiveBroker (read-only intent — order submits skipped in _submit_*)
        #                   Bypasses factory's BOT_LIVE_OK check because no orders placed.
        #   LIVE_SMALL_*  → real LiveBroker via factory (BOT_LIVE_OK + sqlite gate)
        #   LIVE          → real LiveBroker via factory (BOT_LIVE_OK + sqlite gate)
        self.broker = None
        if mode == Mode.LIVE_SHADOW:
            from live_bot.broker.live import LiveBroker
            self.broker = LiveBroker(symbol=CFG.symbol)
            # Use real wallet equity for sizing realism
            try:
                real_equity_usdt = self.broker.equity()
                self.virtual_equity_krw = int(real_equity_usdt * CFG.krw_per_usdt)
                log.info(f"  SHADOW using real wallet equity: ${real_equity_usdt:.2f} USDT")
            except Exception as e:
                log.warning(f"  SHADOW failed to read wallet equity ({e}), using default")
        elif mode in (Mode.LIVE_SMALL_CAPITAL, Mode.LIVE):
            from live_bot.broker import get_broker
            self.broker = get_broker(
                mode="live",
                symbol=CFG.symbol,
                strategy_name=self.strategy_name,
                initial_equity=self.virtual_equity_krw / CFG.krw_per_usdt,
            )

        # Discord notifier (gracefully disabled if no webhook)
        self.notifier = get_notifier()
        self.dashboard = get_dashboard_manager()

        # Resilience counters
        self.consecutive_parquet_errors = 0
        self.MAX_PARQUET_ERRORS = 5  # 5 consecutive failures → emergency stop
        self._emergency_stop = False

        # Track real wallet equity if broker available
        if self.broker is not None:
            try:
                self.state.wallet_equity_usdt = float(self.broker.equity())
            except Exception:
                self.state.wallet_equity_usdt = 0.0

        log.info("Bot initialized:")
        log.info(f"  Mode:      {self.mode}")
        log.info(f"  Capital:   {self.virtual_equity_krw:,} KRW (virtual)")
        log.info(f"  Leverage:  {self.leverage}x")
        log.info(f"  Model:     {self.predictor.model_version}")
        log.info(f"  Broker:    {self.broker.name() if self.broker else 'INTERNAL (PAPER dry-run)'}")
        log.info(f"  Notifier:  {'enabled' if self.notifier.enabled else 'disabled'}")
        log.info(f"  Dashboard: {'enabled' if self.dashboard.enabled else 'disabled'}")

    # ---- helpers ----

    def _get_state_dict(self) -> dict:
        """Build state dict for the dashboard embed."""
        position_dict = None
        if self.state.bot_state == BotState.TRADING:
            position_dict = {
                "direction": "LONG" if self.state.direction == 1 else "SHORT",
                "qty": self.state.qty,
                "entry_price": self.state.entry_price or 0,
            }
        last = self.state.last_prediction
        return {
            "mode": self.mode,
            "capital_krw": self.virtual_equity_krw,
            "leverage": self.leverage,
            "model_version": self.predictor.model_version,
            "bot_state": self.state.bot_state.value,
            "position": position_dict,
            "cycles_today": self.state.cycles_today,
            "signals_today": self.state.signals_today,
            "trades_today": self.state.trades_today,
            "wins_today": self.state.wins_today,
            "losses_today": self.state.losses_today,
            "daily_pnl_pct": self.state.daily_pnl_pct,
            "daily_pnl_krw": self.state.daily_pnl_pct * self.virtual_equity_krw,
            "last_vol_proba": last.vol_proba if last else 0.0,
            "last_dir_proba": last.dir_proba if last else 0.5,
            "last_signal": "TRADE" if (last and last.trade_signal) else "no-trade",
            "wallet_equity_usdt": self.state.wallet_equity_usdt,
        }

    def _safe_dashboard_update(self, force: bool = False):
        try:
            self.dashboard.update(self._get_state_dict(), force=force)
        except Exception as e:
            log.warning(f"Dashboard update failed: {e}")

    def _calc_qty(self, current_price: float) -> float:
        # SAFETY: reserve $8 USDT for fees + maintenance margin + slippage.
        # math.floor (not round) so we never overshoot the buffer.
        # Dynamic capital: query real wallet via broker.equity() each call,
        # so PnL drift doesn't make sizing exceed available margin (Bug 110007).
        SAFETY_BUFFER_USDT = 8.0
        if self.broker is not None:
            try:
                capital_usdt = float(self.broker.equity())
            except Exception as e:
                log.warning(f"_calc_qty: equity() failed ({e}), falling back to config")
                capital_usdt = self.virtual_equity_krw / CFG.krw_per_usdt
        else:
            capital_usdt = self.virtual_equity_krw / CFG.krw_per_usdt
        available_usdt = max(capital_usdt - SAFETY_BUFFER_USDT, 0.0)
        notional = available_usdt * self.leverage
        qty_raw = notional / current_price
        qty = math.floor(qty_raw * 100) / 100  # round DOWN to 0.01
        log.info(
            f"_calc_qty: wallet=${capital_usdt:.2f} buffer=${SAFETY_BUFFER_USDT} "
            f"avail=${available_usdt:.2f} notional=${notional:.2f} → qty={qty}"
        )
        return max(qty, 0.01)

    def _can_trade(self, bybit_qty: float = 0.0) -> bool:
        if self.state.bot_state != BotState.READY:
            return False
        # P1: block new entries while Bybit has an orphan position.
        if self.broker is not None and bybit_qty > 0:
            log.warning(f"_can_trade: blocked — Bybit has orphan position {bybit_qty}")
            return False
        if self.state.consecutive_losses >= CFG.max_consecutive_losses:
            log.warning("Max consecutive losses reached → COOLDOWN")
            self.notifier.risk_alert(
                severity="critical",
                message=f"Max consecutive losses ({self.state.consecutive_losses})",
                daily_pnl_krw=self.state.daily_pnl_pct * self.virtual_equity_krw,
                daily_pnl_pct=self.state.daily_pnl_pct,
            )
            self.state.bot_state = BotState.COOLDOWN
            return False
        if self.state.daily_pnl_pct <= -CFG.max_daily_loss_pct:
            log.warning(f"Max daily loss reached ({self.state.daily_pnl_pct:.3%}) → COOLDOWN")
            self.notifier.risk_alert(
                severity="critical",
                message="Max daily loss reached",
                daily_pnl_krw=self.state.daily_pnl_pct * self.virtual_equity_krw,
                daily_pnl_pct=self.state.daily_pnl_pct,
            )
            self.state.bot_state = BotState.COOLDOWN
            return False
        return True

    # ---- broker actions ----

    def _submit_market(self, side: OrderSide, qty: float, ref_price: float) -> bool:
        """Place a market-equivalent order (Taker). Returns True only on confirmed submit."""
        if self.broker is None:
            log.info(f"  [PAPER] Would place MARKET {side.value} qty={qty} ref=${ref_price:.2f}")
            return True
        if self.mode == Mode.LIVE_SHADOW:
            log.info(f"  [SHADOW] Would place MARKET {side.value} qty={qty} ref=${ref_price:.2f} "
                     f"— signal logged, no order")
            return True
        order = Order(id=0, symbol=CFG.symbol, side=side, order_type=OrderType.MARKET,
                      qty=qty, price=ref_price)
        try:
            oid = self.broker.place_order(order)
        except Exception as e:
            log.error(f"  MARKET order FAILED: {e}")
            self.notifier.error_alert("Order Submit", f"MARKET {side.value} qty={qty}: {e}")
            return False
        log.info(f"  Submitted MARKET {side.value} qty={qty} order_id={oid} (link_id={order.link_id})")
        return True

    def _compute_drift_limit_price(self, exit_side: OrderSide, ref_mid: float) -> float:
        """Compute drift exit limit price based on best bid/ask + drift_offset_bps.

        For LONG exit (sell, exit_side=SHORT): place ABOVE best_ask by offset_bps (passive maker, won't cross).
        For SHORT exit (buy, exit_side=LONG):  place BELOW best_bid by offset_bps.
        Falls back to ref_mid if best bid/ask unavailable.
        """
        offset_bps = getattr(CFG, "drift_offset_bps", 0.5)
        offset_frac = offset_bps / 10000.0  # bp → fraction

        # Try to read latest best bid/ask from today's orderbook parquet
        try:
            from datetime import datetime, timezone
            from mark19.storage import path_for
            today = datetime.now(timezone.utc).date()
            ob_path = path_for("orderbook", "bybit", CFG.symbol, today)
            if ob_path.exists():
                import pandas as pd
                df = pd.read_parquet(ob_path)
                if len(df) > 0 and "bid_0_price" in df.columns and "ask_0_price" in df.columns:
                    last = df.iloc[-1]
                    best_bid = float(last["bid_0_price"])
                    best_ask = float(last["ask_0_price"])
                    if exit_side == OrderSide.SHORT:
                        # selling — place ABOVE best_ask (rest as passive maker)
                        return round(best_ask * (1 + offset_frac), 2)
                    else:
                        # buying — place BELOW best_bid
                        return round(best_bid * (1 - offset_frac), 2)
        except Exception as e:
            log.warning(f"  drift_limit: OB read fail ({e}), fallback to ref_mid")

        # Fallback: ref_mid ± offset
        if exit_side == OrderSide.SHORT:
            return round(ref_mid * (1 + offset_frac), 2)
        else:
            return round(ref_mid * (1 - offset_frac), 2)

    def _submit_maker_limit(self, side: OrderSide, qty: float, price: float) -> bool:
        if self.broker is None:
            log.info(f"  [PAPER] Would place LIMIT(post-only) {side.value} qty={qty} @ ${price:.2f}")
            return True
        if self.mode == Mode.LIVE_SHADOW:
            log.info(f"  [SHADOW] Would place LIMIT(post-only) {side.value} qty={qty} @ ${price:.2f} "
                     f"— signal logged, no order")
            return True
        order = Order(id=0, symbol=CFG.symbol, side=side, order_type=OrderType.LIMIT,
                      qty=qty, price=price, reduce_only=True, post_only=True)
        try:
            oid = self.broker.place_order(order)
        except Exception as e:
            log.error(f"  LIMIT order FAILED: {e}")
            self.notifier.error_alert("Order Submit", f"LIMIT {side.value} qty={qty} @ ${price:.2f}: {e}")
            return False
        log.info(f"  Submitted LIMIT {side.value} qty={qty} @ ${price:.2f} order_id={oid} (link_id={order.link_id})")
        return True

    # ---- state actions ----

    def _open_position(self, prediction: Prediction, ref_price: float):
        qty = self._calc_qty(ref_price)
        side = OrderSide.LONG if prediction.direction == 1 else OrderSide.SHORT

        log.info("OPENING POSITION:")
        log.info(f"  Direction: {side.value} (proba dir={prediction.dir_proba:.3f}, vol={prediction.vol_proba:.3f})")
        log.info(f"  Qty:       {qty} ETH @ ~${ref_price:.2f} (notional ${qty*ref_price:.2f})")

        if not self._submit_market(side, qty, ref_price):
            log.error("  Failed to submit entry, staying READY")
            return

        # Verify Bybit actually filled the order (state-desync guard).
        # PAPER: no broker → trust the dry-run; LIVE: query broker.position().
        actual_entry_price = ref_price
        actual_filled_qty = qty  # P1: track real fill size for partial-fill correctness
        if self.broker is not None:
            time.sleep(1.5)  # let fill propagate
            try:
                pos = self.broker.position(CFG.symbol)
                actual_qty = float(getattr(pos, "qty", 0) or 0)
            except Exception as e:
                log.error(f"  Position verify error: {e}")
                actual_qty = 0.0

            if actual_qty < qty * 0.9:
                log.error(f"  ENTRY VERIFICATION FAILED: requested {qty}, Bybit position {actual_qty}")
                # P1: if there IS a partial position (>0), close it to leave clean state.
                if actual_qty > 0:
                    log.warning(f"  Partial fill {actual_qty} detected, closing to avoid orphan")
                    close_side = OrderSide.SHORT if prediction.direction == 1 else OrderSide.LONG
                    try:
                        self._submit_market(close_side, actual_qty, ref_price)
                    except Exception as e:
                        log.error(f"  Partial close failed: {e}")
                self.notifier.error_alert(
                    "Entry Verify Fail",
                    f"MARKET {side.value} qty={qty} → Bybit position={actual_qty} "
                    f"(need ≥ {qty*0.9:.3f}). Partial closed.",
                )
                return

            # Use actual fill avg if available
            avg_price = float(getattr(pos, "avg_price", 0) or 0)
            if avg_price > 0:
                actual_entry_price = avg_price
            actual_filled_qty = actual_qty  # P1: real Bybit-confirmed quantity
            log.info(f"  Entry verified: Bybit position={actual_qty} @ ${actual_entry_price:.2f}")

        self.state.bot_state = BotState.TRADING
        self.state.entry_time = datetime.now(timezone.utc)
        self.state.entry_price = actual_entry_price
        self.state.direction = prediction.direction
        self.state.qty = qty
        self.state.actual_qty = actual_filled_qty  # P1 fix Bug #3

        # Bybit-native stop-loss: bot-crash safety net.
        # SL = entry ± 1.5% (just below daily-loss cap so it triggers before COOLDOWN).
        if self.broker is not None and self.mode != Mode.LIVE_SHADOW:
            try:
                sl_pct = 0.015  # 1.5%
                if prediction.direction == 1:  # LONG → SL below
                    sl_price = actual_entry_price * (1 - sl_pct)
                    sl_side = "long"
                else:  # SHORT → SL above
                    sl_price = actual_entry_price * (1 + sl_pct)
                    sl_side = "short"
                resp = self.broker._client.set_trading_stop(sl_price=sl_price, side=sl_side)
                if resp.get("retCode") == 0:
                    log.info(f"  Native SL set @ ${sl_price:.2f} ({sl_pct*100:.1f}%)")
                else:
                    log.warning(f"  Native SL set failed: {resp.get('retMsg')}")
            except Exception as e:
                log.warning(f"  Native SL exception: {e}")

        # Notify (only after verify pass)
        self.notifier.position_opened(
            direction=side.value.upper(),
            qty=qty,
            entry_price=actual_entry_price,
            vol_proba=prediction.vol_proba,
            dir_proba=prediction.dir_proba,
            notional_usdt=qty * actual_entry_price,
        )

    def _try_close(self, ref_price: float, bybit_qty: float = -1.0):
        """Place first maker exit limit at 60min mark.

        P0: position==0 case is now handled by _reconcile upstream (calls finalize).
        Bug #2 fix: cancel_all before placing first limit (clean any leftover).
        P1 fix: use state.actual_qty (Bybit-confirmed) instead of intended state.qty.
        """
        if self.state.bot_state != BotState.TRADING:
            return
        elapsed_min = (datetime.now(timezone.utc) - self.state.entry_time).total_seconds() / 60
        if elapsed_min < CFG.exit_target_minutes:
            return

        # Bug #2: clean any leftover orders before first exit limit.
        # Note: set_trading_stop's native SL is position-attached, NOT in open orders,
        # so cancel_all does NOT remove it.
        if self.broker is not None and self.mode != Mode.LIVE_SHADOW:
            try:
                self.broker.cancel_all(CFG.symbol)
            except Exception as e:
                log.warning(f"_try_close: pre-exit cancel_all failed: {e} (continuing)")

        log.info(f"60 min elapsed (={elapsed_min:.1f}min), placing maker exit limit")
        exit_side = OrderSide.SHORT if self.state.direction == 1 else OrderSide.LONG
        exit_qty = self.state.actual_qty if self.state.actual_qty > 0 else self.state.qty
        # sido28b: place at best_bid/ask offset by drift_offset_bps (passive maker)
        limit_price = self._compute_drift_limit_price(exit_side, ref_price)
        if not self._submit_maker_limit(exit_side, exit_qty, limit_price):
            log.error("  Maker exit submit failed, retrying next tick")
            return
        self.state.bot_state = BotState.EXITING
        self.state.exit_started = datetime.now(timezone.utc)
        self.state._last_drift_limit_price = limit_price
        self.state._last_drift_replace_ts = datetime.now(timezone.utc)

    def _check_exit(self, ref_price: float, bybit_qty: float = -1.0):
        """Drift policy with single position source (P0 fix).

        Position==0 detection now lives in _reconcile (called upstream) so we
        finalize via execution-list there. This function only handles:
          (a) PAPER timeout simulation
          (b) WS fills queue (future)
          (c) defensive double-check on bybit_qty==0 (race)
          (d) timeout → taker fallback
          (e) drift cancel/replace with cooldown + min-move guards
        """
        if self.state.bot_state != BotState.EXITING:
            return
        elapsed_min = (datetime.now(timezone.utc) - self.state.exit_started).total_seconds() / 60

        # PAPER: no broker → simulate fill after timeout
        if self.broker is None:
            if elapsed_min >= CFG.exit_max_wait_minutes:
                self._finalize(ref_price, filled_at_maker=True)
            return

        # SHADOW: simulated maker exit at timeout
        if self.mode == Mode.LIVE_SHADOW:
            if elapsed_min >= CFG.exit_max_wait_minutes:
                log.info(f"[SHADOW] simulated maker exit at t={elapsed_min:.1f}min")
                self._finalize(ref_price, filled_at_maker=True)
            return

        # (b) Fills queue (WS — future). Currently always [] on LIVE.
        fills = self.broker.drain_fills()
        if fills:
            log.info(f"Maker exit filled via fills queue (t={elapsed_min:.1f}min)")
            self._resolve_close_and_finalize(ref_price, reason="fills-queue")
            return

        # (c) Defensive: position==0 — race between reconcile + check_exit.
        # Normally _reconcile catches this first; this is a backstop.
        if bybit_qty == 0:
            log.info(f"_check_exit: Bybit pos 0 (defensive backstop, t={elapsed_min:.1f}min)")
            self._resolve_close_and_finalize(ref_price, reason="check_exit-defensive")
            return

        # (d) Timeout → market fallback (taker)
        if elapsed_min >= CFG.exit_max_wait_minutes:
            log.warning(f"Maker exit timeout {elapsed_min:.1f}min ≥ {CFG.exit_max_wait_minutes}min, taker fallback")
            try:
                self.broker.cancel_all(CFG.symbol)
            except Exception as e:
                log.warning(f"cancel_all (timeout) failed: {e}")
            exit_side = OrderSide.SHORT if self.state.direction == 1 else OrderSide.LONG
            exit_qty = self.state.actual_qty if self.state.actual_qty > 0 else self.state.qty
            self._submit_market(exit_side, exit_qty, ref_price)
            time.sleep(1.5)  # let fill propagate before exec-list query
            self._resolve_close_and_finalize(ref_price, reason="timeout-taker")
            return

        # (e) Drift: cancel/replace, gated by cooldown + min-move
        exit_side = OrderSide.SHORT if self.state.direction == 1 else OrderSide.LONG
        new_limit = self._compute_drift_limit_price(exit_side, ref_price)
        last_price = getattr(self.state, "_last_drift_limit_price", None)
        last_ts = getattr(self.state, "_last_drift_replace_ts", None)
        now = datetime.now(timezone.utc)

        cooldown_sec = getattr(CFG, "drift_replace_cooldown_sec", 30)
        min_move_bps = getattr(CFG, "drift_min_replace_move_bps", 0.5)

        if last_ts is not None and (now - last_ts).total_seconds() < cooldown_sec:
            log.info(f"_check_exit drift: cooldown ({(now-last_ts).total_seconds():.1f}s < {cooldown_sec}s), skip replace")
            return
        if last_price is not None and last_price > 0:
            move_bps = abs(new_limit - last_price) / last_price * 10000
            if move_bps < min_move_bps:
                log.info(f"_check_exit drift: move {move_bps:.2f}bp < {min_move_bps}bp, skip replace")
                return

        exit_qty = self.state.actual_qty if self.state.actual_qty > 0 else self.state.qty
        log.info(f"_check_exit drift: t={elapsed_min:.1f}min, replace limit {exit_side.value} qty={exit_qty} @ ${new_limit:.2f} (was ${last_price})")
        try:
            self.broker.cancel_all(CFG.symbol)
        except Exception as e:
            log.warning(f"cancel_all (drift) failed: {e} (continuing)")
        if not self._submit_maker_limit(exit_side, exit_qty, new_limit):
            log.warning(f"_check_exit drift: replace failed, retry next tick")
            return
        self.state._last_drift_limit_price = new_limit
        self.state._last_drift_replace_ts = now

    def _finalize(self, exit_price: float, filled_at_maker: bool):
        if self.state.direction == 1:
            raw = (exit_price - self.state.entry_price) / self.state.entry_price
        else:
            raw = (self.state.entry_price - exit_price) / self.state.entry_price

        fee = (CFG.fee_taker + CFG.fee_maker) if filled_at_maker else (CFG.fee_taker * 2)
        net = raw - fee

        self.state.daily_pnl_pct += net
        if net > 0:
            self.state.consecutive_losses = 0
            self.state.wins_today += 1
        else:
            self.state.consecutive_losses += 1
            self.state.losses_today += 1
        self.state.trades_today += 1

        duration_min = (datetime.now(timezone.utc) - self.state.entry_time).total_seconds() / 60
        direction_str = "LONG" if self.state.direction == 1 else "SHORT"
        net_pnl_krw = net * self.virtual_equity_krw

        log.info("TRADE FINALIZED:")
        log.info(f"  Side: {direction_str} qty={self.state.qty}")
        log.info(f"  Entry ${self.state.entry_price:.2f} → Exit ${exit_price:.2f}")
        log.info(f"  Raw {raw*100:+.3f}% | Fee {fee*100:.3f}% | Net {net*100:+.3f}%")
        log.info(f"  MakerExit={filled_at_maker} | trades_today={self.state.trades_today} "
                 f"daily_pnl={self.state.daily_pnl_pct*100:+.3f}% losses_streak={self.state.consecutive_losses}")

        # Notify
        self.notifier.position_closed(
            direction=direction_str,
            entry_price=self.state.entry_price,
            exit_price=exit_price,
            raw_pnl_pct=raw,
            net_pnl_pct=net,
            net_pnl_krw=net_pnl_krw,
            fee_pct=fee,
            filled_at_maker=filled_at_maker,
            duration_min=duration_min,
        )

        # Risk alert on big single-trade loss
        if net <= -CFG.alert_loss_pct:
            severity = "critical" if net <= -0.05 else "warning"
            self.notifier.risk_alert(
                severity=severity,
                message=f"Big loss: {net*100:+.2f}%",
                daily_pnl_krw=self.state.daily_pnl_pct * self.virtual_equity_krw,
                daily_pnl_pct=self.state.daily_pnl_pct,
            )

        self.state.bot_state = BotState.READY
        self.state.entry_time = None
        self.state.entry_price = None
        self.state.direction = 0
        self.state.qty = 0.0
        self.state.actual_qty = 0.0  # P1 fix
        self.state.exit_started = None
        self.state._last_drift_limit_price = None
        self.state._last_drift_replace_ts = None

        # Dashboard force update on trade close
        self._safe_dashboard_update(force=True)

    # ---- daily reset (KST midnight) ----

    def _check_daily_reset(self) -> bool:
        """At KST midnight, reset daily counters and lift COOLDOWN.

        Idempotent: re-calling within the same KST day is a no-op.
        Returns True iff a reset was performed.
        """
        today_kst = datetime.now(KST).date()

        # First tick after process start
        if self.state.last_reset_date is None:
            self.state.last_reset_date = today_kst
            log.info(f"Daily reset baseline: {today_kst} (KST)")
            return False

        if today_kst == self.state.last_reset_date:
            return False

        # Day rolled over → reset
        prev_date = self.state.last_reset_date
        prev_trades = self.state.trades_today
        prev_wins = self.state.wins_today
        prev_losses = self.state.losses_today
        prev_pnl_pct = self.state.daily_pnl_pct
        prev_pnl_krw = self.state.daily_pnl_pct * self.virtual_equity_krw
        prev_state = self.state.bot_state.value

        log.info("=" * 60)
        log.info(f"DAILY RESET (KST): {prev_date} → {today_kst}")
        log.info(f"  Prev day: trades={prev_trades} wins={prev_wins} losses={prev_losses}")
        log.info(f"  Prev day: pnl={prev_pnl_pct*100:+.3f}% (₩{prev_pnl_krw:+,.0f})")
        log.info(f"  Prev state: {prev_state}")
        log.info("=" * 60)

        # Reset counters
        self.state.consecutive_losses = 0
        self.state.daily_pnl_pct = 0.0
        self.state.trades_today = 0
        self.state.wins_today = 0
        self.state.losses_today = 0
        self.state.cycles_today = 0
        self.state.signals_today = 0
        self.state.last_reset_date = today_kst

        # Lift COOLDOWN if active
        if self.state.bot_state == BotState.COOLDOWN:
            log.info(f"Releasing COOLDOWN → READY")
            self.state.bot_state = BotState.READY

        # Notify Discord daily summary (only if any trade happened yesterday)
        if prev_trades > 0:
            try:
                win_rate = prev_wins / prev_trades
                self.notifier.daily_summary(
                    trades=prev_trades, wins=prev_wins, losses=prev_losses,
                    daily_pnl_krw=prev_pnl_krw, daily_pnl_pct=prev_pnl_pct,
                    win_rate=win_rate, avg_win_pct=0.0, avg_loss_pct=0.0,
                )
            except Exception as e:
                log.warning(f"daily_summary notify failed: {e}")

        try:
            self.notifier.info(f"🌅 새 날 시작 ({today_kst} KST) — 카운터/COOLDOWN reset 완료")
        except Exception:
            pass

        # Dashboard: drop in-memory msg_id → next update POSTs a fresh
        # message for the new day (yesterday's dashboard stays in channel).
        try:
            self.dashboard.reset()
        except Exception as e:
            log.warning(f"Dashboard reset failed: {e}")
        self._safe_dashboard_update(force=True)
        return True

    # ---- state recovery ----

    def _recover_today_state(self):
        """Rebuild today's daily counters from Bybit order history.

        Called once at run() start. Pairs entries (closedSize=0) with exits
        (closedSize>0) in chronological order to compute trades_today,
        wins/losses_today, daily_pnl_pct, consecutive_losses.

        LIVE_* only — PAPER/SHADOW skip silently.
        """
        if self.broker is None or self.mode in (Mode.PAPER, Mode.LIVE_SHADOW):
            log.info(f"State recovery: skipped (mode={self.mode})")
            return

        log.info("=" * 60)
        log.info("STATE RECOVERY: fetching today's Bybit order history")
        log.info("=" * 60)

        try:
            import hmac, hashlib, requests as rq

            now_kst = datetime.now(KST)
            midnight_kst = datetime.combine(now_kst.date(), datetime.min.time(), tzinfo=KST)
            midnight_ms = int(midnight_kst.timestamp() * 1000)

            # Use /v5/execution/list — it reliably populates closedSize per fill.
            ts = str(int(time.time() * 1000))
            params = f"category=linear&symbol={CFG.symbol}&limit=100&startTime={midnight_ms}"
            sig = hmac.new(
                CFG.api_secret.encode(),
                f"{ts}{CFG.api_key}5000{params}".encode(),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "X-BAPI-API-KEY": CFG.api_key,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-SIGN": sig,
                "X-BAPI-RECV-WINDOW": "5000",
            }
            r = rq.get(f"{CFG.base_url}/v5/execution/list?{params}", headers=headers, timeout=10)
            data = r.json()
            if data.get("retCode") != 0:
                log.warning(f"  Execution list fetch failed: {data.get('retMsg')}")
                return

            execs = data.get("result", {}).get("list", []) or []
            execs = sorted(execs, key=lambda x: int(x.get("execTime", 0)))
            log.info(f"  Today's executions: {len(execs)}")
            if not execs:
                log.info("  No trades today — state remains fresh")
                return

            # Group fills by orderId (a single order can have multiple partial fills).
            # Each order is either entry (closedSize == 0 across fills) or exit
            # (sum of closedSize > 0 → closes prior position).
            from collections import OrderedDict
            by_order = OrderedDict()
            for e in execs:
                oid = e.get("orderId", "")
                ts_e = int(e.get("execTime", 0))
                qty_e = float(e.get("execQty", 0) or 0)
                px_e = float(e.get("execPrice", 0) or 0)
                fee_e = float(e.get("execFee", 0) or 0)
                cs_e = float(e.get("closedSize", 0) or 0)
                side_e = e.get("side")
                if oid not in by_order:
                    by_order[oid] = {
                        "ts": ts_e, "side": side_e, "qty": 0.0,
                        "fee": 0.0, "weighted_px": 0.0, "closed": 0.0,
                    }
                rec = by_order[oid]
                rec["qty"] += qty_e
                rec["fee"] += fee_e
                rec["closed"] += cs_e
                rec["weighted_px"] += px_e * qty_e

            for oid, rec in by_order.items():
                rec["avg_price"] = (rec["weighted_px"] / rec["qty"]) if rec["qty"] > 0 else 0.0
                rec["is_exit"] = rec["closed"] > 0

            trades = []
            current_entry = None
            for oid, rec in by_order.items():
                if rec["is_exit"]:
                    if current_entry is None:
                        log.warning(f"  Exit without entry — skipping {oid[:16]}")
                        continue
                    e_side = current_entry["side"]
                    e_price = current_entry["avg_price"]
                    qty = current_entry["qty"]
                    exit_price = rec["avg_price"]
                    if e_side == "Buy":
                        raw_pnl = (exit_price - e_price) / e_price
                    else:
                        raw_pnl = (e_price - exit_price) / e_price
                    notional = e_price * qty
                    fee_pct = ((current_entry["fee"] + rec["fee"]) / notional) if notional > 0 else 0.001
                    net_pnl = raw_pnl - fee_pct
                    pnl_usd = net_pnl * notional
                    trades.append({
                        "side": e_side, "entry": e_price, "exit": exit_price,
                        "qty": qty, "raw": raw_pnl, "net": net_pnl, "pnl_usd": pnl_usd,
                    })
                    log.info(f"  Trade: {e_side:4} qty={qty:.2f} {e_price:.2f} → {exit_price:.2f} = {net_pnl*100:+.3f}% (${pnl_usd:+.2f})")
                    current_entry = None
                else:
                    current_entry = rec

            wins = sum(1 for t in trades if t["net"] > 0)
            losses = sum(1 for t in trades if t["net"] <= 0)
            total_pnl_usd = sum(t["pnl_usd"] for t in trades)
            total_pnl_krw = int(total_pnl_usd * CFG.krw_per_usdt)

            capital_usdt = self.virtual_equity_krw / CFG.krw_per_usdt
            daily_pnl_pct = (total_pnl_usd / capital_usdt) if capital_usdt > 0 else 0.0

            consecutive = 0
            for t in reversed(trades):
                if t["net"] <= 0:
                    consecutive += 1
                else:
                    break

            self.state.trades_today = len(trades)
            self.state.wins_today = wins
            self.state.losses_today = losses
            self.state.daily_pnl_pct = daily_pnl_pct
            self.state.consecutive_losses = consecutive

            log.info("=" * 60)
            log.info(f"  Recovered: {len(trades)} trades  ({wins}W / {losses}L)")
            log.info(f"  Daily PnL: {daily_pnl_pct*100:+.2f}% (${total_pnl_usd:+.2f} / ₩{total_pnl_krw:+,})")
            log.info(f"  Consecutive losses: {consecutive}")

            if consecutive >= CFG.max_consecutive_losses:
                log.warning(f"  Consecutive losses ≥ {CFG.max_consecutive_losses} → COOLDOWN")
                self.state.bot_state = BotState.COOLDOWN
            log.info("=" * 60)

        except Exception as e:
            log.error(f"State recovery failed: {e}", exc_info=True)

    # ---- reconciliation ----

    def _fetch_position(self) -> tuple:
        """P0 fix: single position-of-truth query per tick.

        Returns (qty, side, avg_price). PAPER (broker=None) → (0, None, 0).
        On API error, returns last-known state to avoid spurious desync.
        """
        if self.broker is None:
            return 0.0, None, 0.0
        try:
            pos = self.broker.position(CFG.symbol)
            qty = float(getattr(pos, "qty", 0) or 0)
            side = getattr(pos, "side", None)
            avg = float(getattr(pos, "avg_price", 0) or 0)
            return qty, side, avg
        except Exception as e:
            log.warning(f"_fetch_position failed: {e}; using last-known state")
            side = OrderSide.LONG if self.state.direction == 1 else (
                OrderSide.SHORT if self.state.direction == -1 else None)
            fallback_qty = float(self.state.actual_qty or self.state.qty or 0)
            return fallback_qty, side, 0.0

    def _query_recent_exit_fill(self) -> Optional[dict]:
        """Query Bybit /v5/execution/list for exit fills since state.entry_time.

        Returns {exit_price, was_maker, fill_qty, fee_total, n_fills} or None.
        Used by _resolve_close_and_finalize for accurate PnL recording.
        """
        if self.broker is None or self.mode == Mode.LIVE_SHADOW:
            return None
        if self.state.entry_time is None:
            return None
        try:
            import hmac, hashlib, requests as rq

            start_ms = int(self.state.entry_time.timestamp() * 1000)
            ts = str(int(time.time() * 1000))
            params = f"category=linear&symbol={CFG.symbol}&limit=50&startTime={start_ms}"
            sig = hmac.new(
                CFG.api_secret.encode(),
                f"{ts}{CFG.api_key}5000{params}".encode(),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "X-BAPI-API-KEY": CFG.api_key,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-SIGN": sig,
                "X-BAPI-RECV-WINDOW": "5000",
            }
            r = rq.get(f"{CFG.base_url}/v5/execution/list?{params}", headers=headers, timeout=10)
            data = r.json()
            if data.get("retCode") != 0:
                log.warning(f"_query_recent_exit_fill: retMsg={data.get('retMsg')}")
                return None
            execs = data.get("result", {}).get("list", []) or []
            exit_fills = [e for e in execs if float(e.get("closedSize", 0) or 0) > 0]
            if not exit_fills:
                return None

            total_qty = sum(float(e.get("execQty", 0) or 0) for e in exit_fills)
            if total_qty <= 0:
                return None
            weighted_px = sum(
                float(e.get("execQty", 0) or 0) * float(e.get("execPrice", 0) or 0)
                for e in exit_fills
            )
            avg_price = weighted_px / total_qty
            total_fee = sum(float(e.get("execFee", 0) or 0) for e in exit_fills)
            # Bybit V5 isMaker flag is most reliable; fall back to fee sign.
            maker_qty = sum(
                float(e.get("execQty", 0) or 0) for e in exit_fills
                if (e.get("isMaker") is True or str(e.get("isMaker")).lower() == "true")
            )
            if maker_qty > 0 or any("isMaker" in e for e in exit_fills):
                was_maker = (maker_qty / total_qty) > 0.5
            else:
                # Fallback: maker fees on Bybit are negative (rebate)
                was_maker = total_fee < 0
            return {
                "exit_price": avg_price,
                "was_maker": was_maker,
                "fill_qty": total_qty,
                "fee_total": total_fee,
                "n_fills": len(exit_fills),
            }
        except Exception as e:
            log.warning(f"_query_recent_exit_fill failed: {e}")
            return None

    def _resolve_close_and_finalize(self, ref_price: Optional[float], reason: str = "reconcile"):
        """P0 fix: smart close — query Bybit execution list, finalize with accurate price/maker.

        Falls back to ref_price + filled_at_maker=False if query fails.
        Caller must already know bot is TRADING/EXITING and Bybit position is 0.
        """
        fill = self._query_recent_exit_fill()
        if fill is not None:
            log.info(f"  CLOSE [{reason}]: exec_list={fill['n_fills']} fills, "
                     f"avg ${fill['exit_price']:.2f}, qty {fill['fill_qty']:.4f}, "
                     f"maker={fill['was_maker']}, fee=${fill['fee_total']:.4f}")
            exit_price = fill["exit_price"]
            was_maker = fill["was_maker"]
        else:
            log.warning(f"  CLOSE [{reason}]: exec_list unavailable, fallback ref_price=${ref_price}")
            if ref_price is None or not (isinstance(ref_price, (int, float)) and ref_price > 0):
                exit_price = self.state.entry_price or 0.0
                log.error(f"  CLOSE [{reason}]: no price source, using entry (PnL≈0)")
            else:
                exit_price = float(ref_price)
            was_maker = False  # conservative — assume taker
        self._finalize(exit_price, filled_at_maker=was_maker)

    def _reconcile(self, bybit_qty: float, bybit_side, ref_price: Optional[float]):
        """P0 fix: single-source reconcile using cached position from _fetch_position.

        On EXITING/TRADING + position 0, calls _resolve_close_and_finalize so PnL
        is recorded via Bybit execution list (instead of silent state reset).
        """
        if self.broker is None or self.mode == Mode.LIVE_SHADOW:
            return  # PAPER / SHADOW: no reconcile needed

        bot_state = self.state.bot_state

        # Case 1: Bot READY but Bybit has position → orphan
        if bot_state == BotState.READY and bybit_qty > 0:
            log.warning(f"RECONCILE: Bot READY but Bybit position {bybit_qty} {bybit_side}")
            self.notifier.error_alert(
                "Reconcile: Orphan Position",
                f"Bot=READY, Bybit={bybit_qty} {bybit_side}. Manual intervention may be needed.",
            )
            return

        # Case 2: Bot TRADING/EXITING but Bybit position 0 → close detected → finalize
        if bot_state in (BotState.TRADING, BotState.EXITING) and bybit_qty == 0:
            log.info(f"RECONCILE: Bot {bot_state.value} but Bybit position 0 — finalizing")
            self._resolve_close_and_finalize(ref_price, reason="reconcile")

    # ---- main tick ----

    def _tick(self):
        now = datetime.now(timezone.utc)
        self.state.last_check = now

        # KST midnight reset (idempotent)
        self._check_daily_reset()

        # P0: single position-of-truth — query Bybit ONCE per tick.
        # Pass the cached value to reconcile, _try_close, _check_exit, _can_trade.
        bybit_qty, bybit_side, bybit_avg = self._fetch_position()

        self.state.cycles_today += 1

        # Dashboard rate-limited update (PATCH every 30 min)
        self._safe_dashboard_update(force=False)

        # Build full live dataset; extract both ref_price (ob_mid_price)
        # and the model-feature row from the same final 1-min bar.
        try:
            df = build_live_dataset(
                now=now,
                lookback_hours=25,
                train_medians=self.predictor.train_medians.to_dict(),
            )
            self.consecutive_parquet_errors = 0  # successful fetch resets counter
        except Exception as e:
            log.error(f"Feature fetch error: {e}", exc_info=False)
            err_str = str(e).lower()
            if "parquet" in err_str or "magic bytes" in err_str:
                self.consecutive_parquet_errors += 1
                log.warning(
                    f"Parquet error counter: {self.consecutive_parquet_errors}/{self.MAX_PARQUET_ERRORS}"
                )
                if self.consecutive_parquet_errors >= self.MAX_PARQUET_ERRORS:
                    log.error(f"EMERGENCY STOP: {self.MAX_PARQUET_ERRORS} consecutive parquet errors")
                    self.notifier.error_alert(
                        "Emergency Stop",
                        f"{self.MAX_PARQUET_ERRORS} consecutive parquet errors. Bot stopping.",
                    )
                    self._emergency_stop = True
            # Even without features, run reconcile so close events get finalized.
            self._reconcile(bybit_qty, bybit_side, ref_price=None)
            return

        if df.empty:
            log.warning("No features available")
            self._reconcile(bybit_qty, bybit_side, ref_price=None)
            return

        latest = df.iloc[-1]
        ref_price = float(latest.get("ob_mid_price", float("nan")))
        if not np.isfinite(ref_price) or ref_price <= 0:
            log.warning(f"Invalid ref_price={ref_price}, skipping tick")
            self._reconcile(bybit_qty, bybit_side, ref_price=None)
            return

        feature_row = latest.reindex(self.predictor.feature_cols)
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan)
        feature_row = feature_row.fillna(self.predictor.train_medians).fillna(0)
        feature_row.name = latest.get("timestamp", None)

        # P0: reconcile may transition state via _finalize (TRADING/EXITING + qty=0 → READY).
        self._reconcile(bybit_qty, bybit_side, ref_price=ref_price)

        # State transitions on every tick (state may have just changed via reconcile)
        if self.state.bot_state == BotState.EXITING:
            self._check_exit(ref_price, bybit_qty)
            return
        if self.state.bot_state == BotState.TRADING:
            self._try_close(ref_price, bybit_qty)
            return
        if self.state.bot_state == BotState.COOLDOWN:
            log.debug("COOLDOWN, no action")
            return

        if not self._can_trade(bybit_qty):
            return

        prediction = self.predictor.predict(feature_row, timestamp=feature_row.name)
        self.state.last_prediction = prediction

        sig = "TRADE" if prediction.trade_signal else "no-trade"
        dir_label = {1: "LONG", -1: "SHORT", 0: "-"}[prediction.direction]
        log.info(f"[{now.strftime('%H:%M:%S')}] mid=${ref_price:.2f} "
                 f"vol={prediction.vol_proba:.3f} dir={prediction.dir_proba:.3f} "
                 f"signal={sig} {dir_label}")

        if prediction.trade_signal:
            self.state.signals_today += 1
            self._open_position(prediction, ref_price)

    def _heartbeat(self):
        try:
            heartbeat_mod.write(symbol=CFG.symbol, events_since_last=1,
                                connection_status="alive")
        except Exception:
            pass  # heartbeat errors must not crash the loop

    def run(self, max_minutes: Optional[int] = None):
        log.info("=" * 70)
        log.info("Mark19 Live Trading Bot starting")
        log.info(f"  mode={self.mode}  max_minutes={max_minutes}")
        log.info("=" * 70)

        # Recover today's daily counters from Bybit (LIVE_* only)
        self._recover_today_state()

        # Notify start
        self.notifier.bot_started(
            mode=self.mode,
            capital=self.virtual_equity_krw,
            leverage=self.leverage,
            model=self.predictor.model_version,
        )
        # Dashboard initial post
        self._safe_dashboard_update(force=True)

        start = datetime.now(timezone.utc)
        cycles = 0
        try:
            while True:
                cycle_start = datetime.now(timezone.utc)
                self._tick()
                self._heartbeat()
                cycles += 1

                if self._emergency_stop:
                    log.error("Emergency stop flag set, exiting loop")
                    break

                if max_minutes is not None:
                    elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 60
                    if elapsed >= max_minutes:
                        log.info(f"max_minutes={max_minutes} reached, stopping")
                        break

                cycle_elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_s = max(CFG.cadence_minutes * 60 - cycle_elapsed, 1)
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            log.info("Shutdown requested (Ctrl+C)")
        except Exception as e:
            log.error(f"Fatal error: {e}", exc_info=True)

        log.info("=" * 70)
        log.info(f"Bot stopped after {cycles} cycles")
        log.info(f"  state={self.state.bot_state.value}")
        log.info(f"  trades_today={self.state.trades_today}  daily_pnl={self.state.daily_pnl_pct*100:+.3f}%")
        log.info("=" * 70)

        # Notify stop
        self.notifier.bot_stopped(
            cycles=cycles,
            trades=self.state.trades_today,
            daily_pnl_krw=self.state.daily_pnl_pct * self.virtual_equity_krw,
        )
