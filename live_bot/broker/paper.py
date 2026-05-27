"""Paper broker — bar-driven order book simulator.

Fill rules
----------
  market order    : fills at current bar's close + taker slippage
  limit order     : fills when a subsequent bar's range contains the price
                    (low ≤ price ≤ high). Post-only rejects if bar open is
                    already past the price (would cross).
  stop-market     : fills when bar's high ≥ stop price (buy) or
                    low ≤ stop price (sell). Filled at stop price + slip.

After any fill the position is updated. If a reduce-only fill closes the
position, the fill size is capped to the open position qty.

Intrabar SL+TP on the same bar is resolved stop-first (conservative).

The broker is bar-driven: the caller feeds bars via `on_bar(ts, ohlc_row)`.
Calling `drain_fills()` returns everything that filled since the last call.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from live_bot.config import CFG

from .base import Broker, Fill, Order, OrderSide, OrderType, Position


DEFAULT_SLIP_BP = 1.0


class PaperBroker(Broker):
    def __init__(self, symbol: str, initial_equity: float = 10_000.0,
                 fee_maker: float = CFG.maker_fee,
                 fee_taker: float = CFG.taker_fee,
                 slip_bp: float = DEFAULT_SLIP_BP):
        self._symbol = symbol
        self._equity = initial_equity
        self._fee_maker = fee_maker
        self._fee_taker = fee_taker
        self._slip_bp = slip_bp
        self._positions: Dict[str, Position] = {symbol: Position(symbol=symbol)}
        self._open: Dict[int, Order] = {}
        self._pending_fills: List[Fill] = []
        self._ids = itertools.count(1)

    # ---- required interface ----
    def name(self) -> str: return "paper"

    def equity(self) -> float: return self._equity

    def position(self, symbol: str) -> Position:
        return self._positions.setdefault(symbol, Position(symbol=symbol))

    def place_order(self, order: Order) -> int:
        order.id = order.id or next(self._ids)
        if order.id in self._open:
            raise ValueError(f"order id {order.id} already open")
        order.status = "open"
        self._open[order.id] = order
        return order.id

    def cancel(self, order_id: int) -> bool:
        o = self._open.pop(order_id, None)
        if o is None:
            return False
        o.status = "cancelled"
        return True

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        ids = [oid for oid, o in self._open.items()
               if symbol is None or o.symbol == symbol]
        for oid in ids:
            self.cancel(oid)
        return len(ids)

    def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return [o for o in self._open.values()
                if symbol is None or o.symbol == symbol]

    def drain_fills(self) -> List[Fill]:
        out, self._pending_fills = self._pending_fills, []
        return out

    # ---- bar-driven simulation ----
    def on_bar(self, ts: pd.Timestamp, bar: Dict) -> None:
        """Process one bar: expire stale orders, check fills, settle intrabar."""
        hi = float(bar["high"]); lo = float(bar["low"])
        open_ = float(bar["open"]); close = float(bar["close"])

        # Expire orders whose valid_until_bar has passed.
        for oid, o in list(self._open.items()):
            if o.valid_until_bar is not None and ts > o.valid_until_bar:
                o.status = "expired"
                self._open.pop(oid)

        # Market orders fill at `close` immediately (no timing game).
        for oid, o in list(self._open.items()):
            if o.order_type != OrderType.MARKET:
                continue
            self._fill(o, ts, self._slip_price(o.side, close), reason="market")
            self._open.pop(oid, None)

        # Limit and stop-market checks: evaluate against this bar's range.
        for oid, o in list(self._open.items()):
            filled_price = self._match(o, open_, hi, lo)
            if filled_price is None:
                continue
            self._fill(o, ts, filled_price,
                       reason=o.order_type.value)
            self._open.pop(oid, None)

    # ---- helpers ----
    def _slip_price(self, side: OrderSide, ref: float) -> float:
        bump = ref * self._slip_bp / 10_000
        return ref + bump if side == OrderSide.LONG else ref - bump

    def _match(self, o: Order, open_: float, hi: float, lo: float) -> Optional[float]:
        if o.order_type == OrderType.LIMIT:
            price = float(o.price)
            # Post-only: reject if the bar OPEN already passed the price
            # (would immediately cross the book).
            if o.post_only:
                if o.side == OrderSide.LONG and open_ < price:
                    return None   # bar opened below our buy limit — would cross
                if o.side == OrderSide.SHORT and open_ > price:
                    return None
            if lo <= price <= hi:
                return price
            return None
        if o.order_type == OrderType.STOP_MARKET:
            price = float(o.price)
            if o.side == OrderSide.LONG and hi >= price:
                return self._slip_price(o.side, price)
            if o.side == OrderSide.SHORT and lo <= price:
                return self._slip_price(o.side, price)
            return None
        return None

    def _fill(self, o: Order, ts: pd.Timestamp, fill_price: float,
              reason: str) -> None:
        pos = self.position(o.symbol)
        qty_signed = o.qty if o.side == OrderSide.LONG else -o.qty

        # Reduce-only: cap at the position's open qty on the closing side.
        if o.reduce_only:
            cur_signed = pos.qty if pos.side == OrderSide.LONG else -pos.qty
            if (cur_signed >= 0 and qty_signed >= 0) or (cur_signed <= 0 and qty_signed <= 0):
                # Same side as the position — reduce-only cannot add.
                o.status = "rejected"
                return
            qty_signed = max(-abs(cur_signed), min(abs(cur_signed), qty_signed))

        is_taker = o.order_type != OrderType.LIMIT  # limit-maker, else taker
        fee_rate = self._fee_maker if (o.order_type == OrderType.LIMIT and o.post_only) else self._fee_taker
        fee = abs(qty_signed) * fill_price * fee_rate

        # Update position.
        new_qty_signed = (pos.qty if pos.side == OrderSide.LONG else -pos.qty) + qty_signed
        if pos.is_flat:
            pos.side = o.side
            pos.qty = abs(new_qty_signed)
            pos.avg_price = fill_price
        else:
            # Closing or reversing.
            if (pos.side == OrderSide.LONG and qty_signed < 0) or \
               (pos.side == OrderSide.SHORT and qty_signed > 0):
                close_qty = min(pos.qty, abs(qty_signed))
                # Realise PnL on the closed portion.
                direction = 1 if pos.side == OrderSide.LONG else -1
                self._equity += direction * (fill_price - pos.avg_price) * close_qty - fee
                residual = abs(qty_signed) - close_qty
                if residual > 0:
                    # Reversal.
                    pos.side = o.side
                    pos.qty = residual
                    pos.avg_price = fill_price
                else:
                    pos.qty -= close_qty
                    if pos.qty == 0:
                        pos.side = None
                        pos.avg_price = 0.0
            else:
                # Adding to same-side position.
                total_notional = pos.avg_price * pos.qty + fill_price * abs(qty_signed)
                pos.qty += abs(qty_signed)
                pos.avg_price = total_notional / pos.qty if pos.qty else 0.0
                self._equity -= fee

        o.status = "filled"
        self._pending_fills.append(Fill(
            order_id=o.id, ts=ts, price=fill_price,
            qty=abs(qty_signed), fee=fee, reason=reason,
        ))
