"""Live broker — thin shim over execution.BybitClient.

Kept intentionally minimal. All safety interlocks live in broker/factory.py —
this class cannot be instantiated directly; go through `get_broker('live', ...)`.

Not exercised in tests — live calls would hit the real API.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from live_bot.execution import BybitClient

from .base import Broker, Fill, Order, OrderSide, OrderType, Position


class LiveBroker(Broker):
    def __init__(self, symbol: str):
        self._symbol = symbol
        self._client = BybitClient()
        self._pending_fills: List[Fill] = []

    def name(self) -> str: return "live"

    def equity(self) -> float:
        return self._client.get_wallet_equity()

    def position(self, symbol: str) -> Position:
        p = self._client.get_position()
        if p is None or float(p.get("size", 0) or 0) == 0:
            return Position(symbol=symbol)
        return Position(
            symbol=symbol,
            side=OrderSide.LONG if p.get("side") == "Buy" else OrderSide.SHORT,
            qty=float(p.get("size")),
            avg_price=float(p.get("avgPrice") or p.get("entryPrice") or 0),
        )

    def place_order(self, order: Order) -> int:
        side = "long" if order.side == OrderSide.LONG else "short"
        if order.order_type == OrderType.LIMIT:
            resp = self._client.place_limit(side, order.qty, order.price,
                                            reduce_only=order.reduce_only,
                                            post_only=order.post_only,
                                            order_link_id=order.link_id)
        elif order.order_type == OrderType.STOP_MARKET:
            resp = self._client.place_stop_market(side, order.qty, order.price,
                                                  order_link_id=order.link_id)
        else:  # MARKET — uses Bybit V5 Market orderType (IOC)
            resp = self._client.place_market(side, order.qty,
                                             reduce_only=order.reduce_only,
                                             order_link_id=order.link_id)

        # Bybit V5 returns 200 OK even on business errors. Validate retCode.
        ret_code = resp.get("retCode")
        if ret_code != 0:
            ret_msg = resp.get("retMsg", "")
            raise RuntimeError(
                f"Bybit order rejected: retCode={ret_code} retMsg={ret_msg!r} "
                f"side={side} type={order.order_type.value} qty={order.qty} price={order.price}"
            )

        # Capture broker-assigned orderId. Stash on order.link_id for later
        # cancel/track; map to a non-zero int hash for Broker interface.
        result = resp.get("result", {}) or {}
        bybit_order_id = result.get("orderId", "")
        if bybit_order_id and order.link_id is None:
            order.link_id = bybit_order_id
        return abs(hash(bybit_order_id)) % (2**31) if bybit_order_id else order.id

    def cancel(self, order_id: int) -> bool:
        # Bybit works off orderLinkId; cancel_by_link is the right call.
        try:
            self._client.cancel_by_link(str(order_id))
            return True
        except Exception:
            return False

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        self._client.cancel_all()
        return -1                                   # unknown count

    def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        # Deliberately not implemented at this tier — supervisor maintains
        # its own open-order state; use the REST position endpoint instead.
        return []

    def drain_fills(self) -> List[Fill]:
        # Live fills arrive via a separate WS subscription; wiring to come
        # in M4 (the dashboard layer) or when the first strategy promotes
        # to LIVE_SHADOW. Until then this returns empty — paper mode remains
        # the canonical fill simulator.
        out, self._pending_fills = self._pending_fills, []
        return out
