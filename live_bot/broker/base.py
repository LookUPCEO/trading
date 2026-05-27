"""Broker interface.

Any concrete broker (paper or live) implements the same surface. The simulator
in `backtest.py` historically had its own fill logic; going forward the paper
broker IS the canonical fill engine for both paper and backtest, so we can
guarantee "same code path" between paper and live execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class OrderSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"


@dataclass
class Order:
    id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    price: Optional[float] = None           # limit price / stop trigger
    reduce_only: bool = False
    post_only: bool = False
    valid_until_bar: Optional[pd.Timestamp] = None
    status: str = "open"                    # open | filled | cancelled | expired
    link_id: Optional[str] = None


@dataclass
class Fill:
    order_id: int
    ts: pd.Timestamp
    price: float
    qty: float
    fee: float
    reason: str = ""                        # free-form ("sl", "tp", "market", ...)


@dataclass
class Position:
    symbol: str
    side: Optional[OrderSide] = None        # None = flat
    qty: float = 0.0
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0.0


class Broker(ABC):
    # ---- connection ----
    @abstractmethod
    def name(self) -> str: ...

    # ---- account ----
    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def position(self, symbol: str) -> Position: ...

    # ---- order management ----
    @abstractmethod
    def place_order(self, order: Order) -> int:
        """Submit an order. Returns the broker-assigned order id."""

    @abstractmethod
    def cancel(self, order_id: int) -> bool: ...

    @abstractmethod
    def cancel_all(self, symbol: Optional[str] = None) -> int: ...

    @abstractmethod
    def open_orders(self, symbol: Optional[str] = None) -> List[Order]: ...

    # ---- fill queue ----
    @abstractmethod
    def drain_fills(self) -> List[Fill]:
        """Return fills since the last call. May be empty."""
