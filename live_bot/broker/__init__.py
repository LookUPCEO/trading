"""Broker abstraction.

Concrete backends:
  - PaperBroker : in-process simulator; consumes bars, produces deterministic fills.
  - LiveBroker  : thin shim around execution.BybitClient; only instantiated when
                  the operator has passed every safety interlock.

Callers should never import BybitClient directly — go through `get_broker(mode)`.
"""
from __future__ import annotations

from .base import Broker, OrderSide, OrderType, Order, Fill, Position
from .paper import PaperBroker
from .factory import get_broker

__all__ = [
    "Broker", "OrderSide", "OrderType", "Order", "Fill", "Position",
    "PaperBroker", "get_broker",
]
