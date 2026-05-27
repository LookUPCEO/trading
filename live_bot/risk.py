"""Risk management, position sizing, and pre-trade filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from live_bot.config import CFG


@dataclass
class RiskState:
    equity: float
    day: Optional[pd.Timestamp] = None
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_until_bar: int = -1
    open_position: bool = False

    def reset_day(self, ts: pd.Timestamp) -> None:
        self.day = ts.normalize()
        self.day_start_equity = self.equity
        self.day_pnl = 0.0

    def record_trade(self, pnl: float, bar_index: int, cfg=CFG) -> None:
        self.equity += pnl
        self.day_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= cfg.max_consecutive_losses:
                self.cooldown_until_bar = bar_index + cfg.cooldown_bars
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0


def can_trade(state: RiskState, ts: pd.Timestamp, bar_index: int, cfg=CFG) -> bool:
    """Hard filters that block new trade entries."""
    if state.open_position:
        return False
    if bar_index < state.cooldown_until_bar:
        return False
    day = ts.normalize()
    if state.day is None or day != state.day:
        state.reset_day(ts)
    if state.day_start_equity > 0:
        loss_pct = -state.day_pnl / state.day_start_equity
        if loss_pct >= cfg.max_daily_loss:
            return False
    return True


def position_size(equity: float, entry: float, stop: float, cfg=CFG) -> float:
    """Risk-normalized contract/ETH size. Returns size in ETH."""
    risk_dollars = equity * cfg.risk_per_trade
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0.0
    size = risk_dollars / per_unit_risk
    # cap by leverage allowance
    max_notional = equity * cfg.leverage
    size = min(size, max_notional / entry)
    # round to 3 decimals (Bybit ETHUSDT qty step)
    return round(size, 3)
