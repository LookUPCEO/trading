"""Strategy supervisor — auto-gate ON/OFF based on rolling PF.

Public entry points:
  evaluate(db, strategy_row, now) -> TransitionDecision
  run_once(db, now=None)          -> list[TransitionDecision]
  main()                           -> CLI / cron entry point
"""
from .core import TransitionDecision, evaluate, run_once

__all__ = ["TransitionDecision", "evaluate", "run_once"]
