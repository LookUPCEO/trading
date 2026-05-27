"""Mark19 Live Trading Config.

Forked from trading_bot_v13/config.py.
Mark19 specific: ETHUSDT, 3x leverage, 30만원 capital, 시도 17 model.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ============================================================
    # Bybit API
    # ============================================================
    api_key: str = field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BYBIT_API_SECRET", ""))
    testnet: bool = field(default_factory=lambda: os.getenv("BYBIT_TESTNET", "false").lower() == "true")

    # API endpoints (v5)
    base_url: str = field(default_factory=lambda:
        "https://api-testnet.bybit.com" if os.getenv("BYBIT_TESTNET", "false").lower() == "true"
        else "https://api.bybit.com"
    )
    recv_window: int = 5000

    # ============================================================
    # Trading parameters (Mark19)
    # ============================================================
    symbol: str = "ETHUSDT"
    category: str = "linear"  # USDT perpetual

    # Leverage (mode dependent)
    leverage_live: int = 3  # 정상 운영
    leverage_small: int = 3  # LIVE_SMALL 운영 leverage (사용자 결정)

    # Position
    position_mode: str = "MergedSingle"  # one-way mode (Bybit v5)

    # ============================================================
    # Fees (Bybit USDT perpetual VIP 0)
    # ============================================================
    fee_taker: float = 0.00055   # 0.055% per side
    fee_maker: float = -0.0002   # -0.02% rebate per side

    # Mixed strategy: Taker entry + Maker exit
    # Round trip: 0.055% + (-0.02%) = 0.035%
    fee_mixed_estimate: float = 0.00035  # 0.035% round trip

    # ============================================================
    # Risk management
    # ============================================================
    max_daily_loss_pct: float = 0.05      # -5% → bot 정지 (다음 날까지)
    max_weekly_loss_pct: float = 0.15     # -15% → 일주일 정지
    max_consecutive_losses: int = 5        # 5연속 손실 → cooldown
    cooldown_bars: int = 0                 # 1h cycle 자체가 cooldown

    # Sizing (시도 17 fixed sizing)
    sizing_mode: str = "fixed"  # fixed | risk_per_trade
    fixed_size_pct: float = 1.0  # 자본의 100% (leverage 후)

    # ============================================================
    # Trading cycle (시도 17)
    # ============================================================
    cadence_minutes: int = 1  # 1분마다 신호 체크
    lockout_minutes: int = 60  # 1h cycle (no overlap)
    exit_target_minutes: int = 60  # 60분 후 청산 시도
    exit_max_wait_minutes: int = 30  # Drift policy (Phase 3): 1min 단위 cancel/replace 30회까지

    # ============================================================
    # Drift policy params (sido28b sp0.5_sz0.05 best)
    # ============================================================
    # Place limit at best_bid - DRIFT_OFFSET_BPS / best_ask + DRIFT_OFFSET_BPS
    # 0.5 bp = 0.005% inside best (passive maker, queue back of next price level)
    drift_offset_bps: float = 0.5     # bp from best_bid/ask (positive = passive)
    drift_min_replace_move_bps: float = 0.5  # don't replace unless price moved this much
    drift_replace_cooldown_sec: int = 30      # min seconds between cancel/replace

    # Trading thresholds (시도 17)
    vol_threshold: float = 0.6  # vol_proba > 0.6
    dir_threshold: float = 0.65  # |dir_proba - 0.5| > 0.15 (= dir_proba > 0.65 or < 0.35)

    # ============================================================
    # Model
    # ============================================================
    model_path: str = "models/mark17_v1.joblib"
    model_version: str = "mark17_v1"

    # ============================================================
    # Capital (mode dependent, 사용자 입력 시)
    # ============================================================
    capital_live_krw: int = 300000     # 30만원
    capital_small_krw: int = 282599    # 실 잔고 ($204.78 USDT, 4/28 22:17 갱신)

    # USDT 환산 (대략, 실제는 운영 시 wallet equity 사용)
    krw_per_usdt: float = 1380  # 환율 (변동)

    # ============================================================
    # Discord 알림
    # ============================================================
    discord_webhook: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK", ""))
    discord_webhook_dashboard: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_DASHBOARD", ""))
    discord_alerts_enabled: bool = True

    # Alert thresholds
    alert_loss_pct: float = 0.02  # -2% 손실 시 alert
    alert_drawdown_pct: float = 0.03  # -3% drawdown 시 alert

    # ============================================================
    # Paths
    # ============================================================
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.absolute())
    state_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "live_bot_state")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "live_bot_logs")

    # Collector data (Mark19 의 5 PIDs)
    collector_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")

    def __post_init__(self):
        # Ensure dirs exist
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # v13 backward-compat aliases (used by live_bot/broker/paper.py, risk.py, supervisor/core.py)
        self.maker_fee = self.fee_maker
        self.taker_fee = self.fee_taker
        self.max_daily_loss = self.max_daily_loss_pct
        self.risk_per_trade = 0.01           # unused (sizing_mode='fixed') but referenced by risk.position_size
        self.leverage = self.leverage_live   # default; runtime mode-switch via get_leverage_for_mode()

    def validate(self) -> list:
        """Validate config, return list of errors."""
        errors = []
        if not self.api_key:
            errors.append("BYBIT_API_KEY not set")
        if not self.api_secret:
            errors.append("BYBIT_API_SECRET not set")
        if self.discord_alerts_enabled and not self.discord_webhook:
            errors.append("DISCORD_WEBHOOK not set (or set discord_alerts_enabled=False)")
        return errors


# Global config instance
CFG = Config()


# Mode definitions (state_store/db.py 참고)
class Mode:
    OFF = "OFF"
    PAPER = "PAPER"
    LIVE_SHADOW = "LIVE_SHADOW"
    LIVE_SMALL_CAPITAL = "LIVE_SMALL_CAPITAL"
    LIVE = "LIVE"


def get_leverage_for_mode(mode: str) -> int:
    """Mode 별 leverage."""
    if mode == Mode.LIVE_SMALL_CAPITAL:
        return CFG.leverage_small  # 1x
    elif mode == Mode.LIVE:
        return CFG.leverage_live  # 3x
    else:
        return 1  # PAPER, SHADOW


def get_capital_for_mode(mode: str) -> int:
    """Mode 별 capital (KRW)."""
    if mode == Mode.LIVE_SMALL_CAPITAL:
        return CFG.capital_small_krw  # 10만원
    elif mode == Mode.LIVE:
        return CFG.capital_live_krw  # 30만원
    else:
        return 0  # PAPER, SHADOW (no real capital)
