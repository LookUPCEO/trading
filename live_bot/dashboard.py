"""Mark19 Live Dashboard.

Maintains a single PATCHed message in the dashboard channel.
Updates every 30 minutes + on events (force=True).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))
from typing import Optional

import requests

log = logging.getLogger(__name__)

COLOR_GREEN = 0x00FF00
COLOR_RED = 0xFF0000
COLOR_YELLOW = 0xFFAA00
COLOR_BLUE = 0x0099FF
COLOR_GRAY = 0x808080

UPDATE_INTERVAL_SEC = 600  # 10 minutes


class DashboardManager:
    """A single Discord message that's PATCHed periodically."""

    def __init__(self, webhook_url: str, state_dir: Optional[Path] = None):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

        if state_dir is None:
            state_dir = Path(__file__).resolve().parent.parent / "live_bot_state"
        self.state_dir = state_dir

        self.message_id = self._load_message_id()
        self.last_update = 0.0  # Unix timestamp

    def _state_file_for(self, kst_date_iso: str) -> Path:
        return self.state_dir / f"dashboard_msg_id_{kst_date_iso}.txt"

    def _today_kst_iso(self) -> str:
        return datetime.now(KST).date().isoformat()

    def _load_message_id(self) -> Optional[str]:
        sf = self._state_file_for(self._today_kst_iso())
        if sf.exists():
            try:
                return sf.read_text().strip() or None
            except Exception:
                return None
        return None

    def _save_message_id(self, msg_id: str):
        sf = self._state_file_for(self._today_kst_iso())
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(msg_id)

    def _build_embed(self, state: dict) -> dict:
        bot_state = state.get("bot_state", "UNKNOWN")
        state_emoji = {
            "READY": "🟢", "TRADING": "📈", "EXITING": "⏳", "COOLDOWN": "⛔",
        }.get(bot_state, "⚪")

        daily_pnl_pct = state.get("daily_pnl_pct", 0)
        if daily_pnl_pct > 0.005:
            color = COLOR_GREEN
        elif daily_pnl_pct < -0.005:
            color = COLOR_RED
        else:
            color = COLOR_GRAY

        position = state.get("position")
        if position:
            d = position.get("direction", "?")
            d_kor = "롱" if d == "LONG" else ("숏" if d == "SHORT" else "?")
            position_str = (
                f"{d} ({d_kor}) {position.get('qty', 0)} ETH @ ${position.get('entry_price', 0):.2f}"
            )
        else:
            position_str = "None (없음)"

        cycles = state.get("cycles_today", 0)
        trades = state.get("trades_today", 0)
        wins = state.get("wins_today", 0)
        losses = state.get("losses_today", 0)
        win_rate = (wins / trades * 100) if trades > 0 else 0.0

        vol_proba = state.get("last_vol_proba", 0)
        dir_proba = state.get("last_dir_proba", 0.5)
        last_signal = state.get("last_signal", "no-trade")
        vol_check = "✅" if vol_proba > 0.6 else "⚪"
        dir_long = "✅" if dir_proba > 0.65 else "⚪"
        dir_short = "✅" if dir_proba < 0.35 else "⚪"
        signal_kor = "거래 (TRADE)" if last_signal == "TRADE" else "대기 중 (no-trade)"

        capital_krw = state.get("capital_krw", 0)
        wallet_usdt = state.get("wallet_equity_usdt", 0)

        today_kst = datetime.now(KST).strftime("%Y-%m-%d")
        update_kst = datetime.now(KST).strftime("%H:%M KST")

        embed = {
            "title": f"{state_emoji} Mark19 라이브 대시보드",
            "color": color,
            "fields": [
                {
                    "name": "📌 상태 (Status)",
                    "value": (
                        f"**Mode (모드):** {state.get('mode', 'unknown')}\n"
                        f"**Leverage (레버리지):** {state.get('leverage', 1)}x\n"
                        f"**Capital (자본):** ₩{capital_krw:,} (${wallet_usdt:.2f})\n"
                        f"**Position (포지션):** {position_str}"
                    ),
                    "inline": False,
                },
                {
                    "name": f"📊 오늘 (Today, {today_kst})",
                    "value": (
                        f"**Cycles (사이클):** {cycles} / 1440\n"
                        f"**Signals (시그널):** {state.get('signals_today', 0)}\n"
                        f"**Trades (거래):** {trades} ({wins}승 / {losses}패)\n"
                        f"**Win Rate (승률):** {win_rate:.1f}%\n"
                        f"**PnL (손익):** {daily_pnl_pct*100:+.2f}% (₩{state.get('daily_pnl_krw', 0):+,.0f})"
                    ),
                    "inline": True,
                },
                {
                    "name": "🔮 최근 예측 (Last Prediction)",
                    "value": (
                        f"**Vol (변동성):** {vol_proba:.3f} {vol_check} (>0.6)\n"
                        f"**Dir (방향):** {dir_proba:.3f} (롱 {dir_long}, 숏 {dir_short})\n"
                        f"**Signal (시그널):** {signal_kor}"
                    ),
                    "inline": True,
                },
            ],
            "footer": {"text": f"Model: {state.get('model_version', 'mark17_v1')} | 업데이트 {update_kst}"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return embed

    def update(self, state: dict, force: bool = False) -> bool:
        if not self.enabled:
            return False

        now = time.time()
        if not force and (now - self.last_update) < UPDATE_INTERVAL_SEC:
            return False

        embed = self._build_embed(state)
        payload = {"embeds": [embed]}

        try:
            if self.message_id:
                url = f"{self.webhook_url}/messages/{self.message_id}"
                r = requests.patch(url, json=payload, timeout=5)
                if r.status_code in (200, 204):
                    self.last_update = now
                    return True
                if r.status_code == 404:
                    log.warning("Dashboard message not found (deleted?), creating new")
                    self.message_id = None
                    return self.update(state, force=force)
                log.warning(f"Dashboard PATCH failed: {r.status_code} {r.text[:200]}")
                return False

            url = f"{self.webhook_url}?wait=true"
            r = requests.post(url, json=payload, timeout=5)
            if r.status_code in (200, 204):
                data = r.json()
                self.message_id = data.get("id")
                if self.message_id:
                    self._save_message_id(self.message_id)
                self.last_update = now
                return True
            log.warning(f"Dashboard POST failed: {r.status_code} {r.text[:200]}")
            return False
        except requests.RequestException as e:
            log.warning(f"Dashboard webhook failed: {e}")
            return False
        except Exception as e:
            log.error(f"Dashboard unexpected error: {e}")
            return False

    def reset(self):
        """Drop in-memory message_id so the next update creates a fresh
        message in today's channel. Yesterday's per-day file is preserved
        so prior days' dashboards stay readable in Discord history.
        """
        self.message_id = None


def get_dashboard_manager(webhook_url: Optional[str] = None) -> DashboardManager:
    if webhook_url is None:
        from live_bot.config import CFG
        webhook_url = CFG.discord_webhook_dashboard
    return DashboardManager(webhook_url=webhook_url)
