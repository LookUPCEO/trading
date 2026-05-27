"""Discord Notifier for Mark19 Live Trading Bot.

All send attempts are wrapped in try/except — a notifier failure must never
crash the bot loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

KST = timezone(timedelta(hours=9))

import requests

log = logging.getLogger(__name__)


COLOR_GREEN = 0x00FF00
COLOR_RED = 0xFF0000
COLOR_YELLOW = 0xFFAA00
COLOR_BLUE = 0x0099FF
COLOR_GRAY = 0x808080


@dataclass
class DiscordNotifier:
    """Discord webhook notifier."""
    webhook_url: str = ""
    enabled: bool = True

    def __post_init__(self):
        if not self.webhook_url:
            self.enabled = False
            log.warning("Discord webhook not set, notifier disabled")

    def _send(self, content: str = "", embed: Optional[dict] = None) -> bool:
        if not self.enabled:
            log.debug(f"Notifier disabled, skipping: {(content or 'embed')[:50]}")
            return False

        payload = {}
        if content:
            payload["content"] = content
        if embed:
            payload["embeds"] = [embed]
        if not payload:
            return False

        try:
            r = requests.post(self.webhook_url, json=payload, timeout=5)
            if r.status_code in (200, 204):
                return True
            log.warning(f"Discord webhook returned {r.status_code}: {r.text[:200]}")
            return False
        except requests.RequestException as e:
            log.warning(f"Discord webhook failed: {e}")
            return False
        except Exception as e:
            log.error(f"Discord notifier unexpected error: {e}")
            return False

    # ---- Bot lifecycle ----
    def bot_started(self, mode: str, capital: int, leverage: int, model: str):
        embed = {
            "title": "🟢 Mark19 봇 시작 (Bot Started)",
            "color": COLOR_GREEN,
            "fields": [
                {"name": "Mode (모드)", "value": mode, "inline": True},
                {"name": "Capital (자본)", "value": f"₩{capital:,}", "inline": True},
                {"name": "Leverage (레버리지)", "value": f"{leverage}x", "inline": True},
                {"name": "Model (모델)", "value": model, "inline": False},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send(embed=embed)

    def bot_stopped(self, cycles: int, trades: int, daily_pnl_krw: float):
        embed = {
            "title": "⚪ Mark19 봇 정지 (Bot Stopped)",
            "color": COLOR_GRAY,
            "fields": [
                {"name": "Cycles (사이클)", "value": str(cycles), "inline": True},
                {"name": "Trades (거래)", "value": str(trades), "inline": True},
                {"name": "Daily PnL (일일 손익)", "value": f"{daily_pnl_krw:+,.0f} KRW", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send(embed=embed)

    # ---- Trade events ----
    def position_opened(self, direction: str, qty: float, entry_price: float,
                        vol_proba: float, dir_proba: float, notional_usdt: float):
        d = direction.upper()
        d_kor = "롱" if d == "LONG" else ("숏" if d == "SHORT" else "?")
        emoji = "📈" if d == "LONG" else "📉"
        embed = {
            "title": f"{emoji} 포지션 진입 (Position Opened): {d} ({d_kor})",
            "color": COLOR_BLUE,
            "fields": [
                {"name": "Qty (수량)", "value": f"{qty} ETH", "inline": True},
                {"name": "Entry (진입가)", "value": f"${entry_price:,.2f}", "inline": True},
                {"name": "Notional (명목)", "value": f"${notional_usdt:,.0f}", "inline": True},
                {"name": "Vol Proba (변동성)", "value": f"{vol_proba:.3f}", "inline": True},
                {"name": "Dir Proba (방향)", "value": f"{dir_proba:.3f}", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send(embed=embed)

    def position_closed(self, direction: str, entry_price: float, exit_price: float,
                        raw_pnl_pct: float, net_pnl_pct: float, net_pnl_krw: float,
                        fee_pct: float, filled_at_maker: bool, duration_min: float):
        d = direction.upper()
        d_kor = "롱" if d == "LONG" else ("숏" if d == "SHORT" else "?")
        emoji = "✅" if net_pnl_pct > 0 else "❌"
        color = COLOR_GREEN if net_pnl_pct > 0 else COLOR_RED
        exit_type = "Maker (메이커)" if filled_at_maker else "Taker (테이커, fallback)"
        embed = {
            "title": f"{emoji} 포지션 청산 (Position Closed): {d} ({d_kor})",
            "color": color,
            "fields": [
                {"name": "Entry (진입가)", "value": f"${entry_price:,.2f}", "inline": True},
                {"name": "Exit (청산가)", "value": f"${exit_price:,.2f}", "inline": True},
                {"name": "Duration (보유시간)", "value": f"{duration_min:.1f}분", "inline": True},
                {"name": "Raw PnL (원손익)", "value": f"{raw_pnl_pct*100:+.3f}%", "inline": True},
                {"name": "Fee (수수료)", "value": f"-{fee_pct*100:.3f}%", "inline": True},
                {"name": "Net PnL (순손익)", "value": f"{net_pnl_pct*100:+.3f}% ({net_pnl_krw:+,.0f} KRW)",
                 "inline": True},
                {"name": "Exit Type (청산방식)", "value": exit_type, "inline": False},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send(embed=embed)

    # ---- Risk / Errors ----
    def risk_alert(self, severity: str, message: str,
                   daily_pnl_krw: float = 0, daily_pnl_pct: float = 0):
        if severity == "critical":
            title = f"🚨 위험 (CRITICAL): {message}"
            color = COLOR_RED
        else:
            title = f"⚠️ 경고 (Warning): {message}"
            color = COLOR_YELLOW
        embed = {
            "title": title,
            "color": color,
            "fields": [
                {"name": "Daily PnL (일일 손익)",
                 "value": f"{daily_pnl_krw:+,.0f} KRW ({daily_pnl_pct*100:+.2f}%)",
                 "inline": False},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send(embed=embed)

    def error_alert(self, error_type: str, message: str):
        embed = {
            "title": f"🔴 오류 (Error): {error_type}",
            "description": f"```\n{message[:1000]}\n```",
            "color": COLOR_RED,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send(embed=embed)

    # ---- Daily summary ----
    def daily_summary(self, trades: int, wins: int, losses: int,
                      daily_pnl_krw: float, daily_pnl_pct: float, win_rate: float,
                      avg_win_pct: float, avg_loss_pct: float):
        if daily_pnl_pct > 0:
            emoji, color = "✅", COLOR_GREEN
        elif daily_pnl_pct < 0:
            emoji, color = "❌", COLOR_RED
        else:
            emoji, color = "⚪", COLOR_GRAY
        embed = {
            "title": f"{emoji} 일일 요약 (Daily Summary)",
            "color": color,
            "fields": [
                {"name": "Trades (거래)",
                 "value": f"{trades} ({wins}승 / {losses}패)", "inline": True},
                {"name": "Win Rate (승률)", "value": f"{win_rate:.1%}", "inline": True},
                {"name": "Daily PnL (일일 손익)",
                 "value": f"{daily_pnl_krw:+,.0f} KRW ({daily_pnl_pct*100:+.2f}%)",
                 "inline": False},
                {"name": "Avg Win (평균 승)", "value": f"{avg_win_pct*100:+.3f}%", "inline": True},
                {"name": "Avg Loss (평균 패)", "value": f"{avg_loss_pct*100:+.3f}%", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send(embed=embed)

    def info(self, message: str):
        self._send(content=f"ℹ️ {message}")


def get_notifier(webhook_url: Optional[str] = None) -> DiscordNotifier:
    """Return a DiscordNotifier; if webhook_url is None, use CFG.discord_webhook."""
    if webhook_url is None:
        from live_bot.config import CFG
        webhook_url = CFG.discord_webhook
    return DiscordNotifier(webhook_url=webhook_url)
