"""
Discord webhook notifier — shadow + live alerts.

NEVER raises into caller — bot survives notify failures.

Levels:
  info     → 📊 blue
  warning  → ⚠️ amber
  critical → 🚨 red (also @here mention)

Webhook URL loaded from (priority):
  1. env DISCORD_WEBHOOK
  2. live_bot/.env DISCORD_WEBHOOK
  3. None → notify silently no-ops (logged)

Rate-limit guard: trivial token-bucket (max 5 messages / 2 sec).
"""
from __future__ import annotations
import json, logging, os, time, threading, urllib.request, urllib.error
from pathlib import Path
from typing import Optional


_log = logging.getLogger("discord_notify")
_lock = threading.Lock()
_send_times: list[float] = []   # for rate-limit window
_cached_webhook: Optional[str] = None
_loaded = False


def _load_webhook() -> Optional[str]:
    """Load webhook URL. Cached after first call. Returns None if unavailable."""
    global _cached_webhook, _loaded
    if _loaded:
        return _cached_webhook
    url = os.environ.get("DISCORD_WEBHOOK")
    if not url:
        env_path = Path("/Users/mark/Desktop/Mark/mark19/live_bot/.env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("DISCORD_WEBHOOK=") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val: url = val; break
    if url:
        # Sanity: must look like a Discord webhook
        if "discord.com/api/webhooks/" not in url:
            _log.warning("DISCORD_WEBHOOK does not look like Discord URL — disabled")
            url = None
    _cached_webhook = url
    _loaded = True
    return _cached_webhook


def _respect_rate_limit():
    """Token-bucket: 5 messages / 2 seconds (Discord per-webhook limit)."""
    now = time.time()
    _send_times[:] = [t for t in _send_times if now - t < 2.0]
    if len(_send_times) >= 5:
        sleep_for = 2.0 - (now - _send_times[0]) + 0.05
        time.sleep(max(0, sleep_for))
        _send_times[:] = [t for t in _send_times if time.time() - t < 2.0]
    _send_times.append(time.time())


# Discord embed colors (decimal)
COLORS = {"info": 0x3498DB, "warning": 0xE67E22, "critical": 0xE74C3C}
ICONS = {"info": "📊", "warning": "⚠️", "critical": "🚨"}


def send(title: str, body: str = "", level: str = "info",
         fields: Optional[list[dict]] = None, mention_here: Optional[bool] = None) -> bool:
    """Send a Discord message. Never raises. Returns True on success."""
    url = _load_webhook()
    if not url:
        _log.info(f"[discord_notify] (no webhook) {level.upper()}: {title}")
        return False

    if mention_here is None:
        mention_here = (level == "critical")

    payload = {
        "username": "mark19-shadow",
        "content": "@here" if mention_here else "",
        "embeds": [{
            "title": f"{ICONS.get(level, '')} {title}",
            "description": body[:4000] if body else "",
            "color": COLORS.get(level, 0x95A5A6),
            "fields": fields or [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }],
    }
    body_str = json.dumps(payload).encode()
    try:
        with _lock:
            _respect_rate_limit()
        req = urllib.request.Request(url, data=body_str,
                                       headers={"Content-Type": "application/json",
                                                "User-Agent": "mark19-shadow-bot/1.0"},
                                       method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            code = r.getcode()
            if code in (200, 204):
                return True
            _log.warning(f"[discord_notify] HTTP {code} for '{title}'")
            return False
    except urllib.error.HTTPError as e:
        # 429 = rate limited; back off and try once
        if e.code == 429:
            _log.warning(f"[discord_notify] 429 rate limit — backing off")
            time.sleep(2)
            try:
                req = urllib.request.Request(url, data=body_str,
                                               headers={"Content-Type": "application/json"},
                                               method="POST")
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.getcode() in (200, 204)
            except Exception as e2:
                _log.warning(f"[discord_notify] retry failed: {e2}")
                return False
        _log.warning(f"[discord_notify] HTTPError {e.code}: {e}")
        return False
    except Exception as e:
        _log.warning(f"[discord_notify] {type(e).__name__}: {e}")
        return False


# Convenience wrappers
def info(title, body="", fields=None):
    return send(title, body, "info", fields, mention_here=False)


def warning(title, body="", fields=None):
    return send(title, body, "warning", fields, mention_here=False)


def critical(title, body="", fields=None):
    return send(title, body, "critical", fields, mention_here=True)


if __name__ == "__main__":
    # Smoke test: send 3 messages at each level
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    url = _load_webhook()
    if not url:
        print("No DISCORD_WEBHOOK found — set env var or live_bot/.env")
        raise SystemExit(1)
    # Mask URL for output
    parts = url.split("/")
    masked = "/".join(parts[:-1] + [parts[-1][:4] + "..." + parts[-1][-4:]])
    print(f"Webhook (masked): {masked}")

    print("Sending info...")
    ok = info("Discord notifier smoke test", "If you see this, info works.")
    print(f"  → {'sent' if ok else 'FAILED'}")
    print("Sending warning...")
    ok = warning("Test warning", "Sample warning body.",
                  fields=[{"name": "WS reconnects", "value": "3/5 today", "inline": True}])
    print(f"  → {'sent' if ok else 'FAILED'}")
    print("Sending critical (no actual @here in test)...")
    ok = send("Test critical", "Sample critical body.", "critical", mention_here=False)
    print(f"  → {'sent' if ok else 'FAILED'}")
