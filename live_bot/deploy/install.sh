#!/usr/bin/env bash
# VPS bootstrap for the liquidation collector.
# Idempotent: safe to rerun. Installs systemd units, creates venv, enables services.
#
# Usage (run once as root on a fresh Ubuntu/Debian VPS):
#     sudo bash deploy/install.sh
#
# Pre-requisites: the repo is already cloned to /opt/bot.

set -euo pipefail

BOT_DIR="${BOT_DIR:-/opt/bot}"
BOT_USER="${BOT_USER:-bot}"
SYMBOLS="${SYMBOLS:-BTCUSDT ETHUSDT}"
PY="/usr/bin/python3"

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root" >&2; exit 1
fi

command -v systemctl >/dev/null || { echo "systemd required"; exit 1; }

# 1. Create service user if missing.
if ! id "$BOT_USER" >/dev/null 2>&1; then
    echo ">> creating user $BOT_USER"
    useradd --system --home-dir "$BOT_DIR" --shell /usr/sbin/nologin "$BOT_USER"
fi

# 2. Create state / data directories and fix ownership.
install -d -o "$BOT_USER" -g "$BOT_USER" "$BOT_DIR/state"
install -d -o "$BOT_USER" -g "$BOT_USER" "$BOT_DIR/data/liquidations"
install -d -o "$BOT_USER" -g "$BOT_USER" "$BOT_DIR/data/cache"
install -d -o "$BOT_USER" -g "$BOT_USER" "$BOT_DIR/logs"
chown -R "$BOT_USER:$BOT_USER" "$BOT_DIR"

# 3. Python venv with pinned deps.
if [ ! -d "$BOT_DIR/.venv" ]; then
    echo ">> creating venv"
    sudo -u "$BOT_USER" "$PY" -m venv "$BOT_DIR/.venv"
fi
sudo -u "$BOT_USER" "$BOT_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$BOT_USER" "$BOT_DIR/.venv/bin/pip" install --quiet \
    "websocket-client>=1.6" "pandas>=2.0" "pyarrow>=15" "requests>=2.31" \
    "pytest>=8.0"

# 4. Install systemd units.
echo ">> installing systemd units"
install -m 0644 "$BOT_DIR/deploy/ws_liquidations.service" /etc/systemd/system/ws_liquidations@.service
install -m 0644 "$BOT_DIR/deploy/ws_liquidations_monitor.service" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/ws_liquidations_monitor.timer" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/cache_refresh.service" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/cache_refresh.timer" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/hourly_fetch.service" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/hourly_fetch.timer" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/daily_health.service" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/daily_health.timer" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/supervisor.service" /etc/systemd/system/
install -m 0644 "$BOT_DIR/deploy/supervisor.timer" /etc/systemd/system/

systemctl daemon-reload

# 5. Enable + start collectors (one per symbol via the template unit).
for sym in $SYMBOLS; do
    echo ">> enabling ws_liquidations@${sym}"
    systemctl enable --now "ws_liquidations@${sym}.service"
done

# 6. Enable + start the timers.
systemctl enable --now ws_liquidations_monitor.timer
systemctl enable --now cache_refresh.timer
systemctl enable --now hourly_fetch.timer
systemctl enable --now daily_health.timer
systemctl enable --now supervisor.timer

# 7. Sanity: show status.
echo ">> statuses:"
for sym in $SYMBOLS; do
    systemctl is-active --quiet "ws_liquidations@${sym}.service" \
        && echo "  ws_liquidations@${sym} : active" \
        || echo "  ws_liquidations@${sym} : NOT active (check journalctl)"
done
systemctl is-active --quiet ws_liquidations_monitor.timer \
    && echo "  heartbeat monitor timer : active" \
    || echo "  heartbeat monitor timer : NOT active"
systemctl is-active --quiet cache_refresh.timer \
    && echo "  cache refresh timer     : active" \
    || echo "  cache refresh timer     : NOT active"

echo ">> install.sh done"
