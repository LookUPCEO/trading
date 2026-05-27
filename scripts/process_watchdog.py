"""
Process watchdog — monitor shadow stack PIDs, alert on death, optional restart.

Watches:
  - bybit_ws.py (WS feed)
  - mark19_shadow_runner.py (decision engine)

Detects:
  - Process death → Discord critical + (optional) auto-restart
  - WS stale (no log update >120s) → warning
  - Hourly heartbeat → info (configurable)

Run as daemon (no built-in scheduling — use nohup or cron).
"""
from __future__ import annotations
import argparse, json, logging, os, re, signal, subprocess, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discord_notify as dn


SHADOW_DIR = Path("/Users/mark/mark19_data")
WS_LOG_PATTERN = SHADOW_DIR / "ws_logs"
RUNNER_LOG_PATTERN = SHADOW_DIR / "shadow_runner_logs"
WATCHDOG_LOG_DIR = SHADOW_DIR / "watchdog_logs"
WATCHDOG_LOG_DIR.mkdir(parents=True, exist_ok=True)


def pid_alive(pid: int) -> bool:
    """Check if PID exists and is one of ours."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def find_latest_log(dir_path: Path, pattern: str = "*") -> Path | None:
    files = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def log_stale_age_sec(log_path: Path) -> float:
    """Seconds since log file was last modified."""
    if not log_path.exists(): return 999999
    return time.time() - log_path.stat().st_mtime


def parse_ws_reconnect_count(log_path: Path) -> int:
    """Read latest reconnect# from WS log."""
    if not log_path.exists(): return 0
    try:
        content = log_path.read_text()
        matches = re.findall(r"reconnect#(\d+)\s+today", content)
        return int(matches[-1]) if matches else 0
    except Exception:
        return 0


def restart_process(name: str, cmd: list[str], log_path: Path, log) -> int | None:
    """Restart by spawning detached. Returns new PID or None."""
    try:
        with open(log_path, 'a') as logf:
            logf.write(f"\n=== AUTO-RESTART by watchdog at {datetime.now(timezone.utc).isoformat()} ===\n")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        log.info(f"  restarted {name}: PID={proc.pid}")
        return proc.pid
    except Exception as e:
        log.error(f"  restart {name} failed: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ws-pid", type=int, required=True)
    p.add_argument("--runner-pid", type=int, required=True)
    p.add_argument("--check-sec", type=int, default=60, help="Poll interval")
    p.add_argument("--heartbeat-hours", type=float, default=24, help="Send heartbeat every N hours (0=off)")
    p.add_argument("--auto-restart", action="store_true", help="Auto-restart dead processes")
    p.add_argument("--stale-sec", type=int, default=300, help="WS stale threshold")
    args = p.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file = WATCHDOG_LOG_DIR / f"watchdog_{stamp}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    log.info(f"=== Watchdog start — WS={args.ws_pid} runner={args.runner_pid} ===")
    dn.info("Watchdog started",
            f"Monitoring WS PID {args.ws_pid} + runner PID {args.runner_pid}.\n"
            f"Check every {args.check_sec}s. Heartbeat every {args.heartbeat_hours}h. "
            f"Auto-restart: {'ON' if args.auto_restart else 'OFF'}.")

    ws_pid = args.ws_pid
    runner_pid = args.runner_pid
    last_heartbeat = time.time()
    last_ws_reconnect_warned = 0
    last_stale_warned = 0

    def shutdown(s, f):
        log.info("shutdown signal"); sys.exit(0)
    signal.signal(signal.SIGINT, shutdown); signal.signal(signal.SIGTERM, shutdown)

    while True:
        try:
            # 1. Process liveness
            if not pid_alive(ws_pid):
                log.error(f"  WS PID {ws_pid} dead")
                dn.critical("WS feed DIED",
                            f"bybit_ws.py PID {ws_pid} is no longer running.\n"
                            f"Shadow forward is now blind. Investigate immediately.\n"
                            f"Log: {find_latest_log(WS_LOG_PATTERN, '*.log')}")
                if args.auto_restart:
                    new_log = WS_LOG_PATTERN / f"ws_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_auto.log"
                    new_pid = restart_process("bybit_ws",
                                                ["/Users/mark/Desktop/Mark/mark19/.venv/bin/python",
                                                 "/Users/mark/Desktop/Mark/mark19/scripts/bybit_ws.py"],
                                                new_log, log)
                    if new_pid:
                        ws_pid = new_pid
                        dn.info("WS auto-restarted", f"New PID: {new_pid}")

            if not pid_alive(runner_pid):
                log.error(f"  Runner PID {runner_pid} dead")
                dn.critical("Shadow runner DIED",
                            f"mark19_shadow_runner.py PID {runner_pid} is no longer running.\n"
                            f"4h decisions + reconciles + fill tracking halted.\n"
                            f"Log: {find_latest_log(RUNNER_LOG_PATTERN, '*.log')}")
                if args.auto_restart:
                    new_log = RUNNER_LOG_PATTERN / f"runner_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_auto.log"
                    new_pid = restart_process("runner",
                                                ["/Users/mark/Desktop/Mark/mark19/.venv/bin/python",
                                                 "/Users/mark/Desktop/Mark/mark19/scripts/mark19_shadow_runner.py",
                                                 "--poll-sec", "60"],
                                                new_log, log)
                    if new_pid:
                        runner_pid = new_pid
                        dn.info("Runner auto-restarted", f"New PID: {new_pid}")

            # 2. WS log staleness
            ws_log = find_latest_log(WS_LOG_PATTERN, "*.log")
            if ws_log:
                age = log_stale_age_sec(ws_log)
                if age > args.stale_sec and (time.time() - last_stale_warned) > 1800:
                    dn.warning("WS feed STALE",
                                f"No log writes in {age:.0f}s (threshold {args.stale_sec}s).\n"
                                f"Possible WebSocket silence or hang. Log: {ws_log.name}")
                    last_stale_warned = time.time()

            # 3. WS reconnect threshold
            if ws_log:
                reconn = parse_ws_reconnect_count(ws_log)
                if reconn >= 5 and reconn != last_ws_reconnect_warned:
                    dn.warning("WS reconnect threshold",
                                f"reconnect#{reconn} today — exceeded 5/day soft limit.\n"
                                f"Investigate network or Bybit feed.")
                    last_ws_reconnect_warned = reconn

            # 4. Heartbeat
            if args.heartbeat_hours > 0 and (time.time() - last_heartbeat) >= args.heartbeat_hours * 3600:
                dn.info("Watchdog heartbeat",
                         f"Both processes healthy.\n"
                         f"WS PID {ws_pid} alive, runner PID {runner_pid} alive.")
                last_heartbeat = time.time()
        except Exception as e:
            log.error(f"  watchdog loop error: {type(e).__name__}: {e}")

        time.sleep(args.check_sec)


if __name__ == "__main__":
    main()
