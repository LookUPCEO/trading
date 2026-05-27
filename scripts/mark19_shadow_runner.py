"""
Shadow forward runner — watches live bars and emits decisions every 4 hours.

NO REAL TRADES. Logs:
  - Decision per 4h boundary (P_up, action, limit price, features used)
  - Maker fill estimation (was the limit crossed during the 4h hold?)
  - Per-day health report (mismatch / freq / reconcile / WS / fill rate)
  - Live vs backtest feature mismatch (for the most recent bar)

Runs alongside bybit_ws.py:
  - bybit_ws.py writes 5-min bars to ~/mark19_data/bars_5min_v3_live/ETHUSDT/{date}.parquet
  - This runner polls that folder, detects new 4h boundary, builds long features,
    calls model, writes decision JSON line to ~/mark19_data/shadow_decisions/{date}.jsonl
"""
from __future__ import annotations
import argparse, glob, json, logging, os, signal, sys, time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import discord_notify as dn
except Exception:
    dn = None   # silent if module missing — bot must survive
try:
    import trade_dashboard as td
except Exception:
    td = None


SYMBOL = "ETHUSDT"
BAR_SECONDS = 300
HOLD_BARS = 48                      # 4h
SIGNAL_THRESHOLD = 0.05

LIVE_BARS_DIR = Path("/Users/mark/mark19_data/bars_5min_v3_live") / SYMBOL
DECISIONS_DIR = Path("/Users/mark/mark19_data/shadow_decisions")
HEALTH_DIR = Path("/Users/mark/mark19_data/shadow_health")
FILL_LOG_DIR = Path("/Users/mark/mark19_data/shadow_fills")
LOG_DIR = Path("/Users/mark/mark19_data/shadow_runner_logs")
RECONCILE_DIR = Path("/Users/mark/mark19_data/shadow_reconcile")
MISMATCH_DIR = Path("/Users/mark/mark19_data/shadow_feature_mismatch")
MODEL_PATH = Path("/Users/mark/mark19_data/models_prod/4h_direction_v1.joblib")
HISTORICAL_BARS_DIR = Path("/Users/mark/mark19_data/bars_5min_v3") / SYMBOL  # for warm-up

for d in (DECISIONS_DIR, HEALTH_DIR, FILL_LOG_DIR, LOG_DIR, RECONCILE_DIR, MISMATCH_DIR):
    d.mkdir(parents=True, exist_ok=True)


def compute_long_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of backtest v3 long-feature engineering. MUST stay in sync with build_intraday_bars_v3."""
    df = bars_df.copy().sort_values("bar_open_ts").reset_index(drop=True)
    SHORT_LAG = ['return_5m_bar_bp','obi5_last','micro_dev_bp_last','rv_bar_bp','ofi_proxy','tr_net_size','tr_tick_imb']
    for c in SHORT_LAG:
        if c not in df.columns: continue
        for k in [1,3,6,12]:
            df[f'{c}_lag{k}'] = df[c].shift(k)
    for N,lbl in [(12,'1h'),(48,'4h'),(288,'1d'),(2016,'7d')]:
        df[f'mom_{lbl}_bp'] = (np.log(df.mid_close) - np.log(df.mid_close.shift(N)))*10000
        mma = df.mid_close.rolling(N, min_periods=N//2).mean()
        df[f'dist_ma_{lbl}_bp'] = (df.mid_close - mma) / mma * 10000
        df[f'rv_{lbl}_bp'] = df.return_5m_bar_bp.rolling(N, min_periods=N//2).std()
    for N,lbl in [(48,'4h'),(288,'1d'),(2016,'7d')]:
        df[f'obi5_ma_{lbl}'] = df.obi5_mean.rolling(N, min_periods=N//2).mean()
        df[f'cumflow_{lbl}'] = df.tr_net_size.rolling(N, min_periods=N//2).sum()
        df[f'buyratio_{lbl}'] = df.tr_buy_ratio.rolling(N, min_periods=N//2).mean()
        df[f'spread_ma_{lbl}'] = df.spread_bp_mean.rolling(N, min_periods=N//2).mean()
    return df


def load_all_bars(live_dir: Path, hist_dir: Path, log) -> pd.DataFrame:
    """Concat historical (warm-up) + live bars, sorted by bar_open_ts."""
    hist_files = sorted(hist_dir.glob("*.parquet"))[-15:]   # last 15 days = enough for 7d rolling
    live_files = sorted(live_dir.glob("*.parquet"))
    dfs = []
    for f in hist_files:
        try: dfs.append(pd.read_parquet(f))
        except Exception as e: log.warning(f"warm-up read failed {f.name}: {e}")
    for f in live_files:
        try: dfs.append(pd.read_parquet(f))
        except Exception as e: log.warning(f"live read failed {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['bar_open_ts'] = pd.to_datetime(df['bar_open_ts'], utc=True)
    df = df.sort_values('bar_open_ts').drop_duplicates('bar_open_ts', keep='last').reset_index(drop=True)
    return df


class FillTracker:
    """Tracks pending limit orders (shadow) and estimates fill on subsequent bars."""
    def __init__(self, log):
        self.log = log
        self.pending: list[dict] = []   # [{decision_ts, side, limit_price, hold_until, ...}]

    def add(self, decision: dict):
        self.pending.append({
            **decision,
            'hold_until_ts': (datetime.fromisoformat(decision['decision_ts']) +
                              timedelta(seconds=BAR_SECONDS * HOLD_BARS)).isoformat(),
            'filled': None, 'fill_price': None, 'exit_mid': None,
        })

    def check_against_bar(self, bar: dict):
        """For each pending, see if this bar's range crossed our limit. Mark filled."""
        now = pd.to_datetime(bar['bar_close_ts'], utc=True)
        for p in self.pending:
            if p['filled'] is not None:
                continue   # already resolved
            decision_ts = pd.to_datetime(p['decision_ts'], utc=True)
            if now <= decision_ts:
                continue   # bar is before our decision
            # Maker limit cross check: long bid filled if low <= limit; short ask filled if high >= limit
            if p['action'] == 'LONG' and bar['mid_low'] <= p['limit_price']:
                p['filled'] = 'maker'
                p['fill_price'] = p['limit_price']
            elif p['action'] == 'SHORT' and bar['mid_high'] >= p['limit_price']:
                p['filled'] = 'maker'
                p['fill_price'] = p['limit_price']

    def settle_expired(self, now: datetime, persist_path: Path, current_mid: float = None):
        """For pending past hold_until, finalize and persist. Emits exit alert if td available."""
        try:
            import trade_dashboard as _td
        except Exception:
            _td = None
        still_pending = []
        for p in self.pending:
            hold_until = pd.to_datetime(p['hold_until_ts'], utc=True)
            if now >= hold_until:
                if p['filled'] is None:
                    p['filled'] = 'missed'
                with open(persist_path, 'a') as f:
                    f.write(json.dumps(p, default=str) + "\n")
                # Discord exit alert (shadow virtual)
                if _td and p.get('filled') == 'maker' and current_mid:
                    try:
                        limit = float(p.get('limit_price') or 0)
                        sign = 1 if p.get('action') == 'LONG' else -1
                        pnl_bp = (current_mid - limit) / limit * 10000 * sign - 4.0 if limit else 0
                        pnl_usd = pnl_bp / 10000 * 0.01 * current_mid   # 0.01 ETH size
                        _td.on_trade_event("exit", {
                            "side": p.get('action'),
                            "exit_price": current_mid,
                            "pnl_usd": pnl_usd,
                        }, mode="shadow")
                    except Exception: pass
            else:
                still_pending.append(p)
        self.pending = still_pending


def make_decision(model_artifact, feature_row: pd.Series, bar: dict, log) -> dict | None:
    """Call model on a single feature row. Returns decision dict or None on skip."""
    cols = model_artifact['feature_columns']
    medians = model_artifact['train_medians']
    x = []
    for c in cols:
        v = feature_row.get(c, np.nan)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            v = medians.get(c, 0.0)
        x.append(float(v))
    X = np.array([x], dtype=np.float32)
    p_up = float(model_artifact['model'].predict_proba(X)[0, 1])
    conf = abs(p_up - 0.5)
    mid = float(bar['mid_close'])
    # Estimate top-of-book bid/ask from spread (live bar only has spread mean)
    spread = float(bar['spread_bp_mean']) / 10000 * mid
    best_bid = mid - spread/2; best_ask = mid + spread/2
    if conf <= SIGNAL_THRESHOLD:
        action = "SKIP"; limit_price = None
    else:
        action = "LONG" if p_up > 0.5 else "SHORT"
        limit_price = best_bid if action == "LONG" else best_ask
    return {
        'decision_ts': datetime.now(timezone.utc).isoformat(),
        'bar_open_ts': bar['bar_open_ts'],
        'p_up': p_up, 'confidence': conf,
        'action': action, 'limit_price': float(limit_price) if limit_price else None,
        'mid_close': mid, 'best_bid': float(best_bid), 'best_ask': float(best_ask),
        # Snapshot top-5 long features used (for forensic review)
        'features': {
            'mom_1d_bp': float(feature_row.get('mom_1d_bp', float('nan'))),
            'rv_1d_bp': float(feature_row.get('rv_1d_bp', float('nan'))),
            'dist_ma_1d_bp': float(feature_row.get('dist_ma_1d_bp', float('nan'))),
            'mom_4h_bp': float(feature_row.get('mom_4h_bp', float('nan'))),
            'cumflow_1d': float(feature_row.get('cumflow_1d', float('nan'))),
        },
    }


def daily_health_report(log, reconcile_counts: dict = None) -> dict:
    """Aggregate yesterday's metrics. Run at UTC midnight.
    Five pass criteria: feature_mismatch, freq, reconcile, ws_reconnect, fill_rate."""
    yday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    dec_path = DECISIONS_DIR / f"{yday}.jsonl"
    fill_path = FILL_LOG_DIR / f"{yday}.jsonl"
    rec_path = RECONCILE_DIR / f"{yday}.jsonl"
    mismatch_path = MISMATCH_DIR / f"{yday}.json"
    decisions = []; fills = []; reconciles = []
    if dec_path.exists():
        decisions = [json.loads(l) for l in dec_path.read_text().splitlines() if l.strip()]
    if fill_path.exists():
        fills = [json.loads(l) for l in fill_path.read_text().splitlines() if l.strip()]
    if rec_path.exists():
        reconciles = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()]

    n_dec = len(decisions)
    n_actions = sum(1 for d in decisions if d.get('action') != 'SKIP')
    n_filled = sum(1 for f in fills if f.get('filled') == 'maker')
    n_resolved = sum(1 for f in fills if f.get('filled') is not None)
    fill_rate = (n_filled / n_resolved) if n_resolved else None

    n_rec_total = len(reconciles)
    n_rec_pass = sum(1 for r in reconciles if r.get('ok'))
    reconcile_pass_rate = (n_rec_pass / n_rec_total) if n_rec_total else None

    # Feature mismatch (작업 1 — populated by separate monitor, read if exists)
    mismatch = None
    if mismatch_path.exists():
        try: mismatch = json.loads(mismatch_path.read_text())
        except Exception: mismatch = {"error": "parse_fail"}

    # WS reconnect count from yesterday's ws_logs
    ws_reconnects = None
    ws_log_dir = Path("/Users/mark/mark19_data/ws_logs")
    if ws_log_dir.exists():
        max_count = 0
        for log_f in ws_log_dir.glob(f"*{yday.replace('-','')}*"):
            try:
                content = log_f.read_text()
                # Last "reconnect#N today" mention
                import re
                matches = re.findall(r"reconnect#(\d+)\s+today", content)
                if matches: max_count = max(max_count, int(matches[-1]))
            except Exception: pass
        ws_reconnects = max_count

    report = {
        'date': yday,
        # Pass criteria 1: feature mismatch
        'feature_mismatch': mismatch,
        'criterion_1_feature_mismatch_zero': mismatch is not None and mismatch.get('max_diff_within_tolerance', False) if mismatch else None,
        # Pass criteria 2: trade frequency
        'decisions_total': n_dec,
        'actions_attempted': n_actions,
        'trades_per_day': n_actions,
        'criterion_2_freq_ok': bool(n_actions and 0.5 <= n_actions <= 6),
        # Pass criteria 3: reconcile
        'reconcile_total': n_rec_total,
        'reconcile_passed': n_rec_pass,
        'reconcile_pass_rate': reconcile_pass_rate,
        'criterion_3_reconcile_pass_100': reconcile_pass_rate == 1.0 if n_rec_total else None,
        # Pass criteria 4: WS reconnects
        'ws_reconnects_today': ws_reconnects,
        'criterion_4_ws_reconnects_lt_5': ws_reconnects is not None and ws_reconnects < 5,
        # Pass criteria 5: maker fill rate
        'fills_resolved': n_resolved,
        'maker_filled': n_filled,
        'estimated_maker_fill_rate': fill_rate,
        'criterion_5_fill_rate_measured': fill_rate is not None,
    }
    out_path = HEALTH_DIR / f"{yday}_health.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info(f"  [HEALTH] daily report → {out_path.name}")
    log.info(f"    criteria: feature={report['criterion_1_feature_mismatch_zero']} freq={report['criterion_2_freq_ok']} reconcile={report['criterion_3_reconcile_pass_100']} ws={report['criterion_4_ws_reconnects_lt_5']} fill={report['criterion_5_fill_rate_measured']}")
    # Discord daily summary
    if dn:
        def m(v): return "✅" if v is True else ("❌" if v is False else "⏳")
        body = (
            f"**Day {yday} — Shadow Forward Health**\n\n"
            f"{m(report['criterion_1_feature_mismatch_zero'])} **Feature mismatch=0**: "
            f"{report.get('feature_mismatch','no data')[:80] if isinstance(report.get('feature_mismatch'),str) else 'see file'}\n"
            f"{m(report['criterion_2_freq_ok'])} **Trade freq**: {report['actions_attempted']} actions "
            f"(target 0.5-6/day; decisions total {report['decisions_total']})\n"
            f"{m(report['criterion_3_reconcile_pass_100'])} **Reconcile**: "
            f"{report.get('reconcile_passed',0)}/{report.get('reconcile_total',0)} passes\n"
            f"{m(report['criterion_4_ws_reconnects_lt_5'])} **WS reconnects**: "
            f"{report.get('ws_reconnects_today','?')}/5 today\n"
            f"{m(report['criterion_5_fill_rate_measured'])} **Maker fill rate**: "
            f"{(report['estimated_maker_fill_rate']*100):.1f}% ({report['maker_filled']}/{report['fills_resolved']}) "
            f"(38% assumed)" if report.get('estimated_maker_fill_rate') is not None else
            f"{m(False)} **Maker fill rate**: no fills resolved yet"
        )
        passed = sum(1 for k in ['criterion_1_feature_mismatch_zero','criterion_2_freq_ok',
                                   'criterion_3_reconcile_pass_100','criterion_4_ws_reconnects_lt_5',
                                   'criterion_5_fill_rate_measured'] if report.get(k) is True)
        level = "info" if passed >= 4 else ("warning" if passed >= 2 else "warning")
        dn.send(f"Shadow Day Report ({passed}/5 passed)", body, level)
    return report


def main():
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one cycle on existing bars and exit (no daemon).")
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file = LOG_DIR / f"shadow_runner_{stamp}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    log.info(f"=== mark19 shadow runner (NO TRADES, decision logging only) ===")
    log.info(f"  Symbol={SYMBOL} bar={BAR_SECONDS}s hold={HOLD_BARS}bars thr={SIGNAL_THRESHOLD}")
    log.info(f"  Model: {MODEL_PATH}")
    log.info(f"  Live bars: {LIVE_BARS_DIR}")
    log.info(f"  Decisions: {DECISIONS_DIR}")

    if not MODEL_PATH.exists():
        log.error("Model not found."); sys.exit(2)
    artifact = joblib.load(MODEL_PATH)
    log.info(f"  Model loaded: {len(artifact['feature_columns'])} features, val AUC {artifact['config']['auc_val']:.4f}")

    fill_tracker = FillTracker(log)
    last_processed_bar_idx = None

    # === Reconcile setup (작업 2) ===
    import mark19_live
    exchange = mark19_live.ExchangeAdapter("live", log)  # READ-ONLY; uses bybit_v5.fetch_state_and_map
    rail = mark19_live.RiskRail("live", log)
    reconciler = mark19_live.Reconciler(log, rail)
    internal = mark19_live.InternalBook(balance_usdt=0.0)  # initialized from first fetch
    reconcile_counts = {"checks": 0, "passes": 0, "fails": 0, "errors": 0}

    def do_reconcile():
        """Fetch live state, compare against internal book. Returns (ok, state, error)."""
        try:
            st = exchange.fetch_state()
            # Bootstrap internal book on first fetch (shadow has no positions to reconcile yet)
            if reconcile_counts["checks"] == 0:
                internal.balance_usdt = st.balance_usdt
                internal.positions = list(st.positions)
                internal.open_order_ids = {o["orderId"] for o in st.open_orders}
            ok, diffs = reconciler.check(internal, st)
            reconcile_counts["checks"] += 1
            if ok:
                reconcile_counts["passes"] += 1
            else:
                reconcile_counts["fails"] += 1
            # Persist
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            rec_path = RECONCILE_DIR / f"{today_str}.jsonl"
            with open(rec_path, 'a') as f:
                f.write(json.dumps({
                    "ts": st.ts, "ok": ok, "diffs": diffs,
                    "balance_usdt": st.balance_usdt,
                    "n_positions": len(st.positions),
                    "n_open_orders": len(st.open_orders),
                }, default=str) + "\n")
            return ok, st, None
        except Exception as e:
            reconcile_counts["errors"] += 1
            log.error(f"  [RECONCILE] fetch error: {type(e).__name__}: {e}")
            return False, None, str(e)

    def cycle_once():
        nonlocal last_processed_bar_idx
        try:
            df = load_all_bars(LIVE_BARS_DIR, HISTORICAL_BARS_DIR, log)
            if len(df) < 2016:    # need 7+ days for long features
                log.info(f"  warm-up incomplete: {len(df)} bars (need 2016+)")
                return
            feats = compute_long_features(df)
            last = feats.iloc[-1]
            last_ts = last['bar_open_ts']
            # Update fill tracker against this latest bar
            bar_dict = {
                'bar_open_ts': last['bar_open_ts'].isoformat() if hasattr(last['bar_open_ts'],'isoformat') else str(last['bar_open_ts']),
                'bar_close_ts': last['bar_close_ts'].isoformat() if hasattr(last['bar_close_ts'],'isoformat') else str(last['bar_close_ts']),
                'mid_close': float(last['mid_close']),
                'mid_low': float(last['mid_low']),
                'mid_high': float(last['mid_high']),
                'spread_bp_mean': float(last['spread_bp_mean']),
            }
            fill_tracker.check_against_bar(bar_dict)
            now = datetime.now(timezone.utc)
            today = now.strftime("%Y-%m-%d")
            fill_tracker.settle_expired(now, FILL_LOG_DIR / f"{today}.jsonl",
                                          current_mid=float(last['mid_close']))
            # 4h boundary check (every 48 5-min bars = bar_idx 0, 48, 96, 144, 192, 240, hourly UTC 00, 04, 08, 12, 16, 20)
            bar_idx_today = int(last['bar_idx']) if 'bar_idx' in last else None
            if bar_idx_today is None:
                return
            is_4h_boundary = (bar_idx_today % HOLD_BARS == 0)
            if not is_4h_boundary or bar_idx_today == last_processed_bar_idx:
                return
            # n_updates gate: skip decision if any recent bar is partial (signals stale WS)
            if 'partial' in last.index and bool(last.get('partial', False)):
                log.warning(f"  [DECISION] SKIPPED bar {bar_idx_today} flagged partial (n_updates<250) — model not fed unreliable input")
                last_processed_bar_idx = bar_idx_today
                return
            # Pre-decision reconcile (작업 2 wire)
            ok, st, err = do_reconcile()
            if not ok and err is None:
                log.warning(f"  [DECISION] BLOCKED by reconcile mismatch (halted={rail.halted})")
                # Discord critical (event-driven)
                if dn:
                    dn.critical("Reconcile MISMATCH — runner halted",
                                f"4h decision blocked at {datetime.now(timezone.utc).isoformat()}.\n"
                                f"Halt reason: {rail.halt_reason}\n"
                                f"See ~/mark19_data/shadow_reconcile/ for diff details.")
                last_processed_bar_idx = bar_idx_today
                return
            decision = make_decision(artifact, last, bar_dict, log)
            if decision:
                dec_path = DECISIONS_DIR / f"{today}.jsonl"
                with open(dec_path, 'a') as f:
                    f.write(json.dumps(decision, default=str) + "\n")
                log.info(f"  [DECISION] {decision['bar_open_ts']} p_up={decision['p_up']:.4f} action={decision['action']}"
                         + (f" limit=${decision['limit_price']:.2f}" if decision['limit_price'] else ""))
                if decision['action'] != 'SKIP':
                    fill_tracker.add(decision)
                    # Discord entry alert (가상)
                    if td:
                        try:
                            td.on_trade_event("entry", {
                                "side": decision['action'],
                                "size": "0.01",
                                "price": decision['limit_price'],
                                "p_up": round(decision['p_up'], 4),
                                "confidence": round(decision['confidence'], 4),
                            }, mode="shadow")
                        except Exception as e: log.warning(f"  trade alert: {e}")
            # Periodic snapshot every 4h boundary (shadow dashboard)
            if td:
                try: td.send_snapshot("shadow", log)
                except Exception as e: log.warning(f"  dashboard: {e}")
            last_processed_bar_idx = bar_idx_today
        except Exception as e:
            import traceback
            log.error(f"  [CYCLE] {type(e).__name__}: {e}")
            log.error(traceback.format_exc())

    if args.smoke_test:
        log.info("\n=== SMOKE TEST: one cycle ===")
        cycle_once()
        log.info("Done.")
        return

    # Daemon loop
    log.info(f"\n=== DAEMON: polling every {args.poll_sec}s ===")
    last_health_date = None
    def shutdown(s, f): log.info("shutdown signal"); sys.exit(0)
    signal.signal(signal.SIGINT, shutdown); signal.signal(signal.SIGTERM, shutdown)
    while True:
        cycle_once()
        # Daily health report at first cycle after midnight
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if last_health_date != today and datetime.now(timezone.utc).hour >= 0:
            try: daily_health_report(log)
            except Exception as e: log.warning(f"health report failed: {e}")
            last_health_date = today
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
