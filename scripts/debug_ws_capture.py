"""Bybit V5 WS raw orderbook capture for u-sequence diagnostics.

Separate WS connection (does NOT disturb the running bot's WS).
Subscribes only to orderbook.200.ETHUSDT, dumps every snapshot + delta payload
to JSONL for offline u-continuity analysis.

Usage:
  python scripts/debug_ws_capture.py [--duration 120] [--out /tmp/ws_capture.jsonl]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import websocket

WS_URL = "wss://stream.bybit.com/v5/public/linear"
SYMBOL = "ETHUSDT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=120, help="capture seconds")
    ap.add_argument("--out", type=str, default="/tmp/ws_capture.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    fp = open(out_path, "w")
    t0 = time.time()
    n_snap = 0
    n_delta = 0

    def on_open(ws):
        sub = json.dumps({"op": "subscribe", "args": [f"orderbook.200.{SYMBOL}"]})
        ws.send(sub)
        print(f"[capture] subscribed orderbook.200.{SYMBOL}; duration={args.duration}s out={out_path}")

    def on_message(ws, msg):
        nonlocal n_snap, n_delta
        try:
            d = json.loads(msg)
        except Exception:
            return
        if "topic" not in d:
            return
        if not d["topic"].startswith("orderbook"):
            return
        ut = d.get("type")
        data = d.get("data", {})
        u = data.get("u")
        seq = data.get("seq")
        ts = d.get("ts")
        cts = data.get("cts")
        rec = {
            "recv_ts_ms": int(time.time() * 1000),
            "ts": ts,
            "cts": cts,
            "type": ut,
            "u": u,
            "seq": seq,
            "n_bid_chg": len(data.get("b", [])),
            "n_ask_chg": len(data.get("a", [])),
        }
        fp.write(json.dumps(rec) + "\n")
        if ut == "snapshot":
            n_snap += 1
            # First few bid/ask for snapshot sanity
            bids = data.get("b", [])[:3]
            asks = data.get("a", [])[:3]
            rec_snap = {"snapshot_u": u, "bids_top3": bids, "asks_top3": asks}
            fp.write(json.dumps(rec_snap) + "\n")
            print(f"[capture] SNAPSHOT u={u} seq={seq} ts={ts}")
        else:
            n_delta += 1
        if (time.time() - t0) >= args.duration:
            ws.close()

    def on_error(ws, err):
        print(f"[capture] error: {err}", file=sys.stderr)

    def on_close(ws, code, msg):
        fp.flush(); fp.close()
        print(f"[capture] closed code={code}; total: snap={n_snap} delta={n_delta} elapsed={time.time()-t0:.1f}s")

    ws = websocket.WebSocketApp(WS_URL,
        on_open=on_open, on_message=on_message,
        on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=20, ping_timeout=10)


if __name__ == "__main__":
    main()
