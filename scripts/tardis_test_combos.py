"""Probe each (symbol, data_type, date) combo to find what fails."""
import os, sys, logging
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / "live_bot" / ".env"
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or not line or "=" not in line: continue
        k, v = line.split("=", 1)
        if k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

from tardis_dev import datasets

API_KEY = os.environ.get("TARDIS_API_KEY")
logging.basicConfig(level=logging.WARNING)

results = {}
DOWNLOAD_DIR = "/tmp/tardis_probe"

for sym in ["ETHUSDT", "BTCUSDT", "SOLUSDT"]:
    for dt in ["trades", "book_snapshot_25", "derivative_ticker", "liquidations"]:
        for dstart, dend in [("2026-04-29", "2026-04-30"),
                              ("2026-05-07", "2026-05-08"),
                              ("2026-05-08", "2026-05-09")]:
            key = f"{sym}/{dt}/{dstart}"
            try:
                datasets.download(
                    exchange="bybit",
                    data_types=[dt],
                    from_date=dstart,
                    to_date=dend,
                    symbols=[sym],
                    api_key=API_KEY,
                    download_dir=DOWNLOAD_DIR,
                )
                results[key] = "OK"
            except Exception as e:
                msg = str(e)[:100]
                if "401" in msg: results[key] = "401"
                elif "404" in msg: results[key] = "404"
                else: results[key] = f"ERR {msg[:60]}"
            print(f"  {key}: {results[key]}")

print("\nSummary:")
for k, v in results.items():
    if v != "OK": print(f"  FAIL {k}: {v}")
ok_count = sum(1 for v in results.values() if v == "OK")
print(f"\nOK: {ok_count}/{len(results)}")
