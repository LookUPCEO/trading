"""Single-file test: 1 day, 1 symbol, 1 data type."""
import os, sys
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
print(f"Key prefix: {API_KEY[:10]}... len {len(API_KEY)}")

# Test: trades only, 1 symbol, 1 day
import logging
logging.basicConfig(level=logging.INFO)

try:
    datasets.download(
        exchange="bybit",
        data_types=["trades"],
        from_date="2026-04-29",
        to_date="2026-04-30",  # exclusive → 1 day only
        symbols=["ETHUSDT"],
        api_key=API_KEY,
        download_dir="/tmp/tardis_test",
    )
    print("OK: single file downloaded")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")
