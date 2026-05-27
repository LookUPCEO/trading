"""Download Tardis trial data: bybit ETHUSDT/BTCUSDT/SOLUSDT 4/29-5/8."""
import os, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env
env_path = Path(__file__).resolve().parent.parent / "live_bot" / ".env"
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or not line or "=" not in line: continue
        k, v = line.split("=", 1)
        if k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

API_KEY = os.environ.get("TARDIS_API_KEY")
assert API_KEY, "TARDIS_API_KEY missing from .env"

from tardis_dev import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DOWNLOAD_DIR = "/Users/dohun/Desktop/Mark/mark19/data/tardis_trial_raw"

def main():
    log.info("=" * 70)
    log.info("Tardis Trial Download — Bybit Perpetuals 2026-04-29 ~ 2026-05-09")
    log.info("=" * 70)
    log.info(f"Download dir: {DOWNLOAD_DIR}")
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Tardis "bybit" exchange = Bybit USDT perpetuals (linear)
    # Trial scope verified by probe: 4/29 ~ 5/7 (5/8 returns 401)
    # Phase A: light data first (trades + derivative_ticker + liquidations) — ~3-5 GB
    log.info("Phase A: trades + derivative_ticker + liquidations (light)")
    datasets.download(
        exchange="bybit",
        data_types=["trades", "derivative_ticker", "liquidations"],
        from_date="2026-04-29",
        to_date="2026-05-08",
        symbols=["ETHUSDT", "BTCUSDT", "SOLUSDT"],
        api_key=API_KEY,
        download_dir=DOWNLOAD_DIR,
    )
    log.info("Phase A complete")

    # Phase B: book_snapshot_25 (heavy, ~10-20 GB)
    log.info("Phase B: book_snapshot_25 (heavy)")
    datasets.download(
        exchange="bybit",
        data_types=["book_snapshot_25"],
        from_date="2026-04-29",
        to_date="2026-05-08",
        symbols=["ETHUSDT", "BTCUSDT", "SOLUSDT"],
        api_key=API_KEY,
        download_dir=DOWNLOAD_DIR,
    )
    log.info("Phase B complete")
    log.info("Tardis trial download complete")


if __name__ == "__main__":
    main()
