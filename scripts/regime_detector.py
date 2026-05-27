"""
Fit regime classifiers (HMM + GMM + K-means) on daily microstructure features.

Workflow:
  1. Load regime_features.parquet (output of scripts/regime_features.py)
  2. Z-score normalize features (train window only)
  3. Fit HMM (range of n_states), GMM (BIC scan), K-means (baseline)
  4. Emit regime labels + posteriors per day
  5. Print regime distribution + transition matrix + per-regime feature centroids

Walk-forward defaults (ETH):
  Train  : start .. 2024-12-31
  Val    : 2025-01-01 .. 2025-09-30
  Test   : 2025-10-01 ..

Outputs:
  - regime_labels.parquet: date, symbol, hmm_label, hmm_posterior_*, gmm_label, gmm_posterior_*, kmeans_label
  - regime_summary.md: per-regime centroids + transition matrix
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn import hmm  # type: ignore


FEATURE_COLS = [
    "mean_spread_bp", "mean_depth_top5", "mid_realized_vol_1m_pct",
    "update_rate_per_sec", "depth_slope", "bid_ask_imbalance",
    "mid_range_bp", "depth_top5_cv", "log_mid_drift_bp",
]


def pick_n_states_bic(X: np.ndarray, n_range=range(2, 9)) -> tuple[int, dict]:
    """Use GMM BIC to suggest n_states (proxy, since hmmlearn doesn't have BIC)."""
    bics = {}
    for n in n_range:
        try:
            g = GaussianMixture(n_components=n, covariance_type="full",
                                random_state=0, max_iter=200).fit(X)
            bics[n] = g.bic(X)
        except Exception as e:
            bics[n] = float("nan")
    best = min(bics, key=lambda k: bics[k] if not np.isnan(bics[k]) else float("inf"))
    return best, bics


def fit_hmm(X_train: np.ndarray, n_states: int) -> hmm.GaussianHMM:
    m = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, random_state=0)
    m.fit(X_train)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="/Volumes/PortableSSD/bybit_data/regime_features.parquet")
    p.add_argument("--symbol", default="ETHUSDT", help="Primary symbol for fitting")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-end", default="2025-09-30")
    p.add_argument("--out-labels", default="/Volumes/PortableSSD/bybit_data/regime_labels.parquet")
    p.add_argument("--out-summary", default="/Volumes/PortableSSD/bybit_data/regime_summary.md")
    p.add_argument("--n-states", type=int, default=0, help="0 = auto-pick via BIC")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    feats = pd.read_parquet(args.features)
    feats["date"] = pd.to_datetime(feats["date"])
    feats = feats.sort_values(["symbol", "date"]).reset_index(drop=True)
    log.info(f"Loaded {len(feats)} day-rows ({feats.symbol.value_counts().to_dict()})")

    primary = feats[feats.symbol == args.symbol].copy().reset_index(drop=True)
    if len(primary) < 30:
        log.error(f"Insufficient {args.symbol} days ({len(primary)}). Need ≥30 for fit.")
        sys.exit(1)

    train_mask = primary.date <= pd.Timestamp(args.train_end)
    val_mask = (primary.date > pd.Timestamp(args.train_end)) & (primary.date <= pd.Timestamp(args.val_end))
    test_mask = primary.date > pd.Timestamp(args.val_end)
    log.info(f"  splits — train: {train_mask.sum()}, val: {val_mask.sum()}, test: {test_mask.sum()}")

    if train_mask.sum() < 30:
        log.warning(f"Train split has <30 days; using ALL primary days as train (prototype mode).")
        train_mask = pd.Series([True] * len(primary))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(primary.loc[train_mask, FEATURE_COLS].values)
    X_all = scaler.transform(primary[FEATURE_COLS].values)

    # n_states selection
    if args.n_states <= 0:
        n_states, bics = pick_n_states_bic(X_train)
        log.info(f"  BIC scan: {bics}, picked n_states={n_states}")
    else:
        n_states = args.n_states
        log.info(f"  using n_states={n_states} (user-provided)")

    # 1. HMM
    log.info(f"Fitting HMM (n={n_states})...")
    hmm_model = fit_hmm(X_train, n_states)
    hmm_labels = hmm_model.predict(X_all)
    hmm_post = hmm_model.predict_proba(X_all)

    # 2. GMM (independent of HMM)
    log.info(f"Fitting GMM (n={n_states})...")
    gmm = GaussianMixture(n_components=n_states, covariance_type="full",
                          random_state=0, max_iter=200).fit(X_train)
    gmm_labels = gmm.predict(X_all)
    gmm_post = gmm.predict_proba(X_all)

    # 3. K-means baseline
    log.info(f"Fitting KMeans (k={n_states})...")
    km = KMeans(n_clusters=n_states, random_state=0, n_init=10).fit(X_train)
    km_labels = km.predict(X_all)

    # ---------------- Output labels ----------------
    out = primary[["date", "symbol"]].copy()
    out["hmm_label"] = hmm_labels
    out["gmm_label"] = gmm_labels
    out["kmeans_label"] = km_labels
    for k in range(n_states):
        out[f"hmm_post_{k}"] = hmm_post[:, k]
        out[f"gmm_post_{k}"] = gmm_post[:, k]
    out_path = Path(args.out_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info(f"Labels written → {out_path}")

    # ---------------- Summary ----------------
    lines = []
    lines.append(f"# Regime Summary — {args.symbol}\n")
    lines.append(f"- n_states: {n_states}, samples: {len(primary)} days")
    lines.append(f"- train: {train_mask.sum()}, val: {val_mask.sum()}, test: {test_mask.sum()}\n")

    # Per-regime centroid (HMM)
    lines.append("## HMM regime centroids (original feature scale)\n")
    centroids = pd.DataFrame(index=range(n_states), columns=FEATURE_COLS, dtype=float)
    for k in range(n_states):
        mask = hmm_labels == k
        if mask.sum() == 0:
            continue
        for col in FEATURE_COLS:
            centroids.loc[k, col] = float(primary.loc[mask, col].mean())
    centroids["count_days"] = pd.Series({k: int((hmm_labels == k).sum()) for k in range(n_states)})
    lines.append(centroids.to_string())
    lines.append("\n")

    # Transition matrix (HMM)
    lines.append("## HMM transition counts (from learned model)\n")
    tm = hmm_model.transmat_
    tm_df = pd.DataFrame(tm, index=[f"from_{k}" for k in range(n_states)],
                         columns=[f"to_{k}" for k in range(n_states)])
    lines.append(tm_df.to_string(float_format="%.3f"))
    lines.append("\n")

    # Empirical transitions from labels
    lines.append("## Empirical transitions (from labeled sequence)\n")
    emp = np.zeros((n_states, n_states), dtype=int)
    for i in range(1, len(hmm_labels)):
        emp[hmm_labels[i-1], hmm_labels[i]] += 1
    emp_df = pd.DataFrame(emp, index=[f"from_{k}" for k in range(n_states)],
                          columns=[f"to_{k}" for k in range(n_states)])
    lines.append(emp_df.to_string())
    lines.append("\n")

    # Regime persistence: average run length per state
    lines.append("## Regime persistence (mean run length, days)\n")
    runs: dict = {k: [] for k in range(n_states)}
    cur_state = hmm_labels[0]
    cur_len = 1
    for s in hmm_labels[1:]:
        if s == cur_state:
            cur_len += 1
        else:
            runs[cur_state].append(cur_len)
            cur_state = s; cur_len = 1
    runs[cur_state].append(cur_len)
    for k in range(n_states):
        if runs[k]:
            lines.append(f"  state {k}: mean={np.mean(runs[k]):.1f}d  n_runs={len(runs[k])}  max={max(runs[k])}d")
    lines.append("\n")

    summary = "\n".join(lines)
    Path(args.out_summary).write_text(summary)
    log.info(f"Summary written → {args.out_summary}")
    print(summary)


if __name__ == "__main__":
    main()
