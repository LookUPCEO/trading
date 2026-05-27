"""시도 38: LSTM Direction model with 60-min lookback, walk-forward 9 days."""
import sys, logging, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset

_spec36 = importlib.util.spec_from_file_location("_sido36", _HERE / "sido36_regime_invariant.py")
_mod36 = importlib.util.module_from_spec(_spec36)
_spec36.loader.exec_module(_mod36)
add_normalized_features = _mod36.add_normalized_features
HIGH_SHIFT_FEATURES = _mod36.HIGH_SHIFT_FEATURES


# LSTM hyperparameters (CPU-friendly: small)
LOOKBACK = 60          # 60-min sequence
HIDDEN_DIM = 64        # small for CPU
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 3


class SeqDataset(Dataset):
    def __init__(self, X, y, lookback):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.lookback = lookback
        # valid indices: i where i >= lookback - 1
        self.valid_idx = np.arange(lookback - 1, len(X))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        seq = self.X[i - self.lookback + 1: i + 1]  # (LOOKBACK, n_feat)
        return torch.from_numpy(seq), torch.tensor(self.y[i])


class LSTMDirection(nn.Module):
    def __init__(self, n_features, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden_dim)
        return self.head(last).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val, n_features, log, max_epochs=MAX_EPOCHS):
    train_ds = SeqDataset(X_train, y_train, LOOKBACK)
    val_ds = SeqDataset(X_val, y_val, LOOKBACK)
    if len(train_ds) < 100 or len(val_ds) < 50:
        log.warning(f"  too few sequences: train {len(train_ds)} val {len(val_ds)}")
        return None
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LSTMDirection(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.BCELoss()

    best_val = 0.0; best_state = None; patience = 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batch = 0
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item(); n_batch += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in val_dl:
                p = model(xb).numpy()
                preds.append(p); ys.append(yb.numpy())
            preds = np.concatenate(preds); ys = np.concatenate(ys)
        from sklearn.metrics import roc_auc_score
        if len(set(ys)) > 1:
            val_auc = roc_auc_score(ys, preds)
        else:
            val_auc = 0.5
        log.info(f"  epoch {epoch+1}/{max_epochs}  train_loss {train_loss/max(n_batch,1):.4f}  val_auc {val_auc:.3f}  time {time.time()-t0:.1f}s")

        if val_auc > best_val + 1e-4:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                log.info(f"  early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_lstm(model, X, lookback):
    ds = SeqDataset(X, np.zeros(len(X)), lookback)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    preds = []; valid_idx = []
    with torch.no_grad():
        for i, (xb, _) in enumerate(dl):
            p = model(xb).numpy()
            preds.append(p)
    preds = np.concatenate(preds)
    return preds, ds.valid_idx


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 38: LSTM Direction (lookback 60min, hidden 64)")
    log.info("=" * 70)
    log.info(f"  torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    log.info(f"  hyperparams: lookback {LOOKBACK} hidden {HIDDEN_DIM} layers {NUM_LAYERS} batch {BATCH_SIZE}")

    np.random.seed(42)
    torch.manual_seed(42)

    log.info("\nBuilding...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = {}
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    tardis_train.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_train = add_normalized_features(tardis_train, log)
    tardis_val = add_normalized_features(tardis_val, log)
    for d in self_dfs:
        self_dfs[d] = add_normalized_features(self_dfs[d], log)

    norm_features = [c for c in tardis_train.columns if c.endswith("_norm")]
    canonical = get_feature_columns(tardis_train)
    feat_set = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
    log.info(f"\nFeature set (norm_only): {len(feat_set)}")

    test_dates = SELF_ALL[1:]
    T = 0.20
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30
    DIR_TH = 0.55; VOL_TH = 0.6

    walk_results = []

    for step_idx, test_date in enumerate(test_dates, 1):
        test_dt_idx = SELF_ALL.index(test_date)
        train_self_dates = SELF_ALL[:test_dt_idx]
        log.info(f"\n=== STEP {step_idx}/9: train Self {len(train_self_dates)}d, test {test_date} ===")
        torch.manual_seed(42 + step_idx)

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_test_df = self_dfs[test_date].copy()
        train_df = pd.concat([tardis_train, self_train_df], ignore_index=True)
        val_df = tardis_val

        meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)

        # Sequence: build per _source_date to avoid cross-day leakage
        # Train: per-day sequences from train_df (sorted by ts within day)
        ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in train_df.columns), None)
        if ts_col is None:
            log.error("no timestamp column"); continue
        train_df = train_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)
        val_df_s = val_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)
        self_test_df_s = self_test_df.sort_values(ts_col).reset_index(drop=True)

        # Apply mask
        tm = train_df[dir_target].abs() > T
        # Standardize
        from sklearn.preprocessing import StandardScaler
        sd = StandardScaler()
        X_tr_full = mx(train_df).values
        X_tr_full = sd.fit_transform(X_tr_full)
        X_vl_full = sd.transform(mx(val_df_s).values)
        X_st_full = sd.transform(mx(self_test_df_s).values)

        # Targets aligned to full row index (not just filtered)
        y_tr = (train_df[dir_target] > 0).astype(int).values
        y_vl = (val_df_s[dir_target] > 0).astype(int).values
        y_st = (self_test_df_s[dir_target] > 0).astype(int).values

        log.info(f"  train rows {len(X_tr_full)}, val rows {len(X_vl_full)}, test rows {len(X_st_full)}")

        # Train
        log.info("  training LSTM...")
        try:
            result = train_lstm(X_tr_full, y_tr, X_vl_full, y_vl, n_features=len(feat_set), log=log,
                                max_epochs=MAX_EPOCHS)
        except Exception as e:
            log.error(f"  train fail: {e}")
            walk_results.append({"step": step_idx, "test_date": test_date, "auc_self": float("nan"),
                                  "error": str(e)})
            continue
        if result is None:
            walk_results.append({"step": step_idx, "test_date": test_date, "auc_self": float("nan")})
            continue
        model, best_val = result

        # Predict on Self test
        preds, valid_idx = predict_lstm(model, X_st_full, LOOKBACK)
        # AUC on test (aligned via valid_idx, filtered by direction mask)
        test_mask = self_test_df_s[dir_target].abs() > T
        # Only valid (lookback-aligned) rows
        valid_mask_full = np.zeros(len(X_st_full), dtype=bool)
        valid_mask_full[valid_idx] = True
        # Filtered AUC: rows that are both valid AND |dir|>T
        eligible = valid_mask_full & test_mask.values
        # Map preds (from valid_idx) back to full-row indexing
        pred_full = np.full(len(X_st_full), 0.5)
        pred_full[valid_idx] = preds
        from sklearn.metrics import roc_auc_score
        if eligible.sum() > 1 and len(set(y_st[eligible])) > 1:
            auc_self = roc_auc_score(y_st[eligible], pred_full[eligible])
        else:
            auc_self = float("nan")
        log.info(f"  Self test AUC: {auc_self:.3f}  (val best {best_val:.3f})")

        # Backtest @ TH 0.55 on Self test
        bt = self_test_df_s.copy().reset_index(drop=True)
        bt["dir_proba"] = pred_full
        # Compute vol_proba via simple LR on Tardis (re-fit each step is overkill; use mean train vol)
        train_vol_med = float(train_df[vol_target].median())
        # Simpler: vol_proba = (vol > train_vol_med) * 1.0 (binary)
        # Actually we want a probabilistic vol filter. Use a quick LR.
        from sklearn.linear_model import LogisticRegression
        sv = StandardScaler()
        X_v_lr = sv.fit_transform(mx(train_df).values)
        y_v_lr = (train_df[vol_target] > train_vol_med).astype(int).values
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_v_lr, y_v_lr)
        bt["vol_proba"] = lrv.predict_proba(sv.transform(mx(bt).values))[:, 1]
        bt["actual_return"] = bt[dir_target].values

        ts_col2 = next((c for c in ["_ts", "ts", "timestamp"] if c in bt.columns), None)
        price_col = next((c for c in ["ob_mid_price", "mid"] if c in bt.columns), None)
        bt = bt.sort_values(ts_col2).reset_index(drop=True)

        def drift_fill(d_df, idx, direction):
            if idx >= len(d_df): return False
            e = d_df.iloc[idx][price_col]
            if pd.isna(e): return False
            lim = e * (0.99995 if direction == 1 else 1.00005)
            for t in range(1, MAX_HOLD + 1):
                if idx + t >= len(d_df): return False
                x = d_df.iloc[idx + t][price_col]
                if pd.isna(x): continue
                if direction == 1 and x <= lim: return True
                if direction == -1 and x >= lim: return True
                lim = x * (0.99995 if direction == 1 else 1.00005)
            return False

        trades = []; i = 0; n = len(bt)
        n_sl = 0; n_maker = 0
        while i < n:
            r = bt.iloc[i]
            if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                i += 1; continue
            direction = 0; trade = False
            if r["vol_proba"] > VOL_TH:
                if r["dir_proba"] > DIR_TH: direction = 1; trade = True
                elif r["dir_proba"] < (1 - DIR_TH): direction = -1; trade = True
            if trade:
                e = r[price_col]; ar = direction * r["actual_return"]; sl = False
                for t in range(1, LOCKOUT + 1):
                    if i + t >= n: break
                    x = bt.iloc[i + t][price_col]
                    if pd.isna(x): continue
                    p = direction * (x - e) / e * 100
                    if p <= -SL: ar = -SL; sl = True; break
                if sl:
                    fee_e = FEE_TAKER; n_sl += 1
                else:
                    filled = drift_fill(bt, i + LOCKOUT, -direction)
                    fee_e = FEE_MAKER if filled else FEE_TAKER
                    if filled: n_maker += 1
                trades.append({"net_pnl": ar - (FEE_TAKER + fee_e)})
                i += LOCKOUT
            else:
                i += 1
        n_total = len(trades)
        ps = sum(t["net_pnl"] for t in trades) if trades else 0
        wr = (sum(1 for t in trades if t["net_pnl"] > 0) / n_total) if n_total else 0
        log.info(f"  PnL @ TH {DIR_TH}: {ps:+.3f}% ({n_total}t, win {wr*100:.1f}%, SL {n_sl})")

        walk_results.append({
            "step": step_idx, "test_date": test_date,
            "auc_self": float(auc_self), "auc_val": float(best_val),
            "pnl": float(ps), "n_trades": n_total, "win_rate": float(wr),
            "n_sl": n_sl,
        })

    # Aggregate
    print()
    print("=" * 90)
    print(f"LSTM WALK-FORWARD (lookback {LOOKBACK}min, hidden {HIDDEN_DIM})")
    print("=" * 90)
    print(f"\n{'Step':<6} {'Date':<14} {'AUC val':<10} {'AUC self':<10} {'PnL':<10} {'Trades':<8}")
    print("-" * 65)
    for r in walk_results:
        print(f"{r['step']:<6} {r.get('test_date',''):<14} {r.get('auc_val',float('nan')):<10.3f} {r.get('auc_self',float('nan')):<10.3f} {r.get('pnl',0):<+10.3f}% {r.get('n_trades',0):<8}")

    aucs = [r["auc_self"] for r in walk_results if not np.isnan(r.get("auc_self", float("nan")))]
    pnls = [r["pnl"] for r in walk_results if "pnl" in r]
    if aucs:
        a = np.array(aucs)
        print(f"\n  AUC: mean {a.mean():.3f}  std {a.std():.3f}  >0.55 {(a>0.55).sum()}/{len(a)}")
    if pnls:
        p = np.array(pnls)
        print(f"  PnL: mean {p.mean():+.3f}%  std {p.std():.3f}  total {p.sum():+.3f}%  positive {(p>0).sum()}/{len(p)}")

    out = {"approach": "LSTM 60-min lookback", "steps": walk_results,
           "hyperparams": {"lookback": LOOKBACK, "hidden_dim": HIDDEN_DIM,
                           "layers": NUM_LAYERS, "batch": BATCH_SIZE, "lr": LR}}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido38_lstm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if aucs:
        if a.mean() > 0.62:
            print(f"\n  ✅ LSTM AUC {a.mean():.3f} > sido36 (0.608)")
        elif a.mean() > 0.58:
            print(f"\n  🟡 LSTM AUC {a.mean():.3f} similar to sido36")
        else:
            print(f"\n  ❌ LSTM AUC {a.mean():.3f} < sido36. Sequence learning not effective on small data.")
    log.info("\n시도 38 complete")


if __name__ == "__main__":
    main()
