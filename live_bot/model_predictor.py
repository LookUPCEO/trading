"""Mark17 Model Predictor for live trading.

Loads joblib model + predicts vol/dir/trade_signal from feature row.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Model prediction result."""
    vol_proba: float
    dir_proba: float
    trade_signal: bool
    direction: int  # +1 (long), -1 (short), 0 (no trade)
    timestamp: pd.Timestamp

    def __repr__(self):
        signal = "TRADE" if self.trade_signal else "no-trade"
        dir_str = {1: "LONG", -1: "SHORT", 0: "-"}[self.direction]
        return (f"Prediction(vol={self.vol_proba:.3f}, dir={self.dir_proba:.3f}, "
                f"signal={signal}, direction={dir_str}, ts={self.timestamp})")


class ModelPredictor:
    """Mark17 model predictor.

    Usage:
        predictor = ModelPredictor("models/mark17_v1.joblib")
        prediction = predictor.predict(feature_row)
        if prediction.trade_signal:
            # Place order
            ...
    """

    def __init__(self, model_path: Union[str, Path]):
        model_path = Path(model_path)
        log.info(f"Loading model from {model_path}")

        m = joblib.load(model_path)

        self.vol_lr = m["vol_lr"]
        self.vol_scaler = m["vol_scaler"]
        self.dir_lr = m["dir_lr"]
        self.dir_scaler = m["dir_scaler"]

        self.feature_cols = m["feature_cols"]
        self.train_medians = pd.Series(m["train_medians"])

        self.vol_threshold = m["vol_threshold"]
        self.dir_threshold = m["dir_threshold"]
        self.triple_barrier_T = m.get("triple_barrier_T", 0.20)

        self.model_version = m["model_version"]
        self.n_features = m["n_features"]

        log.info(f"  Model: {self.model_version}")
        log.info(f"  Features: {self.n_features}")
        log.info(f"  Vol threshold: {self.vol_threshold}")
        log.info(f"  Dir threshold: {self.dir_threshold}")

        self.sample_test_row_0 = m.get("sample_test_row_0", {})
        self.sample_vol_proba_0 = m.get("sample_vol_proba_0", None)
        self.sample_dir_proba_0 = m.get("sample_dir_proba_0", None)

    def predict(self, feature_row: pd.Series, timestamp: Optional[pd.Timestamp] = None) -> Prediction:
        """Predict vol_proba + dir_proba + trade_signal from feature row."""
        if not isinstance(feature_row, pd.Series):
            raise TypeError(f"feature_row must be pd.Series, got {type(feature_row)}")

        feature_row = feature_row.reindex(self.feature_cols)
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan)
        feature_row = feature_row.fillna(self.train_medians).fillna(0)

        # vol_scaler was fit on DataFrame (has feature_names_in_) → pass DataFrame
        # dir_scaler was fit on numpy (no feature_names_in_) → pass numpy
        # This avoids sklearn feature-name mismatch warnings.
        feature_df = pd.DataFrame([feature_row.values], columns=self.feature_cols)
        feature_arr = feature_row.values.reshape(1, -1)

        vol_scaled = self.vol_scaler.transform(feature_df)
        vol_proba = float(self.vol_lr.predict_proba(vol_scaled)[0, 1])

        dir_scaled = self.dir_scaler.transform(feature_arr)
        dir_proba = float(self.dir_lr.predict_proba(dir_scaled)[0, 1])

        trade_signal = False
        direction = 0
        if vol_proba > self.vol_threshold:
            if dir_proba > self.dir_threshold:
                trade_signal = True
                direction = +1
            elif dir_proba < (1 - self.dir_threshold):
                trade_signal = True
                direction = -1

        if timestamp is None:
            timestamp = pd.Timestamp.now(tz="UTC")

        return Prediction(
            vol_proba=vol_proba,
            dir_proba=dir_proba,
            trade_signal=trade_signal,
            direction=direction,
            timestamp=timestamp,
        )

    def verify_reproducibility(self) -> dict:
        """Verify model reproduces backtest sample row."""
        if not self.sample_test_row_0 or self.sample_vol_proba_0 is None:
            return {"status": "no_sample", "vol_diff": None, "dir_diff": None}

        sample_series = pd.Series(self.sample_test_row_0).reindex(self.feature_cols)
        prediction = self.predict(sample_series)

        vol_diff = abs(prediction.vol_proba - self.sample_vol_proba_0)
        dir_diff = abs(prediction.dir_proba - self.sample_dir_proba_0)

        status = "OK" if (vol_diff < 1e-6 and dir_diff < 1e-6) else "DRIFT"

        return {
            "status": status,
            "vol_diff": vol_diff,
            "dir_diff": dir_diff,
            "expected_vol": self.sample_vol_proba_0,
            "got_vol": prediction.vol_proba,
            "expected_dir": self.sample_dir_proba_0,
            "got_dir": prediction.dir_proba,
        }
