"""
Learned regression "digital twin" - an ALTERNATIVE expected-temperature model.

The main pipeline uses the within-sequence detrend baseline (see ``detector`` and
``run_pipeline``) because it is far more robust to the train->test covariate
shift in this dataset.  This module keeps the learned variant for reference /
experimentation: it predicts a motor's normal temperature from fault-independent
signals (voltage, motion, time - all unaffected by the synthetic fault) and
feeds the residual into the same detector.

It scores slightly lower out-of-fold and, more importantly, extrapolates poorly
to unseen test trajectories (its residual scale inflates and buries real faults),
which is exactly why it is not the default.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from detector import detect_from_residual


class RegressionTwin:
    """Per-motor temperature regressor + residual fault detector."""

    def __init__(self, reg_model, k=4.0, min_rise=1.0, min_run=5, gap=3):
        self.reg_model = reg_model
        self.k = k
        self.min_rise = min_rise
        self.min_run = min_run
        self.gap = gap
        self._fitted = None

    def fit(self, X_normal: np.ndarray, temp_normal: np.ndarray) -> "RegressionTwin":
        """Train on NORMAL samples only."""
        self._fitted = copy.deepcopy(self.reg_model)
        self._fitted.fit(X_normal, temp_normal)
        return self

    def residual(self, X: np.ndarray, temp: np.ndarray) -> np.ndarray:
        """Signed residual measured - predicted (positive => hotter than normal)."""
        return np.asarray(temp, dtype=float) - self._fitted.predict(X)

    def predict_sequence(self, X: np.ndarray, temp: np.ndarray) -> np.ndarray:
        # single-threshold detection (k_low == k_high) on the learned residual
        return detect_from_residual(self.residual(X, temp), self.k, self.k,
                                    self.min_rise, self.min_run, self.gap)

    def predict_grouped(self, df: pd.DataFrame, feature_cols, temp_col) -> np.ndarray:
        out = np.zeros(len(df), dtype=int)
        for _, idx in df.groupby("test_condition", sort=False).groups.items():
            pos = df.index.get_indexer(idx)
            out[pos] = self.predict_sequence(df.loc[idx, feature_cols].to_numpy(),
                                             df.loc[idx, temp_col].to_numpy())
        return out
