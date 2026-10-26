"""S7: probability calibration, fitted on validation data only.

Platt scaling: a one-feature logistic regression on the model's log-odds.
It is monotonic, so ranking metrics (ROC-AUC, PR-AUC, KS) are unchanged;
only the probability levels move.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrated:
    """Wrap an already-fitted model and recalibrate its probabilities."""

    def __init__(self, model):
        self.model = model

    def fit(self, X_valid, y_valid):
        raw = self.model.predict_proba(X_valid)[:, 1]
        self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000)  # C=1e6: effectively no penalty
        self.calibrator_.fit(_logit(raw).reshape(-1, 1), y_valid)
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        p = self.calibrator_.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])