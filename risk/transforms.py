"""Small reusable scikit-learn transformers.

They live in their own module (not in a script) so saved models can always be loaded again.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Cap chosen columns at a high quantile learned in fit() (training data only).

    Used for the logistic baseline, where a few extreme LTVs (up to 684%) would pull the weights.
    Missing values stay missing.
    """

    def __init__(self, columns=("orig_ltv", "orig_cltv"), quantile=0.999):
        self.columns = columns
        self.quantile = quantile

    def fit(self, X, y=None):
        self.caps_ = {col: float(X[col].quantile(self.quantile)) for col in self.columns}
        return self

    def transform(self, X):
        X = X.copy()  # never change the caller's data
        for col, cap in self.caps_.items():
            X[col] = X[col].clip(upper=cap)
        return X