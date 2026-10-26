"""S7 tests: KS, decile table, Platt calibration, and the bootstrap, on small made-up data."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from risk.calibration import PlattCalibrated
from risk.evaluate import bootstrap_auc_diff, decile_table, ks_statistic


def test_ks_perfect_separation_is_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, p) == pytest.approx(1.0)


def test_ks_known_small_example():
    # Worked by hand: at cut-off 0.4, half the defaulters are caught and no good loan is flagged.
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert ks_statistic(y, p) == pytest.approx(0.5)


def test_decile_table_adds_up():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = decile_table(y, p)
    assert t["loans"].sum() == 1000
    assert t["defaults"].sum() == y.sum()
    assert list(t.index) == list(range(10, 0, -1))
    assert t["cum_share_of_defaults"].iloc[-1] == pytest.approx(1.0)
    assert t.loc[10, "default_rate"] > t.loc[1, "default_rate"]


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=3000)})
    y = (rng.random(3000) < 1 / (1 + np.exp(-(2 * X["x"] - 3)))).astype(int).to_numpy()
    return X, y, LogisticRegression().fit(X, y)


def test_platt_keeps_ranking_and_gives_valid_probabilities(toy):
    X, y, base = toy
    raw = base.predict_proba(X)[:, 1]
    p = PlattCalibrated(base).fit(X, y).predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, raw))  # order unchanged


class ShrunkModel:
    """A deliberately mis-calibrated model: every probability is 3x too small."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1] / 3
        return np.column_stack([1 - p, p])


def test_platt_fixes_a_miscalibrated_model(toy):
    X, y, base = toy
    bad = ShrunkModel(base)
    assert bad.predict_proba(X)[:, 1].mean() < 0.6 * y.mean()  # too low before
    fixed = PlattCalibrated(bad).fit(X, y).predict_proba(X)[:, 1]
    assert fixed.mean() == pytest.approx(y.mean(), abs=0.01)   # right level after


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.2).astype(int)
    noise = rng.random(2000)
    informative = 0.5 * y + rng.random(2000)
    r = bootstrap_auc_diff(y, noise, informative, n_boot=200)
    assert r["ci_low"] <= r["diff"] <= r["ci_high"]
    assert r["diff"] > 0 and r["share_b_better"] > 0.95