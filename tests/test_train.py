"""S6 tests: the capper learns from training data only, and both trainers run end to end.

Uses small made-up loans, so these run in CI without any real data.
"""
import numpy as np
import pandas as pd
import pytest

from risk.features import FEATURES, LABEL
from risk.train import train_lgbm, train_logreg
from risk.transforms import QuantileCapper


def fake_loans(n: int, seed: int) -> pd.DataFrame:
    """Made-up loans with all 19 features, where low credit and high LTV raise default risk."""
    rng = np.random.default_rng(seed)
    credit = rng.integers(600, 820, n).astype(float)
    ltv = rng.integers(40, 120, n).astype(float)
    df = pd.DataFrame({
        "credit_score": credit,
        "orig_dti": rng.integers(10, 50, n).astype(float),
        "orig_ltv": ltv,
        "orig_cltv": ltv,
        "orig_upb": rng.integers(50, 500, n) * 1000.0,
        "orig_interest_rate": rng.uniform(3.0, 5.0, n),
        "mi_pct": rng.choice([0.0, 25.0, 30.0], n),
        "orig_loan_term": rng.choice([180.0, 360.0], n),
        "num_units": np.ones(n),
        "num_borrowers": rng.choice([1.0, 2.0], n),
        "first_time_homebuyer_flag": rng.choice(["Y", "N"], n),
        "occupancy_status": rng.choice(["P", "I", "S"], n),
        "channel": rng.choice(["R", "B", "C"], n),
        "loan_purpose": rng.choice(["P", "C", "N"], n),
        "property_type": rng.choice(["SF", "PU", "CO"], n),
        "property_state": rng.choice(["FL", "CA", "TX"], n),
        "super_conforming_flag": rng.choice(["Y", "N"], n),
        "harp_indicator": rng.choice(["Y", "N"], n),
        "special_program_flag": rng.choice(["Y", "N"], n),
    })[FEATURES]
    df.loc[:19, "orig_dti"] = np.nan  # some missing values, like the HARP loans
    z = -2.0 + (700 - credit) / 30 + (ltv - 80) / 20
    df[LABEL] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


@pytest.fixture(scope="module")
def data():
    return fake_loans(1500, seed=1), fake_loans(800, seed=2)


def test_capper_learns_caps_from_fit_data_only():
    train = pd.DataFrame({"orig_ltv": np.arange(1, 101, dtype=float),
                          "orig_cltv": np.arange(1, 101, dtype=float)})
    capper = QuantileCapper(quantile=0.9).fit(train)
    new = pd.DataFrame({"orig_ltv": [50.0, 684.0, np.nan], "orig_cltv": [50.0, 854.0, np.nan]})
    out = capper.transform(new)
    cap = train["orig_ltv"].quantile(0.9)
    assert out["orig_ltv"].iloc[0] == 50.0          # normal values untouched
    assert out["orig_ltv"].iloc[1] == cap           # extreme value capped at the training cut-off
    assert np.isnan(out["orig_ltv"].iloc[2])        # missing stays missing
    assert new["orig_ltv"].iloc[1] == 684.0         # the input table was not changed


def test_logreg_trains_scores_and_learns(data):
    train, valid = data
    result = train_logreg(train, valid, c_values=[0.1, 1.0])
    p = result["model"].predict_proba(valid[FEATURES])[:, 1]
    assert len(p) == len(valid) and ((p >= 0) & (p <= 1)).all()
    assert set(result["valid_metrics"]) == {"roc_auc", "pr_auc", "brier", "log_loss"}
    assert len(result["trials"]) == 2
    assert result["valid_metrics"]["roc_auc"] > 0.6  # the fake data has a real pattern to find


def test_lgbm_trains_and_scores(data):
    train, valid = data
    result = train_lgbm(train, valid, n_trials=2, n_estimators=50)
    p = result["model"].predict_proba(valid[FEATURES])[:, 1]
    assert len(p) == len(valid) and ((p >= 0) & (p <= 1)).all()
    assert result["params"]["best_iteration"] >= 1
    assert len(result["trials"]) == 2


def test_saved_pipeline_handles_missing_values(data):
    train, valid = data
    result = train_logreg(train, valid, c_values=[1.0])
    one = valid[FEATURES].head(1).copy()
    one["orig_dti"] = np.nan  # like a HARP loan
    p = result["model"].predict_proba(one)[:, 1]
    assert 0 <= p[0] <= 1