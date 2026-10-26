"""S8 explanation tests: column mapping, SHAP adds up exactly, and per-loan output is well formed."""
import numpy as np
import pytest

from risk.explain import Explainer, original_feature
from risk.features import FEATURES
from risk.train import train_lgbm
from tests.test_train import fake_loans


def test_original_feature_mapping():
    assert original_feature("num__credit_score") == "credit_score"
    assert original_feature("cat__property_state_FL") == "property_state"
    assert original_feature("cat__harp_indicator_Y") == "harp_indicator"
    assert original_feature("cat__property_type_infrequent_sklearn") == "property_type"
    assert original_feature("cat__first_time_homebuyer_flag_missing") == "first_time_homebuyer_flag"


@pytest.fixture(scope="module")
def fitted():
    train, valid = fake_loans(1500, seed=1), fake_loans(800, seed=2)
    pipeline = train_lgbm(train, valid, n_trials=1, n_estimators=50)["model"]
    return Explainer(pipeline), valid


def test_every_column_maps_to_a_known_feature(fitted):
    explainer, _ = fitted
    assert set(explainer.groups) <= set(FEATURES)


def test_shap_values_add_up_to_the_model_score(fitted):
    explainer, valid = fitted
    X = valid[FEATURES].head(50)
    per_feature, base = explainer.contributions(X)
    raw_score = explainer.model.predict(explainer.prep.transform(X), raw_score=True)
    assert np.allclose(per_feature.sum(axis=1).to_numpy() + base, raw_score, atol=1e-6)


def test_explain_one_returns_top_drivers_in_order(fitted):
    explainer, valid = fitted
    drivers = explainer.explain_one(valid[FEATURES].head(1), top_n=5)
    assert len(drivers) == 5
    sizes = [abs(d["shap_log_odds"]) for d in drivers]
    assert sizes == sorted(sizes, reverse=True)
    for d in drivers:
        assert d["feature"] in FEATURES
        assert d["direction"] == ("raises risk" if d["shap_log_odds"] > 0 else "lowers risk")