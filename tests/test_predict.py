import math
from pathlib import Path

import duckdb
import pytest

from risk.features import FEATURES
from risk.predict import (
    CALIBRATED_PATH, RAW_PATH, REQUIRED_FIELDS, LoanInputError, explain, score, validate_loan,
)

AGENT_DB = Path(__file__).resolve().parents[1] / "data" / "agent.duckdb"

needs_models = pytest.mark.skipif(
    not (CALIBRATED_PATH.exists() and RAW_PATH.exists()),
    reason="trained model files not present",
)
needs_agent_db = pytest.mark.skipif(not AGENT_DB.exists(), reason="agent.duckdb not built")

BASE = {
    "credit_score": 700, "orig_ltv": 90, "orig_dti": 40, "orig_upb": 250000,
    "orig_interest_rate": 4.5, "loan_purpose": "P", "occupancy_status": "P",
    "property_state": "FL",
}


# ---------- validation (no model files needed; runs in CI) ----------

def test_required_only_fills_the_rest():
    X, filled = validate_loan(BASE)
    assert list(X.columns) == FEATURES
    assert set(filled) == set(FEATURES) - set(REQUIRED_FIELDS)
    assert not X.isna().any().any()


def test_missing_cltv_defaults_to_ltv():
    X, filled = validate_loan(BASE)
    assert X.loc[0, "orig_cltv"] == BASE["orig_ltv"]
    assert filled["orig_cltv"] == BASE["orig_ltv"]


def test_given_values_are_kept():
    X, filled = validate_loan({**BASE, "orig_cltv": 95})
    assert X.loc[0, "credit_score"] == BASE["credit_score"]
    assert X.loc[0, "orig_cltv"] == 95
    assert "credit_score" not in filled and "orig_cltv" not in filled


def test_missing_mi_high_ltv_non_harp_gets_group_median():
    # BASE has LTV 90 and no harp_indicator (filled with the mode, "N").
    X, filled = validate_loan(BASE)
    assert X.loc[0, "mi_pct"] == 30.0
    assert filled["mi_pct"] == 30.0


@pytest.mark.parametrize("extra", [{"harp_indicator": "Y"}, {"orig_ltv": 80}])
def test_missing_mi_otherwise_gets_overall_median(extra):
    X, filled = validate_loan({**BASE, **extra})
    assert X.loc[0, "mi_pct"] == 0.0


def test_given_mi_is_kept_on_high_ltv():
    X, filled = validate_loan({**BASE, "mi_pct": 12})
    assert X.loc[0, "mi_pct"] == 12
    assert "mi_pct" not in filled


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_rejected(field):
    loan = {k: v for k, v in BASE.items() if k != field}
    with pytest.raises(LoanInputError, match=field):
        validate_loan(loan)


def test_blank_dti_counts_as_missing():
    with pytest.raises(LoanInputError, match="orig_dti"):
        validate_loan({**BASE, "orig_dti": ""})


def test_unknown_field_rejected():
    with pytest.raises(LoanInputError, match="postal_code"):
        validate_loan({**BASE, "postal_code": "02115"})


def test_non_numeric_rejected():
    with pytest.raises(LoanInputError, match="must be a number"):
        validate_loan({**BASE, "credit_score": "excellent"})


def test_out_of_training_range_rejected():
    with pytest.raises(LoanInputError, match="training range"):
        validate_loan({**BASE, "orig_interest_rate": 7.0})


def test_invalid_code_rejected():
    with pytest.raises(LoanInputError, match="not a valid code"):
        validate_loan({**BASE, "loan_purpose": "X"})


def test_codes_are_normalized():
    X, _ = validate_loan({**BASE, "property_state": " fl "})
    assert X.loc[0, "property_state"] == "FL"


# ---------- scoring (needs local model files; skipped in CI) ----------

@needs_models
def test_score_returns_probability_and_decision():
    result = score(BASE)
    assert 0.0 < result["probability"] < 1.0
    assert result["decision"] in {"approve", "decline"}
    assert result["note"] == "Demo only, not a credit decision."


@needs_models
def test_explain_returns_top_five_known_features():
    result = explain(BASE)
    assert len(result["drivers"]) == 5
    assert all(d["feature"] in FEATURES for d in result["drivers"])


@needs_models
@needs_agent_db
def test_serving_matches_stored_test_scores():
    """Re-scoring real test loans one at a time must reproduce test_scores exactly."""
    cols = ", ".join(f"l.{f}" for f in FEATURES)
    complete = " AND ".join(f"l.{f} IS NOT NULL" for f in FEATURES)
    con = duckdb.connect(str(AGENT_DB), read_only=True)
    rows = con.sql(
        f"SELECT {cols}, t.probability FROM loans l "
        f"JOIN test_scores t ON l.loan_seq_num = t.loan_seq_num "
        f"WHERE {complete} ORDER BY l.loan_seq_num LIMIT 20"
    ).fetchall()
    con.close()

    checked = 0
    for r in rows:
        try:
            result = score(dict(zip(FEATURES, r[:-1])))
        except LoanInputError:
            continue  # outside the training range; validation rejects it on purpose
        assert math.isclose(result["probability"], r[-1], rel_tol=1e-9, abs_tol=1e-12)
        checked += 1
    assert checked > 0