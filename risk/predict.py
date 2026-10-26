"""A1c: validate one loan, score it, and explain it.

Used by the agent's score_loan / explain_score tools and by the Streamlit form.
Scoring reuses the exact training pipeline and calibration object (no re-implemented
preprocessing), so serving can't drift from training.

Required fields: REQUIRED_FIELDS below. DTI is required because a blank DTI looks
like a HARP refinance to the model. Other fields are filled with typical training
values and reported back. Unknown fields (e.g. postal_code) are rejected, and values
outside the training range are rejected instead of extrapolated.

Run from the repo root:  python -m risk.predict
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from risk.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CALIBRATED_PATH = MODELS_DIR / "lgbm_calibrated.joblib"
RAW_PATH = MODELS_DIR / "lgbm.joblib"
DEFAULTS_PATH = REPORTS_DIR / "feature_defaults.json"
THRESHOLD_PATH = REPORTS_DIR / "s8_threshold.json"

REQUIRED_FIELDS = [
    "credit_score", "orig_ltv", "orig_dti", "orig_upb", "orig_interest_rate",
    "loan_purpose", "occupancy_status", "property_state",
]
DISCLAIMER = "Demo only, not a credit decision."


class LoanInputError(ValueError):
    """The loan fields are invalid. The message says how to fix them."""


@lru_cache(maxsize=1)
def load_defaults() -> dict:
    return json.loads(DEFAULTS_PATH.read_text())


@lru_cache(maxsize=1)
def _load_models():
    # Imported here so validation works without model files (e.g. in CI).
    from risk.explain import Explainer

    calibrated = joblib.load(CALIBRATED_PATH)
    explainer = Explainer(joblib.load(RAW_PATH))
    threshold_info = json.loads(THRESHOLD_PATH.read_text())
    return calibrated, explainer, threshold_info


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return isinstance(value, float) and pd.isna(value)


def _to_python(value):
    """numpy scalars -> plain Python, so results can be turned into JSON."""
    return value.item() if hasattr(value, "item") else value


def validate_loan(fields: dict) -> tuple[pd.DataFrame, dict]:
    """Return a one-row DataFrame with all 19 features, plus the fields that were filled."""
    defaults = load_defaults()

    unknown = sorted(set(fields) - set(FEATURES))
    if unknown:
        raise LoanInputError(f"Unknown fields: {unknown}. Allowed fields: {FEATURES}.")

    missing = [f for f in REQUIRED_FIELDS if _is_blank(fields.get(f))]
    if missing:
        raise LoanInputError(f"Missing required fields: {missing}.")

    row, filled = {}, {}

    for f in NUMERIC_FEATURES:
        spec = defaults["numeric"][f]
        value = fields.get(f)
        if _is_blank(value):
            row[f] = filled[f] = spec["median"]
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise LoanInputError(f"{f} must be a number, got {value!r}.") from None
        if not spec["min"] <= value <= spec["max"]:
            raise LoanInputError(
                f"{f}={value} is outside the training range [{spec['min']}, {spec['max']}]."
            )
        row[f] = value

    for f in CATEGORICAL_FEATURES:
        spec = defaults["categorical"][f]
        value = fields.get(f)
        if _is_blank(value):
            mode = spec["mode"]
            row[f] = float("nan") if mode is None else mode  # NaN -> imputer's "missing"
            filled[f] = mode
            continue
        value = str(value).strip().upper()
        if value not in spec["allowed"]:
            raise LoanInputError(f"{f}={value!r} is not a valid code. Allowed: {spec['allowed']}.")
        row[f] = value

    return pd.DataFrame([row], columns=FEATURES), filled


def score(fields: dict) -> dict:
    """Calibrated default probability and the decision at the chosen threshold."""
    X, filled = validate_loan(fields)
    calibrated, _, info = _load_models()
    p = float(calibrated.predict_proba(X)[:, 1][0])
    threshold = float(info["threshold"])
    return {
        "probability": p,
        "decision": "decline" if p >= threshold else "approve",
        "threshold": threshold,
        "cost_ratio_ASSUMED": info["cost_ratio_ASSUMED"],
        "filled_with_typical_values": filled,
        "note": DISCLAIMER,
    }


def explain(fields: dict, top_n: int = 5) -> dict:
    """Top SHAP drivers for one loan, largest effect first."""
    X, filled = validate_loan(fields)
    _, explainer, _ = _load_models()
    drivers = explainer.explain_one(X, top_n=top_n)
    for d in drivers:
        d["value"] = _to_python(d["value"])
        d["filled_with_typical_value"] = d["feature"] in filled
    return {"drivers": drivers, "filled_with_typical_values": filled, "note": DISCLAIMER}


if __name__ == "__main__":
    example = {
        "credit_score": 700, "orig_ltv": 90, "orig_dti": 40, "orig_upb": 250000,
        "orig_interest_rate": 4.5, "loan_purpose": "P", "occupancy_status": "P",
        "property_state": "FL",
    }
    print(json.dumps(score(example), indent=2))
    print(json.dumps(explain(example), indent=2))