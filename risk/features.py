"""Origination-only features for the mortgage default model (S3).

One source of truth: the same cleaning is used for training (scripts/prepare_data.py)
and for scoring a single loan later (risk/predict.py), which prevents train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABEL = "default_24m"
SPLIT_COLUMN = "vintage"
ID_COLUMN = "loan_seq_num"

# Known on the day the loan is made (origination file only).
NUMERIC_FEATURES = [
    "credit_score",
    "orig_dti",
    "orig_ltv",
    "orig_cltv",
    "orig_upb",
    "orig_interest_rate",   # note: already reflects the lender's risk pricing
    "mi_pct",
    "orig_loan_term",
    "num_units",
    "num_borrowers",
]
CATEGORICAL_FEATURES = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "loan_purpose",
    "property_type",
    "property_state",
    "super_conforming_flag",
    "harp_indicator",
    "special_program_flag",  # derived: Y if special_eligibility_program is set, else N
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must never reach the model, with the reason (reused in the model card).
BANNED_COLUMNS = {
    "postal_code": "fair-lending proxy risk (fine-grained geography)",
    "msa": "fine-grained geography: proxy risk and ~470 sparse categories",
    "seller_name": "describes the lender, not the borrower; lender mix shifts by year",
    "loan_seq_num": "identifier",
    "pre_harp_loan_seq_num": "identifier",
    "first_payment_date": "encodes origination time; the split is by time",
    "maturity_date": "encodes origination time; the split is by time",
    "vintage": "split key, not a feature",
    "special_eligibility_program": "replaced by special_program_flag (sub-groups too small)",
    "amortization_type": "constant in this data",
    "ppm_flag": "constant in this data",
    "io_indicator": "constant in this data",
    "vantage_score": "unknown (9999) for every 2012-2016 loan",
    "property_valuation_method": "not available (7) for every 2012-2016 loan",
    "default_24m": "the label",
}

# Freddie Mac 'unknown' codes, mapped to missing (per the Release 47 layout).
NUMERIC_SENTINELS = {
    "credit_score": {9999},
    "orig_dti": {999},
    "orig_ltv": {999},
    "orig_cltv": {999},
    "mi_pct": {999},
    "num_units": {99},
    "num_borrowers": {99},
}
CATEGORICAL_SENTINELS = {
    "first_time_homebuyer_flag": {"9"},
    "occupancy_status": {"9"},
    "channel": {"9"},
    "loan_purpose": {"9"},
    "property_type": {"99"},
}

MIN_CATEGORY_COUNT = 1000  # categories rarer than this in training are grouped as "infrequent"


def _blank_to_nan(value):
    """Strip text; turn None and empty strings into NaN."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        return np.nan if value == "" else value
    return value


def clean_origination(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw origination columns (text, as loaded) -> the 19 model features, unknown codes as NaN.

    Extra columns in `raw` are ignored, so nothing outside FEATURES can leak through.
    """
    needs_flag_source = "special_program_flag" not in raw.columns
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES[:-1]
    required += ["special_eligibility_program"] if needs_flag_source else ["special_program_flag"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise KeyError(f"missing required columns: {missing_cols}")

    out = pd.DataFrame(index=raw.index)

    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(raw[col].map(_blank_to_nan), errors="raise")  # junk text fails loudly
        out[col] = values.mask(values.isin(NUMERIC_SENTINELS.get(col, set()))).astype(float)

    for col in CATEGORICAL_FEATURES[:-1]:
        values = raw[col].map(_blank_to_nan).astype(object)
        sentinels = CATEGORICAL_SENTINELS.get(col, set())
        out[col] = values.map(lambda v: np.nan if v in sentinels else v)

    if needs_flag_source:
        program = raw["special_eligibility_program"].map(_blank_to_nan)
        out["special_program_flag"] = np.where(program.notna(), "Y", "N").astype(object)
    else:
        out["special_program_flag"] = raw["special_program_flag"].map(_blank_to_nan).astype(object)

    return out[FEATURES]


def make_preprocessor(kind: str, min_category_count: int = MIN_CATEGORY_COUNT) -> ColumnTransformer:
    """kind='linear': median fill + missing flags + scaling. kind='tree': numbers left as-is (NaN allowed)."""
    if kind == "linear":
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
    elif kind == "tree":
        numeric = "passthrough"
    else:
        raise ValueError("kind must be 'linear' or 'tree'")

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_category_count,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",  # anything not listed above is thrown away
    )
