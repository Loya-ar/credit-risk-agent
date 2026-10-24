"""S3 leakage tests: nothing from the performance file, and no banned column, reaches the model."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk.features import (
    BANNED_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURES,
    LABEL,
    NUMERIC_FEATURES,
    clean_origination,
    make_preprocessor,
)

ROOT = Path(__file__).resolve().parents[1]
LOAD_SQL = (ROOT / "sql" / "load_raw.sql").read_text()


def columns_of(table: str) -> set[str]:
    """Read column names for a table straight from load_raw.sql (single source of truth)."""
    block = LOAD_SQL.split(f"CREATE OR REPLACE TABLE {table}")[1].split("CREATE OR REPLACE TABLE")[0]
    return set(re.findall(r"'(\w+)':\s*'VARCHAR'", block))


ORIG_COLUMNS = columns_of("orig_raw")
PERF_COLUMNS = columns_of("perf_raw")
DERIVED_FEATURES = {"special_program_flag"}


def test_layout_was_parsed():
    assert len(ORIG_COLUMNS) == 31
    assert len(PERF_COLUMNS) == 35


def test_no_performance_file_column_is_a_feature():
    assert not (set(FEATURES) & PERF_COLUMNS)


def test_no_banned_column_is_a_feature():
    assert not (set(FEATURES) & set(BANNED_COLUMNS))


def test_label_is_not_a_feature():
    assert LABEL not in FEATURES


def test_every_feature_comes_from_origination_file():
    assert set(FEATURES) - DERIVED_FEATURES <= ORIG_COLUMNS


def test_feature_list_has_no_duplicates():
    assert len(FEATURES) == len(set(FEATURES))


def raw_rows() -> pd.DataFrame:
    """Two raw rows shaped like orig_raw (all text), plus columns that must NOT leak through."""
    normal = {c: "" for c in ORIG_COLUMNS}
    normal.update({
        "credit_score": "720", "orig_dti": "35", "orig_ltv": "80", "orig_cltv": "80",
        "orig_upb": "200000", "orig_interest_rate": "4.0", "mi_pct": "0", "orig_loan_term": "360",
        "num_units": "1", "num_borrowers": "2", "first_time_homebuyer_flag": "N",
        "occupancy_status": "P", "channel": "R", "loan_purpose": "P", "property_type": "SF",
        "property_state": "CA", "super_conforming_flag": "N", "harp_indicator": "N",
        "special_eligibility_program": "", "postal_code": "941", "seller_name": "OTHER",
    })
    unknowns = dict(normal)
    unknowns.update({
        "credit_score": "9999", "orig_dti": "999", "orig_ltv": "999", "orig_cltv": "999",
        "mi_pct": "999", "num_units": "99", "num_borrowers": "99",
        "first_time_homebuyer_flag": "9", "occupancy_status": "9", "channel": "9",
        "loan_purpose": "9", "property_type": "99", "property_state": "FL",
        "harp_indicator": "Y", "special_eligibility_program": "H",
    })
    df = pd.DataFrame([normal, unknowns])
    df["current_loan_delinquency_status"] = ["00", "03"]  # performance-file column
    df[LABEL] = [0, 1]                                      # the label
    return df


def test_clean_output_is_exactly_the_feature_list():
    assert list(clean_origination(raw_rows()).columns) == FEATURES


def test_unknown_codes_become_missing():
    clean = clean_origination(raw_rows())
    unknown_row = clean.iloc[1]
    for col in NUMERIC_FEATURES:
        if col not in ("orig_upb", "orig_interest_rate", "orig_loan_term"):
            assert np.isnan(unknown_row[col]), col
    for col in ["first_time_homebuyer_flag", "occupancy_status", "channel", "loan_purpose", "property_type"]:
        assert pd.isna(unknown_row[col]), col
    assert clean.iloc[0]["credit_score"] == 720


def test_special_program_flag_is_derived():
    clean = clean_origination(raw_rows())
    assert list(clean["special_program_flag"]) == ["N", "Y"]


def test_junk_in_numeric_column_fails_loudly():
    bad = raw_rows()
    bad.loc[0, "credit_score"] = "abc"
    with pytest.raises(ValueError):
        clean_origination(bad)


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_model_inputs_trace_back_only_to_allowed_features(kind):
    pre = make_preprocessor(kind, min_category_count=1)
    pre.fit(clean_origination(raw_rows()))
    for name in pre.get_feature_names_out():
        source = name.split("__", 1)[1].replace("missingindicator_", "")
        assert any(source == f or source.startswith(f + "_") for f in FEATURES), name