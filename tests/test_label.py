"""S2 tests for the 24-month default label.

Part 1: hand-built synthetic loans with known answers (runs anywhere, including CI).
Part 2: invariants on the real labels table (skipped if the local DuckDB file is absent).
"""
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LABEL_SQL = (ROOT / "sql" / "build_label.sql").read_text()
DB_PATH = ROOT / "data" / "credit.duckdb"


def months(loan_id, ages, dq_by_age=None, zb=None):
    """Monthly rows for one synthetic loan: status '00' unless overridden; optional (age, code) termination row."""
    dq_by_age = dq_by_age or {}
    rows = [(loan_id, 2012, str(a), dq_by_age.get(a, "00"), None) for a in ages]
    if zb is not None:
        age, code = zb
        rows.append((loan_id, 2012, str(age), None, code))
    return rows


# loan_id -> (monthly rows, expected label_reason, expected default_24m)
SYNTHETIC = {
    "L01": (months("L01", range(0, 25)), "good_performing_24m", 0),
    "L02": (months("L02", range(0, 25), {8: "01", 9: "02", 10: "03"}), "default_dq90", 1),
    "L03": (months("L03", range(0, 25), {5: "01", 6: "02"}), "good_performing_24m", 0),          # 60 days, cured
    "L04": (months("L04", range(0, 12), zb=(12, "01")), "good_prepaid", 0),
    "L05": (months("L05", range(0, 8), {3: "01", 4: "02", 5: "03"}, zb=(8, "01")), "default_dq90", 1),  # 90+ then prepaid
    "L06": (months("L06", range(0, 15), {13: "01", 14: "02"}, zb=(15, "03")), "default_credit_event", 1),  # short sale
    "L07": (months("L07", range(0, 10)), "excluded_censored", None),                              # only 10 months seen
    "L08": (months("L08", range(0, 25), {20: "RA"}), "default_dq90", 1),                         # REO acquisition
    "L09": (months("L09", range(0, 6), zb=(6, "96")), "excluded_other_termination", None),       # defect buyback
    "L10": (months("L10", range(0, 31), {30: "03"}), "good_performing_24m", 0),                  # 90+ only after month 24
    "L11": (months("L11", range(0, 25), {7: "XX", 8: "XX"}), "good_performing_24m", 0),          # unknown months
    "L12": (months("L12", range(30, 40)), "excluded_no_window_records", None),                   # records start late
    "L13": (months("L13", range(0, 25), {22: "10"}), "default_dq90", 1),                         # '10' must beat '3'
}


@pytest.fixture(scope="module")
def synthetic_labels() -> pd.DataFrame:
    perf_rows = [row for rows, _, _ in SYNTHETIC.values() for row in rows]
    perf = pd.DataFrame(perf_rows, columns=[
        "loan_seq_num", "vintage", "loan_age", "current_loan_delinquency_status", "zero_balance_code",
    ])
    orig = pd.DataFrame({"loan_seq_num": list(SYNTHETIC), "vintage": 2012})

    con = duckdb.connect()  # in-memory database
    con.register("perf_df", perf)
    con.register("orig_df", orig)
    con.execute("CREATE TABLE perf_raw AS SELECT * FROM perf_df")
    con.execute("CREATE TABLE orig_raw AS SELECT * FROM orig_df")
    con.execute(LABEL_SQL)
    labels = con.sql("SELECT * FROM labels").df().set_index("loan_seq_num")
    con.close()
    return labels


@pytest.mark.parametrize("loan_id", list(SYNTHETIC))
def test_synthetic_loan_gets_expected_label(synthetic_labels, loan_id):
    _, expected_reason, expected_label = SYNTHETIC[loan_id]
    row = synthetic_labels.loc[loan_id]
    assert row["label_reason"] == expected_reason
    if expected_label is None:
        assert pd.isna(row["default_24m"])
    else:
        assert row["default_24m"] == expected_label


def test_synthetic_one_row_per_loan(synthetic_labels):
    assert len(synthetic_labels) == len(SYNTHETIC)
    assert synthetic_labels.index.is_unique


# ---------- real data (local only) ----------

needs_real_data = pytest.mark.skipif(not DB_PATH.exists(), reason="local DuckDB not built (CI has no data)")


@pytest.fixture(scope="module")
def real_con():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


@needs_real_data
def test_real_one_label_row_per_loan(real_con):
    n_orig = real_con.execute("SELECT COUNT(*) FROM orig_raw").fetchone()[0]
    n_labels, n_distinct = real_con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT loan_seq_num) FROM labels"
    ).fetchone()
    assert n_labels == n_orig == n_distinct


@needs_real_data
def test_real_labels_are_0_1_or_null(real_con):
    bad = real_con.execute(
        "SELECT COUNT(*) FROM labels WHERE default_24m IS NOT NULL AND default_24m NOT IN (0, 1)"
    ).fetchone()[0]
    assert bad == 0


@needs_real_data
def test_real_excluded_loans_have_no_label(real_con):
    bad = real_con.execute("""
        SELECT COUNT(*) FROM labels
        WHERE (label_reason LIKE 'excluded%') <> (default_24m IS NULL)
    """).fetchone()[0]
    assert bad == 0


@needs_real_data
def test_real_dq90_defaults_happen_inside_window(real_con):
    bad = real_con.execute("""
        SELECT COUNT(*) FROM labels
        WHERE label_reason = 'default_dq90'
          AND NOT (first_dq90_age BETWEEN 0 AND 24)
    """).fetchone()[0]
    assert bad == 0


@needs_real_data
def test_real_performing_goods_were_observed_to_24_with_no_dq90(real_con):
    bad = real_con.execute("""
        SELECT COUNT(*) FROM labels
        WHERE label_reason = 'good_performing_24m'
          AND (max_age_in_window <> 24 OR first_dq90_age IS NOT NULL)
    """).fetchone()[0]
    assert bad == 0