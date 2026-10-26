from pathlib import Path

import duckdb
import pytest

from agent.tools import (
    FEATURE_LABELS, ToolError, _connect, _execute, explain_score, run_sql, score_loan,
)
from risk.features import FEATURES
from risk.predict import CALIBRATED_PATH, RAW_PATH

needs_models = pytest.mark.skipif(
    not (CALIBRATED_PATH.exists() and RAW_PATH.exists()),
    reason="trained models not present (gitignored)",
)

LOAN = {
    "credit_score": 700, "orig_ltv": 90, "orig_dti": 40, "orig_upb": 250000,
    "orig_interest_rate": 4.5, "loan_purpose": "P", "occupancy_status": "P",
    "property_state": "FL",
}


@pytest.fixture
def tiny_db(tmp_path) -> Path:
    """A 3-loan agent database, so these tests run without the real data (e.g. in CI)."""
    path = tmp_path / "agent.duckdb"
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE loans AS SELECT * FROM (VALUES
            ('A', 2016, 650.0, 1), ('B', 2016, 760.0, 0), ('C', 2015, 720.0, 0)
        ) t(loan_seq_num, vintage, credit_score, default_24m)
    """)
    con.execute("""
        CREATE TABLE test_scores AS SELECT * FROM (VALUES
            ('A', 0.06, 10, TRUE), ('B', 0.002, 1, FALSE)
        ) t(loan_seq_num, probability, decile, declined_at_threshold)
    """)
    con.close()
    return path


# ---------- run_sql ----------

def test_run_sql_returns_rows(tiny_db):
    result = run_sql("SELECT vintage, COUNT(*) AS n FROM loans GROUP BY vintage ORDER BY vintage", tiny_db)
    assert result["columns"] == ["vintage", "n"]
    assert result["rows"] == [[2015, 1], [2016, 2]]
    assert result["row_count"] == 2


def test_run_sql_adds_limit(tiny_db):
    assert "LIMIT 200" in run_sql("SELECT * FROM loans", tiny_db)["sql"]


def test_run_sql_guard_rejection(tiny_db):
    with pytest.raises(ToolError, match="guard"):
        run_sql("DROP TABLE loans", tiny_db)


def test_run_sql_multi_statement_injection(tiny_db):
    with pytest.raises(ToolError, match="guard"):
        run_sql("SELECT * FROM loans; DELETE FROM loans", tiny_db)


def test_run_sql_unknown_column(tiny_db):
    with pytest.raises(ToolError, match="SQL error"):
        run_sql("SELECT no_such_column FROM loans", tiny_db)


def test_run_sql_missing_db(tmp_path):
    with pytest.raises(ToolError, match="build_agent_db"):
        run_sql("SELECT * FROM loans", tmp_path / "missing.duckdb")


# ---------- second safety layer (guard bypassed on purpose) ----------

def test_connection_is_read_only(tiny_db):
    con = _connect(tiny_db)
    try:
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE x (a INTEGER)")
    finally:
        con.close()


def test_connection_blocks_file_access(tiny_db, tmp_path):
    csv = tmp_path / "secret.csv"
    csv.write_text("a\n1\n")
    query = f"SELECT * FROM read_csv('{csv.as_posix()}')"
    assert duckdb.connect().execute(query).fetchall() == [(1,)]  # a normal connection can read it
    con = _connect(tiny_db)
    try:
        with pytest.raises(duckdb.Error):
            con.execute(query)  # ours can't
    finally:
        con.close()


def test_timeout_stops_long_query(tiny_db):
    con = _connect(tiny_db)
    try:
        with pytest.raises(ToolError, match="longer than"):
            _execute(con, "SELECT SUM(a.range * b.range) FROM range(100000000) a, range(100000) b", timeout=0.2)
    finally:
        con.close()


# ---------- score_loan / explain_score ----------

def test_score_loan_rejects_missing_fields():
    with pytest.raises(ToolError, match="Missing required"):
        score_loan({"credit_score": 700})


def test_score_loan_rejects_non_dict():
    with pytest.raises(ToolError, match="must be an object"):
        score_loan("credit score 700")


def test_every_feature_has_a_label():
    assert set(FEATURE_LABELS) == set(FEATURES)


@needs_models
def test_score_loan_returns_decision():
    result = score_loan(LOAN)
    assert result["decision"] in {"approve", "decline"}
    assert result["note"] == "Demo only, not a credit decision."


@needs_models
def test_explain_score_has_five_plain_drivers():
    result = explain_score(LOAN)
    assert len(result["summary"]) == 5
    assert all(("raises risk" in s) or ("lowers risk" in s) for s in result["summary"])