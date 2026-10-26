import pytest

from agent.sql_guard import DEFAULT_LIMIT, MAX_LIMIT, GuardError, validate_sql


# ---------- allowed queries ----------

def test_adds_default_limit():
    sql = validate_sql("SELECT vintage, AVG(default_24m) FROM loans GROUP BY vintage")
    assert f"LIMIT {DEFAULT_LIMIT}" in sql


def test_keeps_small_limit():
    assert "LIMIT 50" in validate_sql("SELECT * FROM loans LIMIT 50")


def test_caps_large_limit():
    sql = validate_sql("SELECT * FROM loans LIMIT 5000")
    assert f"LIMIT {MAX_LIMIT}" in sql
    assert "5000" not in sql


def test_allows_join_of_allowed_tables():
    validate_sql(
        "SELECT l.vintage, AVG(t.probability) FROM loans l "
        "JOIN test_scores t ON l.loan_seq_num = t.loan_seq_num GROUP BY l.vintage"
    )


def test_allows_cte():
    validate_sql(
        "WITH d AS (SELECT vintage, default_24m FROM loans) "
        "SELECT vintage, AVG(default_24m) FROM d GROUP BY vintage"
    )


def test_table_names_case_insensitive():
    validate_sql("SELECT COUNT(*) FROM LOANS")


def test_allows_single_trailing_semicolon():
    validate_sql("SELECT COUNT(*) FROM loans;")


# ---------- blocked statements ----------

@pytest.mark.parametrize("query", [
    "DROP TABLE loans",
    "DELETE FROM loans",
    "INSERT INTO loans VALUES (1)",
    "UPDATE loans SET vintage = 0",
    "CREATE TABLE x AS SELECT * FROM loans",
    "ATTACH 'other.db'",
    "COPY loans TO 'out.csv'",
    "PRAGMA show_tables",
])
def test_rejects_non_select(query):
    with pytest.raises(GuardError):
        validate_sql(query)


def test_rejects_multiple_statements():
    with pytest.raises(GuardError, match="exactly 1"):
        validate_sql("SELECT * FROM loans; DROP TABLE loans")


def test_rejects_unknown_table():
    # orig_raw exists in credit.duckdb but must never be reachable by the agent
    with pytest.raises(GuardError, match="Unknown table"):
        validate_sql("SELECT * FROM orig_raw")


def test_rejects_qualified_table():
    with pytest.raises(GuardError):
        validate_sql("SELECT * FROM other_db.loans")


@pytest.mark.parametrize("query", [
    "SELECT * FROM read_csv('secrets.csv')",
    "SELECT * FROM read_parquet('data/x.parquet')",
    "SELECT * FROM loans WHERE vintage IN (SELECT 1 FROM glob('*'))",
])
def test_rejects_file_functions(query):
    with pytest.raises(GuardError):
        validate_sql(query)


def test_rejects_non_integer_limit():
    with pytest.raises(GuardError):
        validate_sql("SELECT * FROM loans LIMIT 'abc'")


@pytest.mark.parametrize("query", ["", "   ", "this is not sql"])
def test_rejects_empty_or_garbage(query):
    with pytest.raises(GuardError):
        validate_sql(query)