"""Validate SQL written by the agent before it touches the database.

Rules: exactly one SELECT statement, only allowlisted tables, no
file/table functions, and a LIMIT (default 200, never above 1000).
The GuardError message is shown to the LLM so it can repair the query.
"""
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

ALLOWED_TABLES = {"loans", "test_scores"}
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000

# Backup layer: DuckDB functions that read files, other databases, or the environment.
# The read-only connection with external access disabled is the main protection.
BLOCKED_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan",
    "read_json", "read_json_auto", "read_ndjson", "read_text", "read_blob",
    "glob", "sniff_csv", "query", "query_table", "getenv",
}


class GuardError(ValueError):
    """A query broke a guard rule."""


def _function_name(func: exp.Func) -> str:
    # Functions sqlglot doesn't know are "Anonymous" and keep their name in .name
    if isinstance(func, exp.Anonymous):
        return func.name.lower()
    return func.sql_name().lower()


def validate_sql(query: str) -> str:
    """Return a safe version of the query (with a LIMIT) or raise GuardError."""
    if not query or not query.strip():
        raise GuardError("Empty query. Write one SELECT statement.")

    try:
        statements = [s for s in sqlglot.parse(query, read="duckdb") if s is not None]
    except ParseError as e:
        raise GuardError(f"Could not parse SQL: {e}") from e

    if len(statements) != 1:
        raise GuardError(
            f"Expected exactly 1 statement, got {len(statements)}. "
            "Remove extra statements and semicolons."
        )
    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise GuardError(f"Only SELECT queries are allowed (got {type(tree).__name__}).")

    # Names defined in WITH ... AS (...) are allowed as table names.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise GuardError(
                "Table functions are not allowed. Query loans or test_scores directly."
            )
        if table.db or table.catalog:
            raise GuardError(f"Use plain table names, not '{table.sql()}'.")
        name = table.name.lower()
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise GuardError(
                f"Unknown table '{table.name}'. Allowed tables: {sorted(ALLOWED_TABLES)}."
            )

    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name in BLOCKED_FUNCTIONS:
            raise GuardError(f"Function '{name}' is not allowed.")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(DEFAULT_LIMIT)
    else:
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            raise GuardError("LIMIT must be a whole number.")
        if int(value.name) > MAX_LIMIT:
            tree = tree.limit(MAX_LIMIT)

    return tree.sql(dialect="duckdb")