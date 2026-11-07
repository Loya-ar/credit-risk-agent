"""A1d: the three tools the agent can call.

run_sql(query)          -> guarded, read-only SQL. By default on the public,
                           aggregate-only DB (agent/public_tables.py); set
                           AGENT_DB=local for the loan-level data/agent.duckdb
score_loan(features)    -> calibrated default probability + decision at the threshold
explain_score(features) -> top 5 SHAP drivers in plain words

Every failure is raised as ToolError with a message the LLM can act on
(A2's repair node shows it to the model and retries once).
"""
from __future__ import annotations

import os
import threading
from decimal import Decimal
from pathlib import Path

import duckdb

from agent.public_tables import ensure_public_db
from agent.sql_guard import GuardError, validate_sql
from risk import predict

ROOT = Path(__file__).resolve().parents[1]
AGENT_DB_PATH = ROOT / "data" / "agent.duckdb"  # loan-level, local only
TIMEOUT_SECONDS = 5.0

FEATURE_LABELS = {
    "credit_score": "credit score",
    "orig_dti": "debt-to-income ratio (%)",
    "orig_ltv": "loan-to-value ratio (%)",
    "orig_cltv": "combined loan-to-value ratio (%)",
    "orig_upb": "original loan amount ($)",
    "orig_interest_rate": "original interest rate (%)",
    "mi_pct": "mortgage insurance coverage (%)",
    "orig_loan_term": "loan term (months)",
    "num_units": "number of units",
    "num_borrowers": "number of borrowers",
    "first_time_homebuyer_flag": "first-time homebuyer",
    "occupancy_status": "occupancy (P primary, S second home, I investment)",
    "channel": "origination channel (R retail, B broker, C correspondent)",
    "loan_purpose": "loan purpose (P purchase, C cash-out refi, N no-cash-out refi)",
    "property_type": "property type (SF single-family, CO condo, PU planned unit, MH manufactured, CP co-op)",
    "property_state": "property state",
    "super_conforming_flag": "super-conforming loan",
    "harp_indicator": "HARP refinance",
    "special_program_flag": "special program flag",
}


class ToolError(Exception):
    """A tool failed. The message is written for the LLM to fix its call."""


# ---------- run_sql ----------

def _connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Second safety layer: the database itself refuses writes and file access."""
    return duckdb.connect(
        str(db_path), read_only=True, config={"enable_external_access": False}
    )


def _plain(value):
    return float(value) if isinstance(value, Decimal) else value


def _execute(con, sql: str, timeout: float) -> tuple[list[str], list[list]]:
    """Run a query, stopping it if it takes longer than `timeout` seconds."""
    timed_out = threading.Event()

    def stop():
        timed_out.set()
        con.interrupt()

    timer = threading.Timer(timeout, stop)
    timer.start()
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = [[_plain(v) for v in row] for row in cursor.fetchall()]
    except duckdb.Error as e:
        if timed_out.is_set():
            raise ToolError(
                f"Query took longer than {timeout:g} s and was stopped. "
                "Simplify it or aggregate more."
            ) from None
        raise ToolError(f"SQL error: {e}") from None
    finally:
        timer.cancel()
    return columns, rows


def default_db_path() -> Path:
    """Public aggregate DB unless AGENT_DB=local (loan-level data, never deployed)."""
    mode = os.environ.get("AGENT_DB", "public").lower()
    if mode == "local":
        if not AGENT_DB_PATH.exists():
            raise ToolError(
                f"Agent database not found at {AGENT_DB_PATH}. Run: python -m scripts.build_agent_db"
            )
        return AGENT_DB_PATH
    if mode != "public":
        raise ToolError(f"AGENT_DB must be 'public' or 'local', got '{mode}'.")
    try:
        return ensure_public_db()
    except FileNotFoundError as e:
        raise ToolError(f"Public tables are not available: {e}") from None


def run_sql(query: str, db_path: Path | None = None, timeout: float = TIMEOUT_SECONDS) -> dict:
    db_path = default_db_path() if db_path is None else Path(db_path)
    if not db_path.exists():
        raise ToolError(f"Agent database not found at {db_path}.")

    con = _connect(db_path)
    try:
        # The guard allows exactly the tables this database has, so the public DB
        # can never be asked for loan rows.
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        try:
            safe_sql = validate_sql(query, allowed_tables=tables)
        except GuardError as e:
            raise ToolError(f"SQL rejected by guard: {e}") from None
        columns, rows = _execute(con, safe_sql, timeout)
    finally:
        con.close()
    return {"sql": safe_sql, "columns": columns, "rows": rows, "row_count": len(rows)}


# ---------- score_loan / explain_score ----------

def _check_features(features) -> dict:
    if not isinstance(features, dict):
        raise ToolError(
            f"features must be an object of field: value pairs, got {type(features).__name__}."
        )
    return features


def score_loan(features: dict) -> dict:
    features = _check_features(features)
    try:
        return predict.score(features)
    except predict.LoanInputError as e:
        raise ToolError(str(e)) from None
    except FileNotFoundError as e:
        raise ToolError(f"Model files not found ({e}). The scoring tool is unavailable.") from None


def explain_score(features: dict) -> dict:
    features = _check_features(features)
    try:
        result = predict.explain(features)
    except predict.LoanInputError as e:
        raise ToolError(str(e)) from None
    except FileNotFoundError as e:
        raise ToolError(f"Model files not found ({e}). The explain tool is unavailable.") from None

    summary = []
    for d in result["drivers"]:
        label = FEATURE_LABELS.get(d["feature"], d["feature"])
        assumed = " (typical value, not provided)" if d["filled_with_typical_value"] else ""
        summary.append(f"{label} = {d['value']}{assumed}: {d['direction']}")
    result["summary"] = summary
    return result