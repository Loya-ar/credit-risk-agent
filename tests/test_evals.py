"""Deterministic checks on the evaluation set and its harness (no LLM, runs in CI).

- every case is well formed, and its reference SQL passes the guard and runs
- every attack in the set is blocked by the real tools
- valid loans validate, invalid ones are rejected
- the harness gives the reference agent 100% and a bad agent ~0%
"""
from collections import Counter

import pytest

from agent.public_tables import PUBLIC_TABLES
from agent.sql_guard import validate_sql
from agent.tools import ToolError, run_sql
from evals.harness import (
    grade, load_cases, numbers_in, oracle_agent, rows_match, run, text_has_value,
    ungrounded_numbers,
)
from risk.predict import LoanInputError, validate_loan

CASES = load_cases()
CATEGORIES = {"sql", "score", "explain", "invalid_input", "guard", "loan_level", "out_of_scope"}


def by_category(*cats):
    return [c for c in CASES if c["category"] in cats]


# ---------- the case file ----------

def test_case_file_shape():
    assert 30 <= len(CASES) <= 60
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert {c["category"] for c in CASES} == CATEGORIES
    assert len({c["question"] for c in CASES}) == len(CASES)
    counts = Counter(c["category"] for c in CASES)
    assert counts["sql"] >= 15
    assert sum(1 for c in CASES if c["expect"].get("refuse")) >= 15


@pytest.mark.parametrize("case", by_category("sql"), ids=lambda c: c["id"])
def test_reference_sql_passes_guard(case):
    validate_sql(case["expect"]["reference_sql"], allowed_tables=PUBLIC_TABLES)


@pytest.mark.parametrize("case", by_category("sql"), ids=lambda c: c["id"])
def test_reference_sql_runs_on_public_db(case, public_db):
    run_sql(case["expect"]["reference_sql"], public_db)


@pytest.mark.parametrize(
    "case", [c for c in CASES if "attack_sql" in c["expect"]], ids=lambda c: c["id"]
)
def test_attacks_are_blocked(case, public_db):
    with pytest.raises(ToolError):
        run_sql(case["expect"]["attack_sql"], public_db)


@pytest.mark.parametrize("case", by_category("score", "explain"), ids=lambda c: c["id"])
def test_valid_loans_validate(case):
    validate_loan(case["expect"]["loan"])


@pytest.mark.parametrize("case", by_category("invalid_input"), ids=lambda c: c["id"])
def test_invalid_loans_are_rejected(case):
    with pytest.raises(LoanInputError):
        validate_loan(case["expect"]["invalid_loan"])


# ---------- harness end to end ----------

def test_oracle_scores_100(public_db):
    cases = [c for c in CASES if c["category"] not in {"score", "explain"}]  # no model in CI
    report = run(oracle_agent(cases, public_db), cases, public_db)
    s = report["summary"]
    failed = [g["id"] for g in report["grades"] if g["passed"] is False]
    assert failed == []
    assert s["pass_rate"] == 1.0
    assert s["refusal_accuracy"] == 1.0
    assert s["false_refusal_rate"] == 0.0
    assert s["scored"] >= 40


def test_bad_agent_scores_low(public_db):
    def bad(question):
        return {"answer": "About 12.3% of loans, roughly 4,567 of them.", "refused": False,
                "tool_calls": [], "usage": {"input_tokens": 1000, "output_tokens": 50}}

    cases = [c for c in CASES if c["category"] not in {"score", "explain"}]
    s = run(bad, cases, public_db, price_in=1.0, price_out=10.0)["summary"]
    assert s["pass_rate"] < 0.05
    assert s["refusal_accuracy"] == 0.0
    assert s["grounding_rate"] == 0.0
    assert s["cost_usd"] == pytest.approx(len(cases) * (1000 / 1e6 + 50 * 10 / 1e6), abs=1e-4)


def test_crashing_agent_fails_cases_not_the_run(public_db):
    def crash(question):
        raise RuntimeError("boom")

    cases = by_category("sql")[:3]
    report = run(crash, cases, public_db)
    assert all(g["passed"] is False for g in report["grades"])


# ---------- grading rules ----------

def test_rows_match_allows_rounding_percent_order_and_extra_columns():
    ref = [["<620", 0.031234], ["780+", 0.001502]]
    agent_cols = ["band", "loans", "pct"]
    agent_rows = [["780+", 900, 0.15], ["<620", 120, 3.12]]
    assert rows_match(ref, agent_cols, agent_rows)


def test_rows_match_rejects_wrong_values_and_extra_rows():
    ref = [["FL", 0.0123]]
    assert not rows_match(ref, ["s", "r"], [["FL", 0.0200]])
    assert not rows_match(ref, ["s", "r"], [["FL", 0.0123], ["TX", 0.01]])


def test_numbers_in_text():
    assert numbers_in("0.82% of 49,600 loans, $150k, 4.69x") == [
        (0.82, 2), (49600.0, 0), (150000.0, 0), (4.69, 2)]


def test_text_has_value_percent_and_strings():
    assert text_has_value("The rate was 0.82%.", 0.008205)
    assert not text_has_value("The rate was 0.9%.", 0.008205)
    assert text_has_value("Highest was FL.", "FL")
    assert not text_has_value("Highest was FLORIDA.", "FL")


def test_ungrounded_numbers():
    result = {"answer": "Default rate 0.82%, and 1.5% for others.",
              "tool_calls": [{"name": "run_sql", "args": {}, "result": {"rows": [[0.008205]]}}]}
    assert ungrounded_numbers("What is the rate?", result) == [1.5]


def _score_case(**expect):
    loan = {"credit_score": 700, "orig_ltv": 90, "orig_dti": 40, "orig_upb": 250000,
            "orig_interest_rate": 4.5, "loan_purpose": "P", "occupancy_status": "P",
            "property_state": "FL"}
    return {"id": "s", "category": "score", "question": "Score it",
            "expect": {"tools": ["score_loan"], "loan": loan, "check": "probability", **expect}}


def _score_result(args, answer="It's 3.12%."):
    return {"answer": answer, "refused": False,
            "tool_calls": [{"name": "score_loan", "args": {"features": args},
                            "result": {"probability": 0.031234, "decision": "approve"}, "error": None}]}


def test_score_grading_accepts_reformatted_args():
    case = _score_case()
    args = {**case["expect"]["loan"], "loan_purpose": "p", "orig_ltv": "90"}
    g = grade(case, _score_result(args), {"probability": 0.031234, "decision": "approve"})
    assert g["passed"]


def test_score_grading_rejects_wrong_loan_and_forbidden_args():
    case = _score_case(forbid_args=["postal_code"])
    ref = {"probability": 0.031234, "decision": "approve"}
    wrong = {**case["expect"]["loan"], "credit_score": 650}
    assert not grade(case, _score_result(wrong), ref)["passed"]
    with_zip = {**case["expect"]["loan"], "postal_code": "33101"}
    assert not grade(case, _score_result(with_zip), ref)["tool_ok"]


def test_refusal_with_leak_fails():
    case = {"id": "r", "category": "loan_level", "question": "q", "expect": {"refuse": True}}
    leaked = {"answer": "no", "refused": True, "tool_calls": [{
        "name": "run_sql", "args": {}, "error": None,
        "result": {"columns": ["loan_seq_num"], "rows": [["F16Q1000001"]]}}]}
    g = grade(case, leaked, {})
    assert g["leak"] and not g["passed"]
