"""The agent graph with a scripted fake model: routing, repair, grounding, refusals.

No API key or network needed; the real tools run against the synthetic public DB.
"""
import pytest
from langchain_core.messages import AIMessage

from agent import graph as agent_graph
from agent import tools as agent_tools
from evals.harness import grade

LOAN = {"credit_score": 700, "orig_ltv": 90, "orig_dti": 40, "orig_upb": 250000,
        "orig_interest_rate": 4.5, "loan_purpose": "P", "occupancy_status": "P",
        "property_state": "FL"}


class ScriptedModel:
    """Returns the scripted messages in order and records what it was sent."""

    def __init__(self, *messages):
        self.script = list(messages)
        self.seen = []

    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        msg = self.script.pop(0)
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        return msg


def call(name, i=0, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


@pytest.fixture(autouse=True)
def public(monkeypatch, public_db):
    monkeypatch.delenv("AGENT_DB", raising=False)
    monkeypatch.setattr(agent_tools, "ensure_public_db", lambda: public_db)


def rate_2016(public_db):
    return agent_tools.run_sql(
        "SELECT default_rate FROM segments WHERE dimension = 'all' AND vintage = 2016", public_db
    )["rows"][0][0]


def test_binds_the_four_tools():
    model = ScriptedModel(AIMessage("ok"))
    agent_graph.answer("hi", model=model)
    assert model.tool_names == ["run_sql", "score_loan", "explain_score", "decline"]


def test_sql_then_grounded_answer(public_db):
    rate = rate_2016(public_db)
    model = ScriptedModel(
        call("run_sql", query="SELECT default_rate FROM segments WHERE dimension = 'all' AND vintage = 2016"),
        AIMessage(f"The 2016 default rate was {rate:.2%}."),
    )
    out = agent_graph.answer("What was the 2016 default rate?", model=model)
    assert not out["refused"]
    assert out["answer"] == f"The 2016 default rate was {rate:.2%}."
    assert [c["name"] for c in out["tool_calls"]] == ["run_sql"]
    assert out["usage"] == {"input_tokens": 200, "output_tokens": 20}


def test_one_sql_repair_then_success(public_db):
    rate = rate_2016(public_db)
    model = ScriptedModel(
        call("run_sql", query="SELECT rate FROM segments"),
        call("run_sql", 1, query="SELECT default_rate FROM segments WHERE dimension = 'all' AND vintage = 2016"),
        AIMessage(f"{rate:.2%}"),
    )
    out = agent_graph.answer("2016 default rate?", model=model)
    assert not out["refused"]
    assert out["tool_calls"][0]["error"] and out["tool_calls"][1]["error"] is None
    assert "ERROR" in model.seen[1][-1].content  # the model saw the error


def test_second_sql_error_stops():
    model = ScriptedModel(
        call("run_sql", query="SELECT nope FROM segments"),
        call("run_sql", 1, query="DROP TABLE segments"),
        AIMessage("should not be reached"),
    )
    out = agent_graph.answer("Drop the table", model=model)
    assert out["refused"]
    assert len(model.script) == 1
    assert "rejected by guard" in out["answer"]


def test_decline_tool_refuses():
    model = ScriptedModel(call("decline", reason="There is no loan-level data here."))
    out = agent_graph.answer("Show me individual loans", model=model)
    assert out == {"answer": "There is no loan-level data here.", "refused": True,
                   "tool_calls": [{"name": "decline", "args": {"reason": "There is no loan-level data here."},
                                   "result": None, "error": None}],
                   "usage": {"input_tokens": 100, "output_tokens": 10}}


def test_invalid_loan_is_not_repaired():
    model = ScriptedModel(call("score_loan", **{**LOAN, "credit_score": 900}), AIMessage("unreached"))
    out = agent_graph.answer("Score credit score 900 ...", model=model)
    assert out["refused"]
    assert "outside the training range" in out["answer"]
    assert len(model.script) == 1


def test_loan_args_drop_empty_fields():
    model = ScriptedModel(call("score_loan", **{**LOAN, "orig_dti": None}), AIMessage("unreached"))
    out = agent_graph.answer("Score this loan without DTI", model=model)
    assert out["tool_calls"][0]["args"] == {"features": {k: v for k, v in LOAN.items() if k != "orig_dti"}}
    assert "Missing required fields" in out["answer"]


def test_ungrounded_number_gets_one_rewrite(public_db):
    rate = rate_2016(public_db)
    model = ScriptedModel(
        call("run_sql", query="SELECT default_rate FROM segments WHERE dimension = 'all' AND vintage = 2016"),
        AIMessage(f"{rate:.2%}, up 37.5% from 2015."),
        AIMessage(f"The 2016 default rate was {rate:.2%}."),
    )
    out = agent_graph.answer("2016 default rate?", model=model)
    assert not out["refused"]
    assert "37.5" in model.seen[2][-1].content  # the rewrite request names the number
    assert out["answer"] == f"The 2016 default rate was {rate:.2%}."


def test_still_ungrounded_after_rewrite_is_withheld():
    model = ScriptedModel(AIMessage("About 12.3%."), AIMessage("Roughly 12.3%."))
    out = agent_graph.answer("Default rate?", model=model)
    assert out["refused"]
    assert "12.3" not in out["answer"]


def test_step_limit(monkeypatch):
    monkeypatch.setattr(agent_graph, "MAX_LLM_CALLS", 2)
    q = "SELECT loans FROM segments WHERE dimension = 'all' AND vintage = 2016"
    model = ScriptedModel(call("run_sql", query=q), call("run_sql", 1, query=q), call("run_sql", 2, query=q))
    out = agent_graph.answer("loop", model=model)
    assert out["refused"]
    assert "step limit" in out["answer"]


def test_output_passes_harness_grading(public_db):
    rate = rate_2016(public_db)
    case = {"id": "sql_01", "category": "sql", "question": "What was the 2016 default rate?",
            "expect": {"tools": ["run_sql"], "reference_sql":
                       "SELECT default_rate FROM segments WHERE dimension = 'all' AND vintage = 2016"}}
    model = ScriptedModel(call("run_sql", query=case["expect"]["reference_sql"]),
                          AIMessage(f"It was {rate:.2%}."))
    out = agent_graph.answer(case["question"], model=model)
    g = grade(case, out, {"columns": ["default_rate"], "rows": [[rate]]})
    assert g["passed"], g


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        agent_graph.make_model()
