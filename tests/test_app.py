"""The Streamlit app, run headless with AppTest on the synthetic public DB.

The agent is stubbed (no API key needed); scoring is checked for its
"model files missing" state because CI has no trained model.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent import graph as agent_graph
from agent import public_tables
from risk import predict

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, public_db, tmp_path):
    monkeypatch.setattr(public_tables, "ensure_public_db", lambda: public_db)
    monkeypatch.setattr(predict, "CALIBRATED_PATH", tmp_path / "missing.joblib")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(agent_graph, "answer", lambda q: {
        "answer": "The 2016 default rate was 5.00%.", "refused": False,
        "tool_calls": [{"name": "run_sql", "args": {"query": "SELECT 1"},
                        "result": {"columns": ["x"], "rows": [[1]]}, "error": None}],
        "usage": {"input_tokens": 10, "output_tokens": 5}})
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_renders_all_tabs(app):
    assert [t.label for t in app.tabs] == ["Ask the agent", "Score a loan", "Explore the data", "About"]
    assert any("Model files" in c.value for c in app.caption)


def test_scoring_explains_missing_models(app):
    assert any("models/" in i.value for i in app.info)


def test_explore_shows_segments_and_switches_dimension(app):
    breakdown = next(s for s in app.selectbox if s.label == "Breakdown")
    assert breakdown.value == "credit_score_band"
    breakdown.set_value("property_state").run()
    assert not app.exception
    assert len(app.dataframe) >= 2  # segment table + model metrics


def test_chat_answer_and_trace(app):
    app.chat_input[0].set_value("What was the 2016 default rate?").run()
    assert not app.exception
    assert any("5.00%" in m.value for m in app.markdown)
    assert any(c.value == "SELECT 1" for c in app.code)


def test_without_key_or_tables(monkeypatch, tmp_path):
    def missing():
        raise FileNotFoundError("no tables")
    monkeypatch.setattr(public_tables, "ensure_public_db", missing)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    texts = " ".join(i.value for i in at.info)
    assert "GOOGLE_API_KEY" in texts and "build_public_tables" in texts


def test_session_question_limit(monkeypatch, public_db, tmp_path):
    calls = []
    monkeypatch.setattr(public_tables, "ensure_public_db", lambda: public_db)
    monkeypatch.setattr(predict, "CALIBRATED_PATH", tmp_path / "missing.joblib")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SESSION_LIMIT", "1")
    monkeypatch.setattr(agent_graph, "answer", lambda q: calls.append(q) or {
        "answer": "ok", "refused": False, "tool_calls": [], "usage": {}})
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.chat_input[0].set_value("first question").run()
    at.chat_input[0].set_value("second question").run()
    assert not at.exception
    assert calls == ["first question"]  # the second never reached the agent
    assert any("used its 1 questions" in w.value for w in at.warning)
