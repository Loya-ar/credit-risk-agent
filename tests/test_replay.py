"""CI regression gate: replay recorded Gemini responses through the real graph.

The first part checks record -> save -> load -> replay round-trips (always runs).
The second part replays every file in evals/recordings/ (made by
`run_eval --record` on a live run): a case that passed when recorded must still
pass, and the graph must ask the model for exactly the recorded turns.
It skips when there are no recordings or no public tables.
"""
import pytest

from agent import graph as agent_graph
from agent import tools as agent_tools
from agent.public_tables import ensure_public_db
from evals.harness import grade, load_cases, reference
from evals.replay import (
    RecordingModel, ReplayExhausted, ReplayModel, load_recordings, save_recording,
)
from risk.predict import CALIBRATED_PATH, RAW_PATH

SQL = "SELECT default_rate FROM segments WHERE dimension = 'all' AND vintage = 2016"


def turn(content="", tool_calls=(), tokens=(100, 10)):
    return {"content": content, "tool_calls": list(tool_calls),
            "usage_metadata": {"input_tokens": tokens[0], "output_tokens": tokens[1],
                               "total_tokens": sum(tokens)}}


@pytest.fixture
def public(monkeypatch, public_db):
    monkeypatch.delenv("AGENT_DB", raising=False)
    monkeypatch.setattr(agent_tools, "ensure_public_db", lambda: public_db)
    return public_db


# ---------- round trip (always runs) ----------

def test_record_save_load_replay(public, tmp_path):
    rate = agent_tools.run_sql(SQL, public)["rows"][0][0]
    live = ReplayModel([
        turn(tool_calls=[{"name": "run_sql", "args": {"query": SQL}, "id": "a"}]),
        turn(f"The 2016 default rate was {rate:.2%}."),
    ])
    rec = RecordingModel(live)
    first = agent_graph.answer("2016 default rate?", model=rec)
    assert len(rec.responses) == 2

    case = {"id": "sql_x", "question": "2016 default rate?"}
    save_recording(case, "fake-model", rec.responses, True, tmp_path)
    [saved] = load_recordings(tmp_path)
    assert saved["case_id"] == "sql_x" and saved["model"] == "fake-model"

    replay = ReplayModel(saved["responses"])
    second = agent_graph.answer(saved["question"], model=replay)
    assert replay.used == 2
    assert second == first


def test_replay_detects_a_changed_path(public):
    """If the graph wants one more model turn than recorded, replay fails loudly."""
    replay = ReplayModel([turn(tool_calls=[{"name": "run_sql", "args": {"query": SQL}, "id": "a"}])])
    with pytest.raises(ReplayExhausted):
        agent_graph.answer("2016 default rate?", model=replay)


def test_no_recordings_dir(tmp_path):
    assert load_recordings(tmp_path / "missing") == []


# ---------- recorded live runs (runs once recordings are committed) ----------

RECORDINGS = load_recordings()
CASES = {c["id"]: c for c in load_cases()}
MODELS_PRESENT = CALIBRATED_PATH.exists() and RAW_PATH.exists()


@pytest.mark.skipif(not RECORDINGS, reason="no recordings yet: run run_eval --record on a live key")
@pytest.mark.parametrize("rec", RECORDINGS, ids=lambda r: r["case_id"])
def test_recorded_case_still_passes(rec, monkeypatch):
    case = CASES.get(rec["case_id"])
    if case is None or case["question"] != rec["question"]:
        pytest.fail(f"{rec['case_id']}: case changed or removed since recording; re-record it")
    if case["category"] in {"score", "explain"} and not MODELS_PRESENT:
        pytest.skip("model files not present")
    try:
        db_path = ensure_public_db()
    except FileNotFoundError:
        pytest.skip("public tables not built")
    monkeypatch.delenv("AGENT_DB", raising=False)

    replay = ReplayModel(rec["responses"])
    result = agent_graph.answer(case["question"], model=replay)
    assert replay.used == len(rec["responses"]), "graph used fewer model turns than recorded"

    g = grade(case, result, reference(case, db_path))
    if rec["passed"]:
        assert g["passed"], f"regression: {g}"
    assert not g["leak"]
