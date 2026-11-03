"""Record live model responses once, replay them in CI without an API key.

A live run with  python -m scripts.run_eval --agent agent.graph:answer --record
saves every model response per case to evals/recordings/<case_id>.json. In CI,
tests/test_replay.py feeds those responses back through the real graph, tools
and grader. The model's side is fixed, so a failing replay means our code
(routing, tools, guard, grounding, grading) changed behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def message_to_dict(msg: AIMessage) -> dict:
    return {
        "content": msg.content,
        "tool_calls": [{"name": tc["name"], "args": tc["args"], "id": tc.get("id")} for tc in msg.tool_calls],
        "usage_metadata": dict(msg.usage_metadata or {}),
    }


def message_from_dict(d: dict) -> AIMessage:
    return AIMessage(content=d["content"], tool_calls=d["tool_calls"],
                     usage_metadata=d.get("usage_metadata") or None)


class RecordingModel:
    """Wraps a chat model and keeps every response it returns."""

    def __init__(self, model):
        self.model = model
        self.responses: list[dict] = []

    def bind_tools(self, tools):
        bound = self.model.bind_tools(tools)
        recorder = self

        class _Bound:
            def invoke(self, messages):
                msg = bound.invoke(messages)
                recorder.responses.append(message_to_dict(msg))
                return msg

        return _Bound()


class ReplayExhausted(RuntimeError):
    """The graph asked the model for more turns than were recorded: its path changed."""


class ReplayModel:
    """Returns recorded responses in order."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.used = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if self.used >= len(self.responses):
            raise ReplayExhausted(f"only {len(self.responses)} responses were recorded")
        msg = message_from_dict(self.responses[self.used])
        self.used += 1
        return msg


def save_recording(case: dict, model_name: str, responses: list[dict], passed, directory: Path = RECORDINGS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case['id']}.json"
    path.write_text(json.dumps({
        "case_id": case["id"], "question": case["question"], "model": model_name,
        "passed": passed, "responses": responses,
    }, indent=2, default=str), encoding="utf-8")
    return path


def load_recordings(directory: Path = RECORDINGS_DIR) -> list[dict]:
    if not directory.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]
