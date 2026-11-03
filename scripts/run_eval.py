"""Run the evaluation set against an agent and write a report.

  python -m scripts.run_eval --agent agent.graph:answer
  python -m scripts.run_eval --agent oracle          # harness self-test, should be 100%
  python -m scripts.run_eval --agent agent.graph:answer --category guard --price-in 0.3 --price-out 2.5
  python -m scripts.run_eval --agent agent.graph:answer --record --min-pass-rate 0.8

Writes reports/eval/<timestamp>.json (every case, answer and grade) and prints the summary.
Prices are USD per million tokens. --record saves each case's model responses to
evals/recordings/ for the CI replay tests. --min-pass-rate exits 1 when the pass
rate is lower or any loan-level column leaked (used by the nightly workflow).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime
from pathlib import Path

from agent.public_tables import ensure_public_db
from evals.harness import load_cases, oracle_agent, run
from evals.replay import RecordingModel, save_recording

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "eval"


def load_agent(spec: str, cases, db_path):
    if spec == "oracle":
        return oracle_agent(cases, db_path)
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "answer")


def recording_agent(cases):
    """The Gemini agent, keeping every model response per case."""
    from agent import graph
    base = graph.make_model()
    case_ids = {c["question"]: c["id"] for c in cases}
    sessions: dict[str, RecordingModel] = {}

    def agent(question: str) -> dict:
        sessions[case_ids[question]] = rec = RecordingModel(base)
        return graph.answer(question, model=rec)

    return agent, sessions


def step_summary(summary: dict, failed: list[str]) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in summary.items() if k != "by_category")
    cats = "\n".join(f"| {k} | {v} |" for k, v in summary["by_category"].items())
    return (f"## Agent evaluation\n\n| Metric | Value |\n|---|---|\n{rows}\n\n"
            f"| Category | Pass rate |\n|---|---|\n{cats}\n\nFailed: {', '.join(failed) or '-'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="module:function, or 'oracle'")
    parser.add_argument("--category", action="append", help="only these categories (repeatable)")
    parser.add_argument("--price-in", type=float, default=0.0)
    parser.add_argument("--price-out", type=float, default=0.0)
    parser.add_argument("--record", action="store_true", help="save model responses for CI replay")
    parser.add_argument("--min-pass-rate", type=float, help="exit 1 below this pass rate")
    args = parser.parse_args()

    cases = load_cases()
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    try:
        db_path = ensure_public_db()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    sessions = None
    if args.record:
        if args.agent != "agent.graph:answer":
            raise SystemExit("--record only works with --agent agent.graph:answer")
        agent, sessions = recording_agent(cases)
    else:
        agent = load_agent(args.agent, cases, db_path)

    report = run(agent, cases, db_path, price_in=args.price_in, price_out=args.price_out)
    report["agent"] = args.agent

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(report["summary"], indent=2))
    failed = [g["id"] for g in report["grades"] if g["passed"] is False]
    skipped = [g["id"] for g in report["grades"] if g["passed"] is None]
    print(f"\nFailed ({len(failed)}): {', '.join(failed) or '-'}")
    print(f"Skipped ({len(skipped)}): {', '.join(skipped) or '-'}")
    print(f"Report: {out.relative_to(ROOT)}")

    if sessions is not None:
        from agent.graph import DEFAULT_MODEL
        model_name = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
        grades = {g["id"]: g for g in report["grades"]}
        for case in cases:
            if case["id"] in sessions:
                save_recording(case, model_name, sessions[case["id"]].responses, grades[case["id"]]["passed"])
        print(f"Recorded {len(sessions)} cases to evals/recordings/")

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(step_summary(report["summary"], failed))

    s = report["summary"]
    if args.min_pass_rate is not None:
        if s["leaks"]:
            raise SystemExit(f"FAIL: {s['leaks']} case(s) leaked loan-level columns")
        if (s["pass_rate"] or 0) < args.min_pass_rate:
            raise SystemExit(f"FAIL: pass rate {s['pass_rate']} < {args.min_pass_rate}")


if __name__ == "__main__":
    main()
