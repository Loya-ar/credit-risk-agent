"""Run the evaluation set against an agent and write a report.

  python -m scripts.run_eval --agent agent.graph:answer
  python -m scripts.run_eval --agent oracle          # harness self-test, should be 100%
  python -m scripts.run_eval --agent agent.graph:answer --category guard --price-in 0.3 --price-out 2.5

Writes reports/eval/<timestamp>.json (every case, answer and grade) and prints the summary.
Prices are USD per million tokens.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

from agent.public_tables import ensure_public_db
from evals.harness import load_cases, oracle_agent, run

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "eval"


def load_agent(spec: str, cases, db_path):
    if spec == "oracle":
        return oracle_agent(cases, db_path)
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "answer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="module:function, or 'oracle'")
    parser.add_argument("--category", action="append", help="only these categories (repeatable)")
    parser.add_argument("--price-in", type=float, default=0.0)
    parser.add_argument("--price-out", type=float, default=0.0)
    args = parser.parse_args()

    cases = load_cases()
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    try:
        db_path = ensure_public_db()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
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


if __name__ == "__main__":
    main()
