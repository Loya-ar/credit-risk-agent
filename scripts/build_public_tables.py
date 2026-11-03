"""A1e: write the aggregate-only tables the public demo is allowed to see.

Inputs (local only, gitignored):
  data/credit.duckdb                   loan rows with labels
  models/lgbm_calibrated.joblib        the calibrated model, to score the test vintage
Outputs (committed, safe to publish):
  public_data/segments.csv, deciles.csv, model_metrics.csv, thresholds.csv
  data/agent_public.duckdb             rebuilt from those CSVs

Run from the repo root:  python -m scripts.build_public_tables
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from agent.public_tables import (
    MIN_CELL, PUBLIC_DIR, build_public_db, decile_table, segment_table,
)
from scripts.build_agent_db import ROOT, THRESHOLD_PATH, load_loans, score_test

REPORTS = ROOT / "reports"


def main() -> None:
    loans = load_loans()
    threshold = json.loads(THRESHOLD_PATH.read_text())["threshold"]
    scores = score_test(loans, threshold)

    PUBLIC_DIR.mkdir(exist_ok=True)
    segments = segment_table(loans, scores, threshold)
    deciles = decile_table(loans, scores)
    segments.to_csv(PUBLIC_DIR / "segments.csv", index=False)
    deciles.to_csv(PUBLIC_DIR / "deciles.csv", index=False)
    shutil.copyfile(REPORTS / "s7_test_metrics.csv", PUBLIC_DIR / "model_metrics.csv")
    shutil.copyfile(REPORTS / "s8_thresholds.csv", PUBLIC_DIR / "thresholds.csv")

    db_path = build_public_db()
    print(f"Wrote {PUBLIC_DIR.relative_to(ROOT)}/ and {db_path.relative_to(ROOT)}")
    print(f"segments: {len(segments)} rows (min cell {MIN_CELL} loans) | deciles: {len(deciles)} rows")
    print(f"smallest segment: {segments['loans'].min()} loans")
    all_rows = segments[segments["dimension"] == "all"].set_index("vintage")
    print(pd.DataFrame({
        "loans": all_rows["loans"],
        "default_rate": all_rows["default_rate"].map("{:.2%}".format),
    }))


if __name__ == "__main__":
    main()
