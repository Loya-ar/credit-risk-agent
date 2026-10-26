"""A1b: build the agent's own database and the scoring defaults file.

Outputs:
  data/agent.duckdb            only two tables: loans, test_scores (gitignored)
  reports/feature_defaults.json  training medians/modes + allowed codes (aggregates only)

Run from the repo root:  python -m scripts.build_agent_db
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from risk.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from risk.split import DB_PATH, TEST_VINTAGES, TRAIN_VINTAGES

ROOT = Path(__file__).resolve().parents[1]
AGENT_DB_PATH = ROOT / "data" / "agent.duckdb"
MODEL_PATH = ROOT / "models" / "lgbm_calibrated.joblib"
THRESHOLD_PATH = ROOT / "reports" / "s8_threshold.json"
DEFAULTS_PATH = ROOT / "reports" / "feature_defaults.json"


def load_loans() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.sql("SELECT * FROM loans").df()
    finally:
        con.close()


def feature_defaults(train: pd.DataFrame) -> dict:
    """Typical values and valid ranges/codes, from training vintages only."""
    numeric = {}
    for f in NUMERIC_FEATURES:
        col = train[f].dropna()
        numeric[f] = {
            "median": float(col.median()),
            "min": float(col.min()),
            "max": float(col.max()),
        }

    categorical = {}
    for f in CATEGORICAL_FEATURES:
        # dropna=False: blank counts as a value, so Y/blank flags (e.g. harp_indicator)
        # get blank as their typical value instead of Y.
        counts = train[f].value_counts(dropna=False)
        mode = counts.index[0]
        categorical[f] = {
            "mode": None if pd.isna(mode) else str(mode),
            "allowed": sorted(str(v) for v in train[f].dropna().unique()),
            "missing_share": round(float(train[f].isna().mean()), 4),
        }

    return {
        "source": f"training vintages {list(TRAIN_VINTAGES)}",
        "numeric": numeric,
        "categorical": categorical,
    }


def score_test(loans: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Calibrated probability and decision for every test-vintage loan."""
    model = joblib.load(MODEL_PATH)
    test = loans[loans["vintage"].isin(TEST_VINTAGES)]
    p = model.predict_proba(test[FEATURES])[:, 1]
    return pd.DataFrame({
        "loan_seq_num": test["loan_seq_num"].to_numpy(),
        "probability": p,
        "declined_at_threshold": p >= threshold,
    })


def build_db(loans: pd.DataFrame, scores: pd.DataFrame) -> None:
    AGENT_DB_PATH.unlink(missing_ok=True)  # start fresh so only our two tables exist
    con = duckdb.connect(str(AGENT_DB_PATH))
    try:
        con.register("loans_df", loans)
        con.register("scores_df", scores)
        con.execute("CREATE TABLE loans AS SELECT * FROM loans_df")
        con.execute("""
            CREATE TABLE test_scores AS
            SELECT
                loan_seq_num,
                probability,
                NTILE(10) OVER (ORDER BY probability, loan_seq_num) AS decile,  -- 10 = riskiest
                declined_at_threshold
            FROM scores_df
        """)
        con.unregister("loans_df")
        con.unregister("scores_df")
    finally:
        con.close()


def check_db(n_loans: int, n_test: int) -> None:
    con = duckdb.connect(str(AGENT_DB_PATH), read_only=True)
    try:
        tables = sorted(r[0] for r in con.sql("SHOW TABLES").fetchall())
        assert tables == ["loans", "test_scores"], f"unexpected tables: {tables}"
        assert con.sql("SELECT COUNT(*) FROM loans").fetchone()[0] == n_loans
        assert con.sql("SELECT COUNT(*) FROM test_scores").fetchone()[0] == n_test

        print("Tables:", tables)
        print("loans rows:", n_loans, "| test_scores rows:", n_test)

        n, defaults, mean_p, declined = con.sql("""
            SELECT COUNT(*), SUM(l.default_24m), AVG(t.probability),
                   SUM(CASE WHEN t.declined_at_threshold THEN 1 ELSE 0 END)
            FROM test_scores t JOIN loans l USING (loan_seq_num)
        """).fetchone()
        print(f"Test defaults: {defaults} of {n} ({defaults / n:.2%})")
        print(f"Mean calibrated probability: {mean_p:.2%}")
        print(f"Declined at threshold: {declined} ({declined / n:.2%})")

        print("\nDecile table (10 = riskiest):")
        rows = con.sql("""
            SELECT t.decile, COUNT(*) AS n, SUM(l.default_24m) AS defaults,
                   AVG(l.default_24m) AS rate
            FROM test_scores t JOIN loans l USING (loan_seq_num)
            GROUP BY t.decile ORDER BY t.decile DESC
        """).fetchall()
        for decile, n_d, d, rate in rows:
            print(f"  decile {decile:>2}: n={n_d:>5}  defaults={d:>3}  rate={rate:.2%}  "
                  f"share of defaults={d / defaults:.1%}")
    finally:
        con.close()


def main() -> None:
    loans = load_loans()
    train = loans[loans["vintage"].isin(TRAIN_VINTAGES)]
    threshold = json.loads(THRESHOLD_PATH.read_text())["threshold"]

    defaults = feature_defaults(train)
    DEFAULTS_PATH.write_text(json.dumps(defaults, indent=2))
    print(f"Wrote {DEFAULTS_PATH.relative_to(ROOT)}")

    scores = score_test(loans, threshold)
    build_db(loans, scores)
    print(f"Wrote {AGENT_DB_PATH.relative_to(ROOT)}  (threshold {threshold:.4f}, ASSUMED cost ratio 20)\n")

    check_db(n_loans=len(loans), n_test=len(scores))


if __name__ == "__main__":
    main()