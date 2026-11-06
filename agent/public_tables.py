"""A1e: aggregate-only tables for the public demo.

Freddie Mac's terms allow publishing noncommercial research results and derived
products only if they "cannot be used to derive or recreate any part of" the
dataset. So the public agent never sees loan rows. It queries these tables:

  segments       default rate by one dimension at a time (credit score band, LTV
                 band, state, ...) per vintage; 2016 rows add model predictions
  deciles        2016 test loans grouped by model score decile
  model_metrics  test metrics per model (copied from reports/s7_test_metrics.csv)
  thresholds     cost-based thresholds per assumed ratio (reports/s8_thresholds.csv)

Privacy rules, enforced here and tested in tests/test_public_tables.py:
  - one-way breakdowns only (never dimension x dimension), so cells stay large
  - groups under MIN_CELL loans are pooled into "Other"; if Other is still too
    small, it is dropped
  - no loan IDs, no loan-level values

The CSVs in public_data/ are the only thing published; data/agent_public.duckdb is
rebuilt from them.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# PUBLIC_DATA_DIR points the app and agent at another set of CSVs (e.g. a synthetic demo).
if os.environ.get("PUBLIC_DATA_DIR"):
    PUBLIC_DIR = Path(os.environ["PUBLIC_DATA_DIR"])
    PUBLIC_DB_PATH = PUBLIC_DIR / "agent_public.duckdb"
else:
    PUBLIC_DIR = ROOT / "public_data"
    PUBLIC_DB_PATH = ROOT / "data" / "agent_public.duckdb"
PUBLIC_TABLES = ("segments", "deciles", "model_metrics", "thresholds")

MIN_CELL = 50
OTHER = "Other (small groups)"
MISSING = "missing"

# (column, bin edges, labels). Bins are right-closed except credit score.
NUMERIC_BANDS = {
    "credit_score_band": (
        "credit_score", [-np.inf, 620, 660, 700, 740, 780, np.inf],
        ["<620", "620-659", "660-699", "700-739", "740-779", "780+"], False,
    ),
    "ltv_band": (
        "orig_ltv", [-np.inf, 60, 80, 90, 95, np.inf],
        ["<=60", "60-80", "80-90", "90-95", ">95"], True,
    ),
    "dti_band": (
        "orig_dti", [-np.inf, 20, 30, 36, 43, 50, np.inf],
        ["<=20", "20-30", "30-36", "36-43", "43-50", ">50"], True,
    ),
    "interest_rate_band": (
        "orig_interest_rate", [-np.inf, 3.5, 4.0, 4.5, 5.0, np.inf],
        ["<=3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0", ">5.0"], True,
    ),
    "loan_amount_band": (
        "orig_upb", [-np.inf, 100_000, 200_000, 300_000, 417_000, np.inf],
        ["<=100k", "100k-200k", "200k-300k", "300k-417k", ">417k"], True,
    ),
}
CATEGORICAL_DIMENSIONS = {
    "property_state": "property_state",
    "loan_purpose": "loan_purpose",
    "occupancy_status": "occupancy_status",
    "channel": "channel",
    "property_type": "property_type",
    "first_time_homebuyer": "first_time_homebuyer_flag",
    "num_borrowers": "num_borrowers",
    "harp": "harp_indicator",
}
DIMENSIONS = ("all", *NUMERIC_BANDS, *CATEGORICAL_DIMENSIONS)


def dimension_buckets(loans: pd.DataFrame, dimension: str) -> pd.Series:
    """The bucket label of every loan for one dimension (missing values get MISSING)."""
    if dimension == "all":
        return pd.Series("all", index=loans.index)
    if dimension in NUMERIC_BANDS:
        column, edges, labels, right = NUMERIC_BANDS[dimension]
        bands = pd.cut(loans[column], bins=edges, labels=labels, right=right)
        return bands.astype(object).where(bands.notna(), MISSING)
    values = loans[CATEGORICAL_DIMENSIONS[dimension]]
    if dimension == "num_borrowers":
        values = values.map(lambda v: v if pd.isna(v) else ("1" if v == 1 else "2+"))
    values = values.astype(object)
    return values.where(values.notna() & (values.astype(str) != ""), MISSING).astype(str)


def pool_small_groups(counts: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Pool buckets under min_cell loans into OTHER.

    If OTHER is still under min_cell, the next-smallest buckets join it until it isn't,
    so no small group can be recovered by subtracting visible rows from the total.
    counts has one row per bucket with additive columns (loans, defaults, sums).
    """
    if not (counts["loans"] < min_cell).any():
        return counts
    if counts["loans"].sum() < min_cell:
        return counts.iloc[0:0]
    ordered = counts.sort_values("loans", kind="mergesort").reset_index(drop=True)
    pooled_loans, n_pooled = 0, 0
    for loans in ordered["loans"]:
        if loans >= min_cell and pooled_loans >= min_cell:
            break
        pooled_loans += loans
        n_pooled += 1
    other = ordered.iloc[:n_pooled].drop(columns="bucket").sum().to_frame().T
    other.insert(0, "bucket", OTHER)
    return pd.concat([ordered.iloc[n_pooled:], other], ignore_index=True)


def segment_table(
    loans: pd.DataFrame,
    test_scores: pd.DataFrame,
    threshold: float,
    min_cell: int = MIN_CELL,
) -> pd.DataFrame:
    """Default rate by each dimension and vintage; test-vintage rows add model columns.

    loans: loan rows with vintage, default_24m and the feature columns.
    test_scores: loan_seq_num, probability for the test vintage.
    """
    df = loans.merge(test_scores[["loan_seq_num", "probability"]], on="loan_seq_num", how="left")
    df["scored"] = df["probability"].notna().astype(int)
    df["prob_sum"] = df["probability"].fillna(0.0)
    df["declined"] = (df["probability"] >= threshold).astype(int)

    rows = []
    for dimension in DIMENSIONS:
        df["bucket"] = dimension_buckets(df, dimension)
        for vintage, group in df.groupby("vintage"):
            counts = group.groupby("bucket", as_index=False).agg(
                loans=("default_24m", "size"),
                defaults=("default_24m", "sum"),
                scored=("scored", "sum"),
                prob_sum=("prob_sum", "sum"),
                declined=("declined", "sum"),
            )
            counts = pool_small_groups(counts, min_cell)
            counts.insert(0, "vintage", int(vintage))
            counts.insert(0, "dimension", dimension)
            rows.append(counts)

    out = pd.concat(rows, ignore_index=True)
    out["loans"] = out["loans"].astype(int)
    out["defaults"] = out["defaults"].astype(int)
    out["default_rate"] = (out["defaults"] / out["loans"]).round(6)
    has_scores = out["scored"] == out["loans"]
    out["mean_predicted"] = (out["prob_sum"] / out["loans"]).round(6).where(has_scores)
    out["declined_share"] = (out["declined"] / out["loans"]).round(6).where(has_scores)
    return out[[
        "dimension", "bucket", "vintage", "loans", "defaults", "default_rate",
        "mean_predicted", "declined_share",
    ]]


def decile_table(loans: pd.DataFrame, test_scores: pd.DataFrame) -> pd.DataFrame:
    """Test loans by score decile (10 = riskiest), with lift over the overall rate."""
    df = test_scores[["loan_seq_num", "probability"]].merge(
        loans[["loan_seq_num", "default_24m"]], on="loan_seq_num"
    ).sort_values(["probability", "loan_seq_num"], kind="mergesort")
    df["decile"] = np.arange(len(df)) * 10 // len(df) + 1

    out = df.groupby("decile", as_index=False).agg(
        loans=("default_24m", "size"),
        defaults=("default_24m", "sum"),
        mean_predicted=("probability", "mean"),
        min_predicted=("probability", "min"),
        max_predicted=("probability", "max"),
    )
    overall = out["defaults"].sum() / out["loans"].sum()
    out["default_rate"] = out["defaults"] / out["loans"]
    out["lift"] = out["default_rate"] / overall
    out["share_of_defaults"] = out["defaults"] / out["defaults"].sum()
    out = out.sort_values("decile", ascending=False).reset_index(drop=True)
    for col in ["mean_predicted", "min_predicted", "max_predicted", "default_rate",
                "lift", "share_of_defaults"]:
        out[col] = out[col].round(6)
    return out[[
        "decile", "loans", "defaults", "default_rate", "lift", "share_of_defaults",
        "mean_predicted", "min_predicted", "max_predicted",
    ]]


def build_public_db(public_dir: Path = PUBLIC_DIR, db_path: Path = PUBLIC_DB_PATH) -> Path:
    """(Re)build the public DuckDB file from the published CSVs."""
    missing = [t for t in PUBLIC_TABLES if not (Path(public_dir) / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing public tables {missing} in {public_dir}. "
            "Run python -m scripts.build_public_tables where the loan data is available."
        )
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for table in PUBLIC_TABLES:
            csv = (Path(public_dir) / f"{table}.csv").as_posix()
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto('{csv}', header=true)")
    finally:
        con.close()
    return db_path


def ensure_public_db(public_dir: Path = PUBLIC_DIR, db_path: Path = PUBLIC_DB_PATH) -> Path:
    """Build the public DB if it is missing or older than any of its CSVs."""
    db_path = Path(db_path)
    csvs = [Path(public_dir) / f"{t}.csv" for t in PUBLIC_TABLES]
    if db_path.exists() and all(
        c.exists() and c.stat().st_mtime <= db_path.stat().st_mtime for c in csvs
    ):
        return db_path
    return build_public_db(public_dir, db_path)
