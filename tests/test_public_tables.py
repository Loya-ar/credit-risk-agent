"""The public demo tables: aggregate-only, no small groups, no loan rows.

Runs on synthetic loans, so it works in CI without the Freddie Mac data.
"""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from agent import tools
from agent.public_tables import (
    DIMENSIONS, MIN_CELL, OTHER, PUBLIC_TABLES, build_public_db, decile_table, dimension_buckets,
    ensure_public_db, pool_small_groups, segment_table,
)
from agent.tools import ToolError, run_sql

THRESHOLD = 0.05


@pytest.fixture(scope="module")
def loans() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "loan_seq_num": [f"L{i:05d}" for i in range(n)],
        "vintage": rng.choice([2014, 2015, 2016], n),
        "credit_score": rng.integers(600, 820, n).astype(float),
        "orig_ltv": rng.integers(40, 98, n).astype(float),
        "orig_dti": rng.integers(10, 50, n).astype(float),
        "orig_interest_rate": rng.uniform(3.0, 5.5, n),
        "orig_upb": rng.integers(50_000, 600_000, n).astype(float),
        # WY is rare on purpose: it must never show up as its own group.
        "property_state": rng.choice(["CA", "TX", "FL", "WY"], n, p=[0.4, 0.33, 0.26, 0.01]),
        "loan_purpose": rng.choice(["P", "C", "N"], n),
        "occupancy_status": rng.choice(["P", "I", "S"], n, p=[0.9, 0.07, 0.03]),
        "channel": rng.choice(["R", "B", "C"], n),
        "property_type": rng.choice(["SF", "PU", "CO"], n),
        "first_time_homebuyer_flag": rng.choice(["Y", "N"], n),
        "num_borrowers": rng.choice([1.0, 2.0], n),
        "harp_indicator": rng.choice(["Y", None], n, p=[0.05, 0.95]),
        "default_24m": rng.binomial(1, 0.05, n),
    })
    df.loc[df["harp_indicator"] == "Y", "orig_dti"] = np.nan
    return df


@pytest.fixture(scope="module")
def scores(loans) -> pd.DataFrame:
    test = loans[loans["vintage"] == 2016]
    p = np.clip(0.02 + (700 - test["credit_score"]) / 4000, 0.001, 0.2)
    return pd.DataFrame({"loan_seq_num": test["loan_seq_num"], "probability": p.to_numpy()})


@pytest.fixture(scope="module")
def segments(loans, scores) -> pd.DataFrame:
    return segment_table(loans, scores, THRESHOLD)


@pytest.fixture
def public_dir(tmp_path, segments, loans, scores):
    d = tmp_path / "public_data"
    d.mkdir()
    segments.to_csv(d / "segments.csv", index=False)
    decile_table(loans, scores).to_csv(d / "deciles.csv", index=False)
    pd.DataFrame({"model": ["lightgbm"], "roc_auc": [0.81]}).to_csv(d / "model_metrics.csv", index=False)
    pd.DataFrame({"cost_ratio_ASSUMED": [20], "threshold_from_2015": [0.049]}).to_csv(
        d / "thresholds.csv", index=False)
    return d


@pytest.fixture
def public_db(public_dir, tmp_path):
    return build_public_db(public_dir, tmp_path / "agent_public.duckdb")


# ---------- privacy rules ----------

def test_no_group_below_min_cell(segments):
    assert segments["loans"].min() >= MIN_CELL


def test_no_loan_level_columns(segments, loans, scores):
    for table in (segments, decile_table(loans, scores)):
        assert "loan_seq_num" not in table.columns
        assert len(table) < len(loans) / 10


def test_rare_state_is_pooled_not_shown(segments):
    states = segments[segments["dimension"] == "property_state"]
    assert "WY" not in set(states["bucket"])
    assert OTHER in set(states["bucket"])


def test_pooling_keeps_every_loan(segments, loans):
    """Nothing is dropped, so 'all' minus visible rows can't reveal a hidden group."""
    per_vintage = loans.groupby("vintage").size()
    totals = segments.groupby(["dimension", "vintage"])["loans"].sum()
    for (dimension, vintage), n in totals.items():
        assert n == per_vintage[vintage], (dimension, vintage)


def test_single_small_group_pools_with_next_smallest():
    counts = pd.DataFrame({"bucket": ["a", "b", "c"], "loans": [500, 120, 10], "defaults": [5, 2, 1]})
    pooled = pool_small_groups(counts, min_cell=50)
    assert set(pooled["bucket"]) == {"a", OTHER}
    assert pooled.set_index("bucket").loc[OTHER, "loans"] == 130
    assert pooled["loans"].sum() == 630


def test_all_small_returns_nothing():
    counts = pd.DataFrame({"bucket": ["a", "b"], "loans": [10, 20], "defaults": [0, 1]})
    assert pool_small_groups(counts, min_cell=50).empty


# ---------- content ----------

def test_one_way_dimensions_only(segments):
    assert set(segments["dimension"]) <= set(DIMENSIONS)


def test_all_row_matches_raw_rate(segments, loans):
    row = segments[(segments["dimension"] == "all") & (segments["vintage"] == 2016)].iloc[0]
    raw = loans[loans["vintage"] == 2016]["default_24m"]
    assert row["loans"] == len(raw)
    assert row["default_rate"] == pytest.approx(raw.mean(), abs=1e-6)


def test_predictions_only_on_test_vintage(segments, scores):
    test = segments[segments["vintage"] == 2016]
    assert test["mean_predicted"].notna().all()
    assert segments[segments["vintage"] != 2016]["mean_predicted"].isna().all()
    overall = test[test["dimension"] == "all"].iloc[0]
    assert overall["mean_predicted"] == pytest.approx(scores["probability"].mean(), abs=1e-6)
    assert overall["declined_share"] == pytest.approx(
        (scores["probability"] >= THRESHOLD).mean(), abs=1e-6)


def test_missing_values_get_their_own_bucket(loans):
    buckets = dimension_buckets(loans, "dti_band")
    assert (buckets[loans["orig_dti"].isna()] == "missing").all()
    harp = dimension_buckets(loans, "harp")
    assert set(harp) == {"Y", "missing"}


def test_deciles(loans, scores):
    d = decile_table(loans, scores)
    assert list(d["decile"]) == list(range(10, 0, -1))
    assert d["loans"].sum() == len(scores)
    assert d["share_of_defaults"].sum() == pytest.approx(1.0, abs=1e-5)
    assert d.iloc[0]["mean_predicted"] > d.iloc[-1]["mean_predicted"]


# ---------- public DB + run_sql ----------

def test_public_db_has_only_public_tables(public_db):
    con = duckdb.connect(str(public_db), read_only=True)
    try:
        tables = sorted(r[0] for r in con.execute("SHOW TABLES").fetchall())
    finally:
        con.close()
    assert tables == sorted(PUBLIC_TABLES)


def test_catalog_queries_are_blocked(public_db):
    with pytest.raises(ToolError, match="plain table names"):
        run_sql("SELECT table_name FROM information_schema.tables", public_db)


def test_public_db_answers_questions(public_db):
    result = run_sql(
        "SELECT bucket, default_rate FROM segments "
        "WHERE dimension = 'credit_score_band' AND vintage = 2016 ORDER BY bucket",
        public_db,
    )
    assert result["row_count"] >= 3


def test_public_db_rejects_loan_tables(public_db):
    with pytest.raises(ToolError, match="Unknown table 'loans'.*segments"):
        run_sql("SELECT * FROM loans", public_db)


def test_missing_public_csvs(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_public_tables"):
        build_public_db(tmp_path, tmp_path / "x.duckdb")


def test_ensure_rebuilds_when_csv_changes(public_dir, public_db):
    pd.DataFrame({"model": ["new"], "roc_auc": [0.9]}).to_csv(public_dir / "model_metrics.csv", index=False)
    later = public_db.stat().st_mtime + 10
    os.utime(public_dir / "model_metrics.csv", (later, later))
    ensure_public_db(public_dir, public_db)
    assert run_sql("SELECT model FROM model_metrics", public_db)["rows"] == [["new"]]


# ---------- choosing the database ----------

def test_default_is_public(monkeypatch, public_db):
    monkeypatch.delenv("AGENT_DB", raising=False)
    monkeypatch.setattr(tools, "ensure_public_db", lambda: public_db)
    assert tools.default_db_path() == public_db


def test_local_mode_needs_local_db(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DB", "local")
    monkeypatch.setattr(tools, "AGENT_DB_PATH", tmp_path / "missing.duckdb")
    with pytest.raises(ToolError, match="build_agent_db"):
        tools.default_db_path()


def test_unknown_mode(monkeypatch):
    monkeypatch.setenv("AGENT_DB", "prod")
    with pytest.raises(ToolError, match="'public' or 'local'"):
        tools.default_db_path()


def test_public_mode_without_csvs(monkeypatch):
    def missing():
        raise FileNotFoundError("Missing public tables")
    monkeypatch.delenv("AGENT_DB", raising=False)
    monkeypatch.setattr(tools, "ensure_public_db", missing)
    with pytest.raises(ToolError, match="not available"):
        tools.default_db_path()
