"""The public demo tables: aggregate-only, no small groups, no loan rows.

Runs on synthetic loans, so it works in CI without the Freddie Mac data.
"""
import os

import duckdb
import pandas as pd
import pytest

from agent import tools
from agent.public_tables import (
    DIMENSIONS, MIN_CELL, OTHER, PUBLIC_TABLES, build_public_db, decile_table, dimension_buckets,
    ensure_public_db, pool_small_groups,
)
from agent.tools import ToolError, run_sql

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


def test_predictions_only_on_test_vintage(segments, scores, threshold):
    test = segments[segments["vintage"] == 2016]
    assert test["mean_predicted"].notna().all()
    assert segments[segments["vintage"] != 2016]["mean_predicted"].isna().all()
    overall = test[test["dimension"] == "all"].iloc[0]
    assert overall["mean_predicted"] == pytest.approx(scores["probability"].mean(), abs=1e-6)
    assert overall["declined_share"] == pytest.approx(
        (scores["probability"] >= threshold).mean(), abs=1e-6)


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
