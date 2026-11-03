"""Shared fixtures: a synthetic public database, so agent and eval tests run in CI
without the Freddie Mac data."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.public_tables import build_public_db, decile_table, segment_table

REPORTS = Path(__file__).resolve().parents[1] / "reports"


@pytest.fixture(scope="session")
def threshold() -> float:
    return 0.05


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def scores(loans) -> pd.DataFrame:
    test = loans[loans["vintage"] == 2016]
    p = np.clip(0.02 + (700 - test["credit_score"]) / 4000, 0.001, 0.2)
    return pd.DataFrame({"loan_seq_num": test["loan_seq_num"], "probability": p.to_numpy()})


@pytest.fixture(scope="session")
def segments(loans, scores, threshold) -> pd.DataFrame:
    return segment_table(loans, scores, threshold)


@pytest.fixture
def public_dir(tmp_path, segments, loans, scores):
    """Synthetic segments/deciles; model_metrics and thresholds are the real (public) reports."""
    d = tmp_path / "public_data"
    d.mkdir()
    segments.to_csv(d / "segments.csv", index=False)
    decile_table(loans, scores).to_csv(d / "deciles.csv", index=False)
    shutil.copyfile(REPORTS / "s7_test_metrics.csv", d / "model_metrics.csv")
    shutil.copyfile(REPORTS / "s8_thresholds.csv", d / "thresholds.csv")
    return d


@pytest.fixture
def public_db(public_dir, tmp_path):
    return build_public_db(public_dir, tmp_path / "agent_public.duckdb")
