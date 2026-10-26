"""S5: time-based split by origination year (vintage).

Train on the earliest vintages, validate on the next, test on the latest.
No random shuffling across time.

Run from the repo root to see the split sizes:  python -m risk.split
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

TRAIN_VINTAGES = (2012, 2013, 2014)
VALID_VINTAGES = (2015,)
TEST_VINTAGES = (2016,)
VINTAGE_COL = "vintage"
LABEL_COL = "default_24m"

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "credit.duckdb"


def check_vintages(train, valid, test) -> None:
    """Fail loudly if the split rules are broken: empty, overlapping, or out of time order."""
    if not train or not valid or not test:
        raise ValueError("Train, validation, and test must each have at least one vintage.")
    all_vintages = list(train) + list(valid) + list(test)
    if len(set(all_vintages)) != len(all_vintages):
        raise ValueError(f"A vintage appears in more than one split: {all_vintages}")
    if max(train) >= min(valid) or max(valid) >= min(test):
        raise ValueError("Splits must be in time order: train < validation < test.")


def split_by_vintage(
    df: pd.DataFrame,
    train=TRAIN_VINTAGES,
    valid=VALID_VINTAGES,
    test=TEST_VINTAGES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, valid_df, test_df). Every row must land in exactly one split."""
    check_vintages(train, valid, test)

    unassigned = set(df[VINTAGE_COL].unique()) - set(train) - set(valid) - set(test)
    if unassigned:
        raise ValueError(f"Vintages not assigned to any split: {sorted(unassigned)}")

    train_df = df[df[VINTAGE_COL].isin(train)].copy()
    valid_df = df[df[VINTAGE_COL].isin(valid)].copy()
    test_df = df[df[VINTAGE_COL].isin(test)].copy()
    return train_df, valid_df, test_df


def split_summary(train_df, valid_df, test_df) -> pd.DataFrame:
    """Loans, defaults, and default rate per split (aggregates only)."""
    rows = []
    for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        rows.append({
            "split": name,
            "vintages": sorted(part[VINTAGE_COL].unique().tolist()),
            "loans": len(part),
            "defaults": int(part[LABEL_COL].sum()),
            "default_rate_%": round(100 * part[LABEL_COL].mean(), 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)
    loans = con.sql("SELECT * FROM loans").df()
    con.close()
    print(split_summary(*split_by_vintage(loans)).to_string(index=False))