"""S5 split tests: vintages never overlap, and test is always later than train and validation."""
import duckdb
import pandas as pd
import pytest

from risk.split import (
    DB_PATH,
    TEST_VINTAGES,
    TRAIN_VINTAGES,
    VALID_VINTAGES,
    check_vintages,
    split_by_vintage,
)


@pytest.fixture
def toy_loans() -> pd.DataFrame:
    """Three made-up loans per vintage, 2012-2016."""
    vintages = [v for v in range(2012, 2017) for _ in range(3)]
    return pd.DataFrame({
        "loan_seq_num": [f"L{i}" for i in range(len(vintages))],
        "vintage": vintages,
        "default_24m": [0, 0, 1] * 5,
    })


def test_default_config_is_valid():
    check_vintages(TRAIN_VINTAGES, VALID_VINTAGES, TEST_VINTAGES)


def test_vintages_do_not_overlap(toy_loans):
    train, valid, test = split_by_vintage(toy_loans)
    tr, va, te = set(train["vintage"]), set(valid["vintage"]), set(test["vintage"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)


def test_test_is_later_than_train_and_valid(toy_loans):
    train, valid, test = split_by_vintage(toy_loans)
    assert test["vintage"].min() > train["vintage"].max()
    assert test["vintage"].min() > valid["vintage"].max()
    assert valid["vintage"].min() > train["vintage"].max()


def test_every_row_lands_in_exactly_one_split(toy_loans):
    train, valid, test = split_by_vintage(toy_loans)
    assert len(train) + len(valid) + len(test) == len(toy_loans)
    ids = pd.concat([train["loan_seq_num"], valid["loan_seq_num"], test["loan_seq_num"]])
    assert ids.is_unique


def test_overlapping_config_is_rejected():
    with pytest.raises(ValueError, match="more than one split"):
        check_vintages((2012, 2013), (2013,), (2016,))


def test_out_of_order_config_is_rejected():
    with pytest.raises(ValueError, match="time order"):
        check_vintages((2014, 2015), (2016,), (2013,))


def test_unassigned_vintage_is_rejected(toy_loans):
    extra = pd.concat([toy_loans, pd.DataFrame({"loan_seq_num": ["X"], "vintage": [2017], "default_24m": [0]})])
    with pytest.raises(ValueError, match="not assigned"):
        split_by_vintage(extra)


@pytest.mark.skipif(not DB_PATH.exists(), reason="local DuckDB not built (CI has no data)")
def test_real_data_splits_are_clean():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    loans = con.sql("SELECT loan_seq_num, vintage, default_24m FROM loans").df()
    con.close()
    train, valid, test = split_by_vintage(loans)
    assert len(train) + len(valid) + len(test) == len(loans)
    assert len(train) > 0 and len(valid) > 0 and len(test) > 0
    assert set(test["vintage"]) == set(TEST_VINTAGES)
    assert test["vintage"].min() > max(train["vintage"].max(), valid["vintage"].max())