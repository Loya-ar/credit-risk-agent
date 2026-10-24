"""Data preparation for credit-risk-agent.

S1: load Freddie Mac sample files into DuckDB and run integrity checks.
S2: validate status/termination codes, build the 24-month default label, summarize it.

Run from the repo root:
    python scripts/prepare_data.py              # load raw files + checks + labels
    python scripts/prepare_data.py --skip-load  # reuse raw tables already in the DuckDB file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets this script import our `risk` package
import pandas as pd  # noqa: E402
from risk.features import FEATURES, clean_origination  # noqa: E402
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "credit.duckdb"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"

VINTAGES = [2012, 2013, 2014, 2015, 2016]
# SFLLD Release 47 (July 2026) layout: field counts per file type
EXPECTED_FIELDS = {"orig": 31, "perf": 35}
# Zero-balance codes listed in the Release 47 layout
KNOWN_ZB_CODES = ("01", "02", "03", "09", "15", "16", "96")


# ---------- S1: load ----------

def raw_file(kind: str, vintage: int) -> Path:
    return RAW_DIR / f"sample_{kind}_{vintage}.txt"


def count_fields(path: Path) -> int:
    """Count pipe-delimited fields on the first line of a file."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().rstrip("\r\n")
    return first_line.count("|") + 1


def check_files() -> list[str]:
    """Fail fast if a file is missing or its field count doesn't match the layout."""
    problems = []
    for v in VINTAGES:
        for kind, expected in EXPECTED_FIELDS.items():
            path = raw_file(kind, v)
            if not path.exists():
                problems.append(f"missing file: {path}")
                continue
            n = count_fields(path)
            if n != expected:
                problems.append(f"{path.name}: {n} fields, layout expects {expected}")
    return problems


def sql_list(paths: list[Path]) -> str:
    return "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"


def load_raw(con: duckdb.DuckDBPyConnection) -> None:
    sql = (SQL_DIR / "load_raw.sql").read_text()
    sql = sql.replace("{{ORIG_FILES}}", sql_list([raw_file("orig", v) for v in VINTAGES]))
    sql = sql.replace("{{PERF_FILES}}", sql_list([raw_file("perf", v) for v in VINTAGES]))
    con.execute(sql)


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return con.execute(query).fetchone()[0]


def run_raw_checks(con: duckdb.DuckDBPyConnection) -> bool:
    ok = True

    counts = con.sql("""
        WITH o AS (
            SELECT vintage,
                   COUNT(*) AS orig_rows,
                   COUNT(DISTINCT loan_seq_num) AS distinct_loans
            FROM orig_raw GROUP BY vintage
        ),
        p AS (
            SELECT vintage,
                   COUNT(*) AS perf_rows,
                   COUNT(DISTINCT loan_seq_num) AS loans_with_perf,
                   MIN(monthly_reporting_period) AS first_period,
                   MAX(monthly_reporting_period) AS last_period
            FROM perf_raw GROUP BY vintage
        )
        SELECT o.vintage, o.orig_rows, o.distinct_loans,
               p.perf_rows, p.loans_with_perf, p.first_period, p.last_period
        FROM o LEFT JOIN p USING (vintage)
        ORDER BY o.vintage
    """).df()
    print("\nRow counts per vintage:")
    print(counts.to_string(index=False))
    REPORTS_DIR.mkdir(exist_ok=True)
    counts.to_csv(REPORTS_DIR / "s1_row_counts.csv", index=False)  # aggregates only

    total_orig = scalar(con, "SELECT COUNT(*) FROM orig_raw")
    distinct_orig = scalar(con, "SELECT COUNT(DISTINCT loan_seq_num) FROM orig_raw")
    blank_ids = scalar(con, "SELECT COUNT(*) FROM orig_raw WHERE loan_seq_num IS NULL OR trim(loan_seq_num) = ''")
    print(f"\n[check] origination rows = distinct loan IDs: {total_orig} vs {distinct_orig}")
    print(f"[check] blank loan IDs in origination: {blank_ids}")
    if total_orig != distinct_orig or blank_ids > 0:
        ok = False

    orphans = scalar(con, """
        SELECT COUNT(*) FROM perf_raw p
        LEFT JOIN orig_raw o ON p.loan_seq_num = o.loan_seq_num
        WHERE o.loan_seq_num IS NULL
    """)
    print(f"[check] performance rows with no origination row: {orphans}")
    if orphans > 0:
        ok = False

    vintage_mismatch = scalar(con, """
        SELECT COUNT(*) FROM perf_raw p
        JOIN orig_raw o ON p.loan_seq_num = o.loan_seq_num
        WHERE p.vintage <> o.vintage
    """)
    print(f"[check] perf rows whose file vintage differs from origination: {vintage_mismatch}")
    if vintage_mismatch > 0:
        ok = False

    id_year_mismatch = scalar(con, """
        SELECT COUNT(*) FROM orig_raw
        WHERE substr(loan_seq_num, 2, 2) <> right(CAST(vintage AS VARCHAR), 2)
    """)
    print(f"[check] loan IDs whose embedded year differs from file vintage: {id_year_mismatch}")
    if id_year_mismatch > 0:
        ok = False

    no_perf = scalar(con, """
        SELECT COUNT(*) FROM orig_raw o
        WHERE NOT EXISTS (SELECT 1 FROM perf_raw p WHERE p.loan_seq_num = o.loan_seq_num)
    """)
    print(f"[info]  origination loans with zero performance rows: {no_perf}")

    return ok


# ---------- S2: codes + label ----------

def run_code_checks(con: duckdb.DuckDBPyConnection) -> bool:
    """Every code the label depends on must be one we understand."""
    ok = True

    non_numeric_dq = con.sql("""
        SELECT COALESCE(current_loan_delinquency_status, '<NULL>') AS dq_code, COUNT(*) AS rows
        FROM perf_raw
        WHERE current_loan_delinquency_status IS NULL
           OR NOT regexp_full_match(current_loan_delinquency_status, '[0-9]+')
        GROUP BY 1 ORDER BY 2 DESC
    """).df()
    print("\nNon-numeric delinquency codes (all rows):")
    print(non_numeric_dq.to_string(index=False) if len(non_numeric_dq) else "  none")
    unknown_dq = set(non_numeric_dq["dq_code"]) - {"RA", "XX", "<NULL>"}
    print(f"[check] unknown delinquency codes: {sorted(unknown_dq) or 'none'}")
    if unknown_dq:
        ok = False

    max_dq = scalar(con, """
        SELECT MAX(TRY_CAST(current_loan_delinquency_status AS INTEGER)) FROM perf_raw
        WHERE regexp_full_match(current_loan_delinquency_status, '[0-9]+')
    """)
    print(f"[info]  highest numeric delinquency code: {max_dq}")

    zb = con.sql("""
        SELECT NULLIF(trim(zero_balance_code), '') AS zb_code, COUNT(*) AS rows
        FROM perf_raw
        WHERE NULLIF(trim(zero_balance_code), '') IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).df()
    print("\nZero-balance codes (all rows):")
    print(zb.to_string(index=False))
    unknown_zb = set(zb["zb_code"]) - set(KNOWN_ZB_CODES)
    print(f"[check] unknown zero-balance codes: {sorted(unknown_zb) or 'none'}")
    if unknown_zb:
        ok = False

    multi_zb = scalar(con, """
        SELECT COUNT(*) FROM (
            SELECT loan_seq_num FROM perf_raw
            WHERE NULLIF(trim(zero_balance_code), '') IS NOT NULL
            GROUP BY loan_seq_num
            HAVING COUNT(DISTINCT trim(zero_balance_code)) > 1
        )
    """)
    print(f"[check] loans with more than one distinct zero-balance code: {multi_zb}")
    if multi_zb > 0:
        ok = False

    bad_age = scalar(con, """
        SELECT COUNT(*) FROM perf_raw
        WHERE loan_age IS NOT NULL AND TRY_CAST(loan_age AS INTEGER) IS NULL
    """)
    null_age = scalar(con, "SELECT COUNT(*) FROM perf_raw WHERE loan_age IS NULL")
    print(f"[check] loan_age values that aren't integers: {bad_age}")
    print(f"[info]  rows with NULL loan_age: {null_age}")
    if bad_age > 0:
        ok = False

    return ok


def build_labels(con: duckdb.DuckDBPyConnection) -> None:
    con.execute((SQL_DIR / "build_label.sql").read_text())


def summarize_labels(con: duckdb.DuckDBPyConnection) -> bool:
    ok = True
    summary = con.sql("""
        SELECT
            vintage,
            COUNT(*)                                                         AS loans,
            COUNT(default_24m)                                               AS labeled,
            SUM(default_24m)                                                 AS defaults,
            ROUND(AVG(default_24m), 4)                                       AS default_rate,
            COUNT(*) FILTER (WHERE label_reason = 'default_dq90')            AS dq90,
            COUNT(*) FILTER (WHERE label_reason = 'default_credit_event')    AS credit_evt,
            COUNT(*) FILTER (WHERE label_reason = 'good_performing_24m')     AS good_24m,
            COUNT(*) FILTER (WHERE label_reason = 'good_prepaid')            AS prepaid,
            COUNT(*) FILTER (WHERE label_reason = 'excluded_censored')       AS ex_censored,
            COUNT(*) FILTER (WHERE label_reason = 'excluded_other_termination') AS ex_other,
            COUNT(*) FILTER (WHERE label_reason = 'excluded_no_window_records') AS ex_no_window,
            COUNT(*) FILTER (WHERE dq_unknown_months > 0)                    AS loans_with_xx
        FROM labels
        GROUP BY vintage
        ORDER BY vintage
    """).df()
    print("\nLabel summary per vintage (default_rate is among labeled loans):")
    print(summary.to_string(index=False))
    summary.to_csv(REPORTS_DIR / "s2_label_summary.csv", index=False)  # aggregates only

    bad_labels = scalar(con, """
        SELECT COUNT(*) FROM labels
        WHERE (label_reason LIKE 'excluded%') <> (default_24m IS NULL)
    """)
    print(f"\n[check] loans where 'excluded' and 'has no label' disagree: {bad_labels}")
    if bad_labels > 0:
        ok = False

    samples = con.sql("""
        SELECT label_reason, loan_seq_num FROM labels
        QUALIFY row_number() OVER (PARTITION BY label_reason ORDER BY loan_seq_num) = 1
        ORDER BY label_reason
    """).df()
    print("\nOne sample loan per label reason (for hand-checking):")
    print(samples.to_string(index=False))
    return ok

def build_loans_table(con: duckdb.DuckDBPyConnection) -> None:
    """Labeled loans + cleaned origination features -> DuckDB table `loans` and a Parquet file."""
    raw = con.sql("""
        SELECT o.*, l.default_24m
        FROM orig_raw o
        JOIN labels l ON o.loan_seq_num = l.loan_seq_num
        WHERE l.default_24m IS NOT NULL
    """).df()
    features = clean_origination(raw)
    loans = pd.concat(
        [raw[["loan_seq_num", "vintage"]], features, raw[["default_24m"]].astype(int)],
        axis=1,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(PROCESSED_DIR / "loans.parquet", index=False)
    con.register("loans_df", loans)
    con.execute("CREATE OR REPLACE TABLE loans AS SELECT * FROM loans_df")
    con.unregister("loans_df")

    print(f"\nSaved {len(loans):,} labeled loans x {len(FEATURES)} features "
          "-> data/processed/loans.parquet and DuckDB table `loans`")
    missing = loans[FEATURES].isna().sum()
    print("Missing values per feature (after mapping unknown codes):")
    print(missing[missing > 0].to_string())




def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-load", action="store_true",
                        help="reuse raw tables already in data/credit.duckdb")
    args = parser.parse_args()

    if not args.skip_load:
        problems = check_files()
        if problems:
            print("File check failed:")
            for p in problems:
                print("  -", p)
            return 1
        print("File check passed: all 10 files present with expected field counts.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        if not args.skip_load:
            load_raw(con)
        ok = run_raw_checks(con)
        ok = run_code_checks(con) and ok
        build_labels(con)
        ok = summarize_labels(con) and ok
        build_loans_table(con)
    finally:
        con.close()

    print("\nAll checks PASSED" if ok else "\nSome checks FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())