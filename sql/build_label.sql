-- S2: 24-month default label.
-- Window = performance records with loan_age 0..24 (months since origination),
-- using loan age rather than row count so gaps or late-starting records can't stretch the window.
-- Rule order (first match wins):
--   no window records                         -> excluded_no_window_records
--   90+ days delinquent (status >= 3) or 'RA'  -> default_dq90           (1)
--   zero-balance credit event 02/03/09         -> default_credit_event   (1)
--   voluntary prepayment 01                    -> good_prepaid           (0)
--   other termination 15/16/96                 -> excluded_other_termination
--   active and observed through age 24         -> good_performing_24m    (0)
--   otherwise (active, < 24 months observed)   -> excluded_censored

CREATE OR REPLACE TABLE perf_window AS
SELECT
    loan_seq_num,
    TRY_CAST(loan_age AS INTEGER) AS loan_age,
    current_loan_delinquency_status AS dq_raw,
    -- only convert codes that are all digits; 'RA', 'XX' and blanks stay NULL here
    CASE WHEN regexp_full_match(current_loan_delinquency_status, '[0-9]+')
         THEN TRY_CAST(current_loan_delinquency_status AS INTEGER) END AS dq_months,
    COALESCE(current_loan_delinquency_status = 'RA', false) AS is_reo,
    COALESCE(current_loan_delinquency_status = 'XX', false) AS is_dq_unknown,
    NULLIF(trim(zero_balance_code), '') AS zb_code
FROM perf_raw
WHERE TRY_CAST(loan_age AS INTEGER) BETWEEN 0 AND 24;

CREATE OR REPLACE TABLE labels AS
WITH per_loan AS (
    SELECT
        loan_seq_num,
        MAX(loan_age) AS max_age_in_window,
        MIN(CASE WHEN dq_months >= 3 OR is_reo THEN loan_age END) AS first_dq90_age,
        MIN(CASE WHEN zb_code IS NOT NULL THEN loan_age END) AS zb_age,
        MAX(zb_code) AS zb_code,  -- a loan terminates at most once (checked in prepare_data.py)
        SUM(CASE WHEN is_dq_unknown THEN 1 ELSE 0 END) AS dq_unknown_months
    FROM perf_window
    GROUP BY loan_seq_num
),
reasoned AS (
    SELECT
        o.loan_seq_num,
        o.vintage,
        p.max_age_in_window,
        p.first_dq90_age,
        p.zb_code,
        p.zb_age,
        COALESCE(p.dq_unknown_months, 0) AS dq_unknown_months,
        CASE
            WHEN p.loan_seq_num IS NULL             THEN 'excluded_no_window_records'
            WHEN p.first_dq90_age IS NOT NULL       THEN 'default_dq90'
            WHEN p.zb_code IN ('02', '03', '09')    THEN 'default_credit_event'
            WHEN p.zb_code = '01'                   THEN 'good_prepaid'
            WHEN p.zb_code IS NOT NULL              THEN 'excluded_other_termination'
            WHEN p.max_age_in_window >= 24          THEN 'good_performing_24m'
            ELSE 'excluded_censored'
        END AS label_reason
    FROM orig_raw o
    LEFT JOIN per_loan p ON o.loan_seq_num = p.loan_seq_num
)
SELECT
    *,
    CASE WHEN label_reason LIKE 'default%' THEN 1
         WHEN label_reason LIKE 'good%'    THEN 0
    END AS default_24m   -- NULL for excluded loans
FROM reasoned;