-- S1: load Freddie Mac sample files (pipe-delimited, no header) into DuckDB.
-- Layout: SFLLD Release 47 (July 2026). Origination = 31 fields, performance = 35 fields.
-- Every column is VARCHAR on purpose: sentinel values (e.g. 9999 credit score)
-- and codes (e.g. delinquency '03', 'RA', 'XX') are handled explicitly in S2/S3,
-- not lost to a blind cast here.
-- {{ORIG_FILES}} / {{PERF_FILES}} are replaced by prepare_data.py with an explicit file list.

CREATE OR REPLACE TABLE orig_raw AS
SELECT
    CAST(regexp_extract(filename, 'sample_orig_(\d{4})', 1) AS INTEGER) AS vintage,
    * EXCLUDE (filename)
FROM read_csv(
    {{ORIG_FILES}},
    delim = '|',
    header = false,
    filename = true,
    columns = {
        'credit_score': 'VARCHAR',
        'first_payment_date': 'VARCHAR',
        'first_time_homebuyer_flag': 'VARCHAR',
        'maturity_date': 'VARCHAR',
        'msa': 'VARCHAR',
        'mi_pct': 'VARCHAR',
        'num_units': 'VARCHAR',
        'occupancy_status': 'VARCHAR',
        'orig_cltv': 'VARCHAR',
        'orig_dti': 'VARCHAR',
        'orig_upb': 'VARCHAR',
        'orig_ltv': 'VARCHAR',
        'orig_interest_rate': 'VARCHAR',
        'channel': 'VARCHAR',
        'ppm_flag': 'VARCHAR',
        'amortization_type': 'VARCHAR',
        'property_state': 'VARCHAR',
        'property_type': 'VARCHAR',
        'postal_code': 'VARCHAR',
        'loan_seq_num': 'VARCHAR',
        'loan_purpose': 'VARCHAR',
        'orig_loan_term': 'VARCHAR',
        'num_borrowers': 'VARCHAR',
        'seller_name': 'VARCHAR',
        'super_conforming_flag': 'VARCHAR',
        'pre_harp_loan_seq_num': 'VARCHAR',
        'special_eligibility_program': 'VARCHAR',
        'harp_indicator': 'VARCHAR',
        'property_valuation_method': 'VARCHAR',
        'io_indicator': 'VARCHAR',
        'vantage_score': 'VARCHAR'
    }
);

CREATE OR REPLACE TABLE perf_raw AS
SELECT
    CAST(regexp_extract(filename, 'sample_perf_(\d{4})', 1) AS INTEGER) AS vintage,
    * EXCLUDE (filename)
FROM read_csv(
    {{PERF_FILES}},
    delim = '|',
    header = false,
    filename = true,
    columns = {
        'loan_seq_num': 'VARCHAR',
        'monthly_reporting_period': 'VARCHAR',
        'current_actual_upb': 'VARCHAR',
        'current_loan_delinquency_status': 'VARCHAR',
        'loan_age': 'VARCHAR',
        'remaining_months_to_maturity': 'VARCHAR',
        'defect_settlement_date': 'VARCHAR',
        'modification_flag': 'VARCHAR',
        'zero_balance_code': 'VARCHAR',
        'zero_balance_effective_date': 'VARCHAR',
        'current_interest_rate': 'VARCHAR',
        'current_non_interest_bearing_upb': 'VARCHAR',
        'ddlpi': 'VARCHAR',
        'mi_recoveries': 'VARCHAR',
        'net_sale_proceeds': 'VARCHAR',
        'non_mi_recoveries': 'VARCHAR',
        'total_expenses': 'VARCHAR',
        'legal_costs': 'VARCHAR',
        'maintenance_preservation_costs': 'VARCHAR',
        'taxes_insurance': 'VARCHAR',
        'misc_expenses': 'VARCHAR',
        'actual_loss': 'VARCHAR',
        'cumulative_modification_cost': 'VARCHAR',
        'interest_rate_step_indicator': 'VARCHAR',
        'payment_deferral_flag': 'VARCHAR',
        'eltv': 'VARCHAR',
        'zero_balance_removal_upb': 'VARCHAR',
        'delinquent_accrued_interest': 'VARCHAR',
        'delinquency_due_to_disaster': 'VARCHAR',
        'borrower_assistance_plan': 'VARCHAR',
        'current_period_modification_cost': 'VARCHAR',
        'interest_bearing_upb': 'VARCHAR',
        'mi_cancellation_indicator': 'VARCHAR',
        'servicer_name': 'VARCHAR',
        'bankruptcy_cramdown_costs': 'VARCHAR'
    }
);