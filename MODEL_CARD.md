# Model Card: 24-Month Mortgage Default Model

**Status:** portfolio demo built on public data. **Not for real credit decisions.**
**Author:** Arjun Loya · **Repo:** github.com/Loya-ar/credit-risk-agent

## 1. Intended use

- **For:** demonstrating credit-risk modeling practice (label design, time-based validation, calibration, cost-based thresholds, explainability) and serving as a scoring tool for the portfolio agent in this repo.
- **Not for:** approving, declining, or pricing real loans; any decision about a real person. The model has not been validated, monitored, or reviewed for fair-lending compliance.

## 2. Data

- **Source:** Freddie Mac Single-Family Loan-Level Dataset, **sample files**, Release 47 (July 2026) layout. Obtained under Freddie Mac's terms of use. **Raw and loan-level data are not included in this repository.**
- **Vintages:** loans originated 2012–2016, 50,000 per year (250,000 loans, 18,949,325 monthly performance rows). Performance data runs through March 2026.
- **Why these vintages:** they avoid the 2006–2008 crisis era, whose loans behave very differently, and every loan's 24-month observation window ends before the 2020 COVID forbearance period.

## 3. Label

**Default = 1** if, within loan ages 0–24, the loan reaches 90+ days delinquent (delinquency status ≥ 3, or `RA`), or terminates through a credit event (zero-balance code 02, 03, or 09).
**Default = 0** if the loan stays active through age 24 with neither, or prepays (code 01) with no prior 90+ delinquency.

| | Loans |
|---|---|
| Labeled | 246,040 |
| Defaults | 1,821 (0.74%) |
| Excluded: no performance records in months 0–24 (records begin at age 25+; not HARP; most likely acquired late, i.e. left-truncated) | 3,495 |
| Excluded: other termination (codes 15, 16, 96) | 464 |
| Excluded: right-censored (records stop before age 24 with no termination) | 1 |

Loans first observed after month 24 are excluded because they have, by construction, survived the window; including them would bias the default rate downward (survivorship bias).
**Why 90 days, not 180:** 1,818 loans reached 90+ days within the window versus 932 that reached 180+; 90 days is the more common industry definition and keeps more events for a rare outcome.

## 4. Features

19 features, all from the **origination file only** (known on the day the loan was made). Nothing from the monthly performance file is used except the label. A test (`tests/test_leakage.py`) enforces this.

- **Numeric (10):** credit_score, orig_dti, orig_ltv, orig_cltv, orig_upb, orig_interest_rate, mi_pct, orig_loan_term, num_units, num_borrowers
- **Categorical (9):** first_time_homebuyer_flag, occupancy_status, channel, loan_purpose, property_type, property_state, super_conforming_flag, harp_indicator, special_program_flag (derived: Y if a special eligibility program is set)

**Excluded columns and why**

| Column | Reason |
|---|---|
| postal_code | Fair-lending proxy risk: fine-grained geography can stand in for protected traits |
| msa | Fine-grained geography (proxy risk) with hundreds of sparse categories (overfitting risk) |
| seller_name | Describes the lender, not the borrower; the lender mix shifts year to year |
| loan IDs, dates, vintage | Identifiers; dates and vintage encode origination time, which the time-based test must not see |
| amortization_type, ppm_flag, io_indicator | Constant in this data |
| vantage_score | Unknown (9999) for every 2012–2016 loan |
| property_valuation_method | Not available (7) for every 2012–2016 loan |
| All performance-file columns | Post-origination information (leakage) |

**Kept with caveats**
- **property_state:** coarser than ZIP or metro, and state-level factors (such as foreclosure rules and regional economies) affect risk. It is still geography, so proxy risk is reduced, not eliminated.
- **orig_interest_rate:** known at origination, but it already reflects the lender's own risk pricing, so part of what the model learns is the lender's existing view of the borrower.

**Data handling:** Freddie Mac "not available" codes (e.g. 9999 for credit score, 999 for DTI/LTV/CLTV/MI) are mapped to missing; a check confirmed none remain. Categories with fewer than 1,000 training loans are grouped as "infrequent."

## 5. Split (time-based, no shuffling)

| Split | Vintages | Loans | Defaults | Default rate |
|---|---|---|---|---|
| Train | 2012, 2013, 2014 | 147,840 | 1,068 | 0.72% |
| Validation | 2015 | 48,600 | 346 | 0.71% |
| Test | 2016 | 49,600 | 407 | 0.82% |

The test vintage was evaluated once, with all rules (calibration method, sensitivity check, threshold procedure) fixed beforehand. Nothing was tuned on it.

## 6. Models

- **Baseline: logistic regression** (C = 0.1). Median imputation with missing-value indicators, standard scaling, one-hot encoding. LTV and CLTV capped at the 99.9th percentile of the training data.
- **Main: LightGBM.** 12-trial random search on validation, early stopping on validation log loss (100 rounds), best trial chosen by validation PR-AUC. Selected settings: learning_rate 0.05, num_leaves 15, min_child_samples 50, subsample 0.85, colsample_bytree 1.0, reg_lambda 0.0, 91 trees.
- **No class weighting:** it tends to distort probabilities, and calibrated probabilities were a goal.
- **Calibration:** Platt scaling fitted on validation (2015) only. Chosen over isotonic regression because validation has only 346 defaults, and isotonic is prone to overfitting with few events.
- Runs tracked in MLflow (local).

## 7. Results

### Validation (2015, used for tuning, so slightly optimistic)

| Metric | Logistic | LightGBM |
|---|---|---|
| ROC-AUC | 0.8087 | 0.8194 |
| PR-AUC | 0.0373 | 0.0386 |
| Brier | 0.0070 | 0.0070 |

### Test (2016, calibrated, evaluated once)

| Metric | Logistic | LightGBM |
|---|---|---|
| ROC-AUC | 0.8008 | **0.8117** |
| PR-AUC (random ≈ 0.0082) | 0.0322 | **0.0340** |
| KS | 0.4580 | **0.4858** |
| Brier | 0.0081 | 0.0080 |
| Log loss | 0.0428 | 0.0424 |

**Is LightGBM better?** A paired bootstrap (1,000 resamples of the test loans) puts the ROC-AUC gain at **0.0109**, 95% interval **[0.0011, 0.0213]**; LightGBM was better in **98.6%** of resamples. The gain is real but small; most of the signal is simple and monotonic.

**Ranking (LightGBM, test deciles):** the riskiest 10% of loans default at **3.85%** (lift **4.69**) and contain **46.9%** of all defaults; the top 30% contain **77.9%**; the safest half contain 41 of 407 defaults.

**Calibration:** on 2016 the model **under-predicts**: mean predicted probability 0.52% raw and **0.65%** calibrated, versus an actual rate of **0.82%**. The gap is largest in the riskiest bins (see `reports/s7_calibration.png`).

Plots: `reports/s7_roc_pr.png`, `reports/s7_calibration.png`, `reports/s7_deciles.png`, `reports/s8_shap_global.png`.

## 8. The 2016 test year: disasters and mix shift

Two findings from EDA shape how these results should be read:

1. **Disaster-related delinquency.** 215 of the 407 test defaults (about half) had a disaster-flagged delinquency month (`delinquency_due_to_disaster = Y`) within the 24-month window; in Florida, 103 of 115. This is consistent with the 2017 hurricane season falling inside 2016 loans' windows, though the flag does not identify the event or prove it caused the default. The label was kept as defined (90+ days late is a real loss risk), and a **sensitivity check** re-scored the test set without the 324 flagged loans (49,276 loans, 192 defaults): LightGBM ROC-AUC **0.8298** and KS **0.5191**; logistic ROC-AUC 0.8016. Calibration then flips to **over**-prediction (0.65% predicted vs. 0.39% actual). Because the flag is recorded mostly on delinquent loans, this view removes far more defaults than good loans and is not a clean population.
2. **HARP mix shift.** HARP refinances made up 35.3% of 2012 loans but 5.1% of 2016 loans, and default about three times as often. The overall default rate looks flat across years, while the non-HARP rate rose from 0.23% (2012) to 0.77% (2016). Excluding disaster-flagged defaults, the non-HARP rate rose from 0.20% (2012) to 0.41% (2014), then stayed flat to slightly lower (0.41%, 0.35%). The model sees fewer HARP loans in 2016 and predicts a safer year, which helps explain the under-prediction.

**Summary:** the model's ranking holds up in a disaster year; its probability levels depend on events no origination data can predict. In practice this calls for monitoring predicted versus actual default rates and periodic recalibration on recent data.

## 9. Decision threshold (ASSUMED cost ratios)

Decline if the calibrated probability ≥ threshold. The threshold minimizes expected cost = **ratio × missed defaults + false declines**, where the ratio (how many times more a missed default costs than a false decline) is an **assumption**, not a measured business number. Thresholds were chosen on 2015 and applied unchanged to 2016.

| Assumed ratio | Threshold | Theory 1/(1+ratio) | Test: declined | Defaults caught | Cost / 1,000 | Approve-all cost / 1,000 |
|---|---|---|---|---|---|---|
| 10 | 9.47% | 9.09% | 0.11% | 0 of 407 | 83.13 | 82.06 |
| **20 (primary)** | **4.93%** | 4.76% | 1.09% | 28 (6.9%) | 163.21 | 164.11 |
| 40 | 1.85% | 2.44% | 7.90% | 161 (39.6%) | 274.11 | 328.23 |

At an assumed ratio of 10 the model is slightly worse than approving everyone on 2016; at 20 it is about break-even; at 40 it cuts the assumed cost by about 16%. Few loans are risky enough to cross a decline threshold because defaults are rare and the data contains only loans lenders already approved. **This model is better suited to ranking, risk tiers, or pricing than to automatic declines.** The demo app uses the primary assumed ratio of 20 (`reports/s8_threshold.json`).

## 10. Explainability (SHAP)

TreeSHAP values computed by LightGBM (`pred_contrib=True`) in log-odds, with one-hot columns summed back to the 19 original features. Mean |SHAP| on 5,000 test loans:

| Feature | Mean \|SHAP\| |
|---|---|
| credit_score | 0.648 |
| num_borrowers | 0.425 |
| orig_dti | 0.261 |
| orig_interest_rate | 0.169 |
| orig_ltv | 0.110 |
| orig_cltv | 0.108 |

`harp_indicator`, `super_conforming_flag`, and `special_program_flag` have zero importance. For HARP, this does **not** mean HARP is unrelated to risk: missing DTI marks the same loans almost perfectly (all 45,481 labeled HARP loans plus 11 others), so the model uses that instead.

**Limits:** SHAP explains the model, not causes in the world. Features carrying the same information (missing DTI and HARP; LTV and CLTV) can split credit between them arbitrarily. Explanations are of the raw LightGBM score; Platt calibration is monotonic, so directions carry over to the calibrated probability.

## 11. Limitations

- **Selection bias:** only loans Freddie Mac purchased are in the data, so applicants lenders declined are never seen. The model cannot be assumed to work on a general applicant population.
- **Macro conditions differ by vintage:** five post-crisis years, one housing market cycle; the test year was disaster-affected.
- **Single data source** and **sample files** (50,000 loans per year), not the full dataset.
- **Validation reuse:** 2015 was used for early stopping, model selection, and calibration, so validation metrics are optimistic.
- **Label includes temporary delinquency:** a 90+ day delinquency counts as default even if the borrower later recovered (for example after a disaster).
- **Fairness not evaluated:** the data has no protected attributes, so disparate impact could not be measured. Fine-grained geography was removed, but property_state and other features may still correlate with protected traits.
- **Cost ratios are assumptions;** a real lender would estimate them from its own loss and revenue data.
- **No monitoring, no recalibration schedule, no challenger process:** all would be required before any real use.

## 12. Open questions

- Why 10,545 HARP loans have LTV of 60% or less (unexpected if HARP served mainly underwater borrowers).
- Why the non-HARP default rate rose from 2012 to 2014 even excluding disaster-flagged defaults (untested).

## 13. Reproduce

```bash
python scripts/prepare_data.py     # needs Freddie Mac sample files in data/raw/
python -m risk.split
python -m risk.train
python -m risk.evaluate
python -m risk.threshold
python -m risk.explain
pytest
```

Exact package versions are pinned in `requirements.txt`.