"""S8: choose a decision threshold by minimizing expected cost under ASSUMED cost ratios.

Cost model (an ASSUMPTION, not a measured number):
    cost = RATIO * missed_defaults + false_declines
A missed default (approved loan that defaults) costs RATIO times a false decline
(declined loan that would have paid).

The threshold is chosen on validation (2015) and applied unchanged to test (2016).
Run from the repo root:  python -m risk.threshold
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk.features import FEATURES, LABEL
from risk.split import DB_PATH, split_by_vintage

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

COST_RATIOS = [10, 20, 40]  # ASSUMED, not measured
PRIMARY_RATIO = 20          # ASSUMED; used by the app in A1


def expected_cost(y, p, threshold: float, ratio: float) -> dict:
    """Decline when p >= threshold. Returns counts, rates, and the assumed cost."""
    y, p = np.asarray(y), np.asarray(p)
    decline = p >= threshold
    tp = int(np.sum(decline & (y == 1)))   # defaults caught
    fp = int(np.sum(decline & (y == 0)))   # good loans wrongly declined
    fn = int(np.sum(~decline & (y == 1)))  # defaults missed
    cost = ratio * fn + fp
    return {
        "threshold": float(threshold),
        "declined": tp + fp,
        "decline_rate": (tp + fp) / len(y),
        "defaults_caught": tp,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "false_declines": fp,
        "missed_defaults": fn,
        "cost": float(cost),
        "cost_per_1000": 1000 * cost / len(y),
    }


def candidate_thresholds(p, n: int = 400) -> np.ndarray:
    """Cut-offs spread across the scores, plus one above every score (= approve everyone)."""
    p = np.asarray(p)
    return np.unique(np.concatenate([np.quantile(p, np.linspace(0, 1, n)), [p.max() + 1e-9]]))


def best_threshold(y, p, ratio: float) -> float:
    cands = candidate_thresholds(p)
    costs = [expected_cost(y, p, t, ratio)["cost"] for t in cands]
    return float(cands[int(np.argmin(costs))])


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    loans = con.sql("SELECT * FROM loans").df()
    con.close()
    _, valid_df, test_df = split_by_vintage(loans)

    model = joblib.load(MODELS_DIR / "lgbm_calibrated.joblib")
    y_va = valid_df[LABEL].astype(int).to_numpy()
    y_te = test_df[LABEL].astype(int).to_numpy()
    p_va = model.predict_proba(valid_df[FEATURES])[:, 1]
    p_te = model.predict_proba(test_df[FEATURES])[:, 1]

    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cands = candidate_thresholds(p_va)
    for ratio in COST_RATIOS:
        t = best_threshold(y_va, p_va, ratio)
        test = expected_cost(y_te, p_te, t, ratio)
        rows.append({
            "cost_ratio_ASSUMED": ratio,
            "threshold_from_2015": t,
            "theory_1_over_1_plus_ratio": 1 / (1 + ratio),
            "valid_cost_per_1000": expected_cost(y_va, p_va, t, ratio)["cost_per_1000"],
            "test_decline_rate": test["decline_rate"],
            "test_defaults_caught": test["defaults_caught"],
            "test_recall": test["recall"],
            "test_precision": test["precision"],
            "test_false_declines": test["false_declines"],
            "test_missed_defaults": test["missed_defaults"],
            "test_cost_per_1000": test["cost_per_1000"],
            "test_approve_all_cost_per_1000": 1000 * ratio * int(y_te.sum()) / len(y_te),
        })
        costs = [expected_cost(y_va, p_va, c, ratio)["cost_per_1000"] for c in cands]
        ax.plot(cands, costs, label=f"ratio {ratio} (assumed)")
        ax.axvline(t, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set(title="Expected cost vs. threshold, validation 2015 (ASSUMED cost ratios)",
           xlabel="Threshold (decline if probability >= this)", ylabel="Cost per 1,000 loans")
    ax.legend()
    plt.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    plt.savefig(REPORTS_DIR / "s8_cost_curves.png", dpi=120)
    plt.close()

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "s8_thresholds.csv", index=False)
    primary = table[table["cost_ratio_ASSUMED"] == PRIMARY_RATIO].iloc[0]
    (REPORTS_DIR / "s8_threshold.json").write_text(json.dumps({
        "cost_ratio_ASSUMED": PRIMARY_RATIO,
        "threshold": float(primary["threshold_from_2015"]),
        "chosen_on": "validation 2015, calibrated LightGBM",
        "note": "The cost ratio is an assumption for a portfolio demo, not a measured business number.",
    }, indent=2))

    pd.set_option("display.width", 200)
    print("Thresholds chosen on 2015, applied to 2016. Cost ratios are ASSUMPTIONS.\n")
    print(table.round(4).T.to_string())
    print(f"\nPrimary (assumed ratio {PRIMARY_RATIO}) threshold saved to reports/s8_threshold.json")


if __name__ == "__main__":
    main()