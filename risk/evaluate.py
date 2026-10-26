"""S7: evaluate both models ONCE on the test vintage (2016).

Rules decided before running:
- Calibration: Platt scaling fitted on validation (2015); the calibrated model is final.
- Report all of 2016, and 2016 without disaster-flagged loans (sensitivity check).
- No tuning after seeing these numbers.

Run from the repo root:  python -m risk.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from risk.calibration import PlattCalibrated
from risk.features import FEATURES, LABEL
from risk.split import DB_PATH, split_by_vintage

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
N_BOOT = 1000
SEED = 42

DISASTER_SQL = """
SELECT loan_seq_num,
       MAX(CASE WHEN delinquency_due_to_disaster = 'Y' THEN 1 ELSE 0 END) AS disaster_flag
FROM perf_raw
WHERE TRY_CAST(loan_age AS INTEGER) BETWEEN 0 AND 24
GROUP BY loan_seq_num
"""


def ks_statistic(y, p) -> float:
    """Largest gap between the share of defaulters caught and good loans flagged, over all cut-offs."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def all_metrics(y, p) -> dict:
    return {
        "loans": int(len(y)),
        "defaults": int(np.sum(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "mean_predicted": float(np.mean(p)),
        "actual_rate": float(np.mean(y)),
    }


def decile_table(y, p) -> pd.DataFrame:
    """Default rate by score decile. Decile 10 = riskiest 10% of loans."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    t = df.groupby("decile").agg(loans=("y", "size"), defaults=("y", "sum"), mean_predicted=("p", "mean"))
    t["default_rate"] = t["defaults"] / t["loans"]
    t["lift"] = t["default_rate"] / df["y"].mean()
    t = t.sort_index(ascending=False)
    t["cum_share_of_defaults"] = t["defaults"].cumsum() / t["defaults"].sum()
    return t


def bootstrap_auc_diff(y, p_a, p_b, n_boot=N_BOOT, seed=SEED) -> dict:
    """Paired bootstrap of ROC-AUC(b) - ROC-AUC(a) on the same resampled loans."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    auc_a, auc_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue  # a resample with only one class has no AUC
        auc_a.append(roc_auc_score(y[idx], p_a[idx]))
        auc_b.append(roc_auc_score(y[idx], p_b[idx]))
    auc_a, auc_b = np.array(auc_a), np.array(auc_b)
    diffs = auc_b - auc_a
    return {
        "diff": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "share_b_better": float((diffs > 0).mean()),
        "auc_a_ci": [float(np.percentile(auc_a, 2.5)), float(np.percentile(auc_a, 97.5))],
        "auc_b_ci": [float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5))],
        "n_boot": int(len(diffs)),
    }


def save_plots(y, preds: dict, raw_lgbm) -> None:
    # 1. ROC and precision-recall curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(title="ROC curve, test 2016", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(np.mean(y), color="k", ls="--", lw=1, label="random (default rate)")
    axes[1].set(title="Precision-recall curve, test 2016", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "s7_roc_pr.png", dpi=120)
    plt.close()

    # 2. Calibration: predicted vs. actual default rate, 10 equal-size bins
    fig, ax = plt.subplots(figsize=(5.5, 5))
    curves = {"lightgbm raw": raw_lgbm, **{f"{n} calibrated": p for n, p in preds.items()}}
    top = 0.0
    for name, p in curves.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=name)
        top = max(top, frac.max(), mean_pred.max())
    ax.plot([0, top], [0, top], "k--", lw=1, label="perfect calibration")
    ax.set(title="Calibration, test 2016", xlabel="Mean predicted probability", ylabel="Actual default rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "s7_calibration.png", dpi=120)
    plt.close()

    # 3. Default rate by decile (LightGBM, calibrated)
    t = decile_table(y, preds["lightgbm"]).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index.astype(str), 100 * t["default_rate"])
    ax.axhline(100 * np.mean(y), color="k", ls="--", lw=1, label="overall rate")
    ax.set(title="Default rate by score decile (10 = riskiest), test 2016",
           xlabel="Decile", ylabel="Default rate (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "s7_deciles.png", dpi=120)
    plt.close()


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    loans = con.sql("SELECT * FROM loans").df()
    flags = con.sql(DISASTER_SQL).df()
    con.close()

    _, valid_df, test_df = split_by_vintage(loans)  # train is not needed here
    test_df = test_df.merge(flags, on="loan_seq_num", how="left")
    test_df["disaster_flag"] = test_df["disaster_flag"].fillna(0).astype(int)

    X_va, y_va = valid_df[FEATURES], valid_df[LABEL].astype(int).to_numpy()
    X_te, y_te = test_df[FEATURES], test_df[LABEL].astype(int).to_numpy()
    no_flag = (test_df["disaster_flag"] == 0).to_numpy()

    models = {name: joblib.load(MODELS_DIR / f"{file}.joblib")
              for name, file in [("logreg", "logreg"), ("lightgbm", "lgbm")]}
    calibrated = {name: PlattCalibrated(m).fit(X_va, y_va) for name, m in models.items()}  # 2015 only

    rows, preds, raw_preds = [], {}, {}
    for name in models:
        raw = models[name].predict_proba(X_te)[:, 1]
        cal = calibrated[name].predict_proba(X_te)[:, 1]
        raw_preds[name], preds[name] = raw, cal
        rows.append({"model": name, "version": "raw", "population": "all_test", **all_metrics(y_te, raw)})
        rows.append({"model": name, "version": "calibrated", "population": "all_test", **all_metrics(y_te, cal)})
        rows.append({"model": name, "version": "calibrated", "population": "no_disaster_flag",
                     **all_metrics(y_te[no_flag], cal[no_flag])})
    metrics_df = pd.DataFrame(rows)

    boot = bootstrap_auc_diff(y_te, preds["logreg"], preds["lightgbm"])

    REPORTS_DIR.mkdir(exist_ok=True)
    metrics_df.to_csv(REPORTS_DIR / "s7_test_metrics.csv", index=False)
    for name in models:
        decile_table(y_te, preds[name]).to_csv(REPORTS_DIR / f"s7_deciles_{name}.csv")
    (REPORTS_DIR / "s7_bootstrap.json").write_text(json.dumps(boot, indent=2))
    save_plots(y_te, preds, raw_preds["lightgbm"])
    for name, model in calibrated.items():
        joblib.dump(model, MODELS_DIR / f"{'lgbm' if name == 'lightgbm' else name}_calibrated.joblib")

    pd.set_option("display.width", 200)
    print(f"Test 2016: {len(y_te):,} loans, {int(y_te.sum())} defaults | "
          f"without disaster-flagged loans: {int(no_flag.sum()):,} loans, {int(y_te[no_flag].sum())} defaults\n")
    print(metrics_df.round(4).to_string(index=False))
    print("\nLightGBM (calibrated) by decile:")
    print(decile_table(y_te, preds["lightgbm"]).round(4).to_string())
    print("\nBootstrap, ROC-AUC lightgbm minus logreg:")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in boot.items()}, indent=2))
    print(f"\nSaved reports and plots to {REPORTS_DIR}")


if __name__ == "__main__":
    main()