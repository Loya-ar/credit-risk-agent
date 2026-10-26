"""S6: train a logistic regression baseline and a LightGBM model, tuned on validation only.

Run from the repo root:
    python -m risk.train --sample 20000 --trials 2   # quick check that the code works
    python -m risk.train                             # full run

The test vintage (2016) is never used here. It is evaluated exactly once in S7.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from risk.features import FEATURES, LABEL, make_preprocessor
from risk.split import DB_PATH, TRAIN_VINTAGES, VALID_VINTAGES, split_by_vintage
from risk.transforms import QuantileCapper

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "credit-risk-default-24m"
SEED = 42

LOGREG_C_VALUES = [0.01, 0.1, 1.0]  # smaller C = stronger regularization (simpler model)
LGBM_SEARCH_SPACE = {
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [50, 100, 200, 500],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def score_metrics(y_true, p) -> dict:
    """Validation metrics used in S6. The full evaluation (KS, calibration, deciles) is S7."""
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p)),
    }


def xy(df: pd.DataFrame):
    return df[FEATURES], df[LABEL].astype(int).to_numpy()


def train_logreg(train_df, valid_df, c_values=LOGREG_C_VALUES) -> dict:
    """Try a few regularization strengths; keep the best by validation PR-AUC."""
    X_tr, y_tr = xy(train_df)
    X_va, y_va = xy(valid_df)
    trials, best = [], None
    for c in c_values:
        pipe = Pipeline([
            ("cap", QuantileCapper()),
            ("prep", make_preprocessor("linear")),
            ("model", LogisticRegression(C=c, max_iter=2000)),
        ])
        pipe.fit(X_tr, y_tr)
        m = score_metrics(y_va, pipe.predict_proba(X_va)[:, 1])
        trials.append({"C": c, **m})
        if best is None or m["pr_auc"] > best["valid_metrics"]["pr_auc"]:
            best = {"model": pipe, "params": {"C": c}, "valid_metrics": m}
    best["trials"] = pd.DataFrame(trials)
    return best


def train_lgbm(train_df, valid_df, n_trials=12, n_estimators=3000, seed=SEED) -> dict:
    """Small random search with early stopping on validation; keep the best by validation PR-AUC."""
    X_tr, y_tr = xy(train_df)
    X_va, y_va = xy(valid_df)
    prep = make_preprocessor("tree").fit(X_tr)  # learned from training years only
    Xt, Xv = prep.transform(X_tr), prep.transform(X_va)

    rng = np.random.default_rng(seed)
    trials, best = [], None
    for trial in range(n_trials):
        params = {k: rng.choice(v).item() for k, v in LGBM_SEARCH_SPACE.items()}
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, subsample_freq=1, random_state=seed, verbose=-1, **params
        )
        model.fit(
            Xt, y_tr,
            eval_set=[(Xv, y_va)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        m = score_metrics(y_va, model.predict_proba(Xv)[:, 1])
        trials.append({"trial": trial, **params, "best_iteration": int(model.best_iteration_), **m})
        if best is None or m["pr_auc"] > best["valid_metrics"]["pr_auc"]:
            best = {
                "model": Pipeline([("prep", prep), ("model", model)]),
                "params": {**params, "best_iteration": int(model.best_iteration_)},
                "valid_metrics": m,
            }
    best["trials"] = pd.DataFrame(trials)
    return best


def load_train_valid(sample: int | None = None):
    con = duckdb.connect(str(DB_PATH), read_only=True)
    loans = con.sql("SELECT * FROM loans").df()
    con.close()
    train_df, valid_df, _ = split_by_vintage(loans)  # the test split is deliberately thrown away here
    if sample:
        train_df = train_df.sample(n=min(sample, len(train_df)), random_state=SEED)
        valid_df = valid_df.sample(n=min(sample, len(valid_df)), random_state=SEED)
    return train_df, valid_df


def log_run(name: str, result: dict, train_df, valid_df, sample) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "model": name,
            "train_vintages": ",".join(map(str, TRAIN_VINTAGES)),
            "valid_vintages": ",".join(map(str, VALID_VINTAGES)),
            "n_train": len(train_df),
            "n_valid": len(valid_df),
            "n_features": len(FEATURES),
            "sample": sample or "full",
            **result["params"],
        })
        mlflow.log_metrics({f"valid_{k}": v for k, v in result["valid_metrics"].items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline + LightGBM (validation only).")
    parser.add_argument("--trials", type=int, default=12, help="LightGBM random-search trials")
    parser.add_argument("--sample", type=int, default=None, help="use N rows per split (quick check)")
    args = parser.parse_args()

    train_df, valid_df = load_train_valid(args.sample)
    print(f"train: {len(train_df):,} loans, {int(train_df[LABEL].sum())} defaults | "
          f"valid: {len(valid_df):,} loans, {int(valid_df[LABEL].sum())} defaults")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    t0 = time.time()
    logreg = train_logreg(train_df, valid_df)
    print(f"logistic regression done in {time.time() - t0:.0f}s, best C = {logreg['params']['C']}")

    t0 = time.time()
    lgbm = train_lgbm(train_df, valid_df, n_trials=args.trials)
    print(f"LightGBM done in {time.time() - t0:.0f}s ({args.trials} trials)")

    for name, result in [("logreg", logreg), ("lightgbm", lgbm)]:
        log_run(name, result, train_df, valid_df, args.sample)

    print("\nValidation (2015) metrics:")
    print(pd.DataFrame({n: r["valid_metrics"] for n, r in [("logreg", logreg), ("lightgbm", lgbm)]}).round(4))
    print("\nBest LightGBM settings:", lgbm["params"])

    if args.sample is None:  # only full runs save models and reports
        MODELS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        joblib.dump(logreg["model"], MODELS_DIR / "logreg.joblib")
        joblib.dump(lgbm["model"], MODELS_DIR / "lgbm.joblib")
        logreg["trials"].to_csv(REPORTS_DIR / "s6_logreg_trials.csv", index=False)
        lgbm["trials"].to_csv(REPORTS_DIR / "s6_lgbm_trials.csv", index=False)
        summary = {n: {"params": r["params"], "valid_metrics": r["valid_metrics"]}
                   for n, r in [("logreg", logreg), ("lightgbm", lgbm)]}
        (REPORTS_DIR / "s6_training_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSaved models to {MODELS_DIR} and reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()