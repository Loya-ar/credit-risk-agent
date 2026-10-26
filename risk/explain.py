"""S8: SHAP explanations for the LightGBM model, in terms of the 19 original features.

LightGBM computes exact TreeSHAP values itself (pred_contrib=True), in log-odds.
Platt calibration is monotonic, so a feature that raises the raw score also raises
the calibrated probability. One-hot columns are summed back to their original feature.

Limits: SHAP explains the model, not causes in the world; features carrying the same
information (missing DTI and harp_indicator) can split credit between them.

Run from the repo root:  python -m risk.explain
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk.features import CATEGORICAL_FEATURES, FEATURES
from risk.split import DB_PATH, split_by_vintage

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SAMPLE_SIZE = 5000
SEED = 42


def original_feature(column: str) -> str:
    """'num__credit_score' -> 'credit_score'; 'cat__property_state_FL' -> 'property_state'."""
    prefix, _, rest = column.partition("__")
    if prefix == "num":
        return rest
    matches = [f for f in CATEGORICAL_FEATURES if rest == f or rest.startswith(f + "_")]
    if not matches:
        raise ValueError(f"Cannot map column to a feature: {column}")
    return max(matches, key=len)


class Explainer:
    """Wraps the raw LightGBM pipeline (preprocessor + model)."""

    def __init__(self, pipeline):
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["model"]
        self.columns = list(self.prep.get_feature_names_out())
        self.groups = [original_feature(c) for c in self.columns]

    def contributions(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """SHAP values per original feature (log-odds), plus the base value for each row."""
        Xt = self.prep.transform(X[FEATURES])
        contrib = np.asarray(self.model.predict(Xt, pred_contrib=True))
        per_column = pd.DataFrame(contrib[:, :-1], columns=self.columns, index=X.index)
        per_feature = per_column.T.groupby(self.groups).sum().T.reindex(columns=FEATURES, fill_value=0.0)
        return per_feature, contrib[:, -1]

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        """Mean absolute SHAP value per feature: how much each feature moves scores on average."""
        per_feature, _ = self.contributions(X)
        return per_feature.abs().mean().sort_values(ascending=False)

    def explain_one(self, loan: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Top drivers for a single loan (one-row DataFrame), largest effect first."""
        per_feature, _ = self.contributions(loan)
        row = per_feature.iloc[0]
        top = row.abs().sort_values(ascending=False).index[:top_n]
        out = []
        for f in top:
            value = loan.iloc[0][f]
            out.append({
                "feature": f,
                "value": None if pd.isna(value) else value,
                "shap_log_odds": float(row[f]),
                "direction": "raises risk" if row[f] > 0 else "lowers risk",
            })
        return out


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    loans = con.sql("SELECT * FROM loans").df()
    con.close()
    _, _, test_df = split_by_vintage(loans)
    sample = test_df.sample(n=SAMPLE_SIZE, random_state=SEED)

    explainer = Explainer(joblib.load(MODELS_DIR / "lgbm.joblib"))
    importance = explainer.global_importance(sample)

    REPORTS_DIR.mkdir(exist_ok=True)
    importance.rename("mean_abs_shap").to_csv(REPORTS_DIR / "s8_shap_global.csv")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    importance.sort_values().plot.barh(ax=ax)
    ax.set(title=f"Global feature importance (mean |SHAP|, {SAMPLE_SIZE:,} test loans)",
           xlabel="Mean |SHAP value| (log-odds)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "s8_shap_global.png", dpi=120)
    plt.close()

    print("Global importance (mean |SHAP|, log-odds):")
    print(importance.round(4).to_string())

    calibrated = joblib.load(MODELS_DIR / "lgbm_calibrated.joblib")
    p = calibrated.predict_proba(sample[FEATURES])[:, 1]
    for label, idx in [("riskiest", int(np.argmax(p))), ("safest", int(np.argmin(p)))]:
        loan = sample.iloc[[idx]]
        print(f"\n{label} loan in the sample: calibrated probability {p[idx]:.2%}")
        for d in explainer.explain_one(loan):
            print(f"  {d['feature']:<28} value={d['value']!s:<10} {d['shap_log_odds']:+.3f}  {d['direction']}")


if __name__ == "__main__":
    main()