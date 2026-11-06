"""A3: Streamlit demo.

Tabs: ask the agent (Gemini), score a loan (model only, no LLM), explore the
public aggregate tables, and about. Each tab says what it needs when it's missing:
public_data/ for the tables, models/ for scoring, GOOGLE_API_KEY for the agent.

Run from the repo root:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

from agent import public_tables
from agent import tools as agent_tools
from agent.public_tables import CATEGORICAL_DIMENSIONS, MISSING, NUMERIC_BANDS, OTHER
from risk import predict

st.set_page_config(page_title="Credit risk agent", page_icon="🏠", layout="wide")

try:  # Streamlit Cloud keeps the key in secrets; locally it's an environment variable.
    if not os.environ.get("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

PURPOSES = {"Purchase": "P", "Cash-out refinance": "C", "No-cash-out refinance": "N"}
OCCUPANCY = {"Primary residence": "P", "Second home": "S", "Investment": "I"}
PROPERTY_TYPES = {"Single-family": "SF", "Condo": "CO", "Planned unit": "PU",
                  "Manufactured": "MH", "Co-op": "CP"}
DIMENSION_NAMES = {
    "credit_score_band": "Credit score", "ltv_band": "Loan-to-value (%)", "dti_band": "Debt-to-income (%)",
    "interest_rate_band": "Interest rate (%)", "loan_amount_band": "Loan amount ($)",
    "property_state": "State", "loan_purpose": "Loan purpose", "occupancy_status": "Occupancy",
    "channel": "Channel", "property_type": "Property type", "first_time_homebuyer": "First-time buyer",
    "num_borrowers": "Borrowers", "harp": "HARP refinance",
}
EXAMPLES = [
    "What was the default rate for 2016 loans by credit score band?",
    "How did investment properties compare with primary residences in 2016?",
    "What share of all 2016 defaults is in the riskiest score decile?",
    "Score a purchase loan: credit score 700, LTV 90, DTI 40, $250,000 at 4.5%, primary residence, Florida.",
]


# ---------- what's available ----------

def public_db():
    try:
        return public_tables.ensure_public_db()
    except FileNotFoundError:
        return None


DB = public_db()
HAS_MODELS = predict.CALIBRATED_PATH.exists() and predict.RAW_PATH.exists()
HAS_KEY = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


@st.cache_data
def table(name: str, db_path: str, mtime: float) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(f"SELECT * FROM {name}").df()
    finally:
        con.close()


def load(name: str) -> pd.DataFrame:
    return table(name, str(DB), DB.stat().st_mtime)


def pct(x) -> str:
    return "–" if pd.isna(x) else f"{x:.2%}"


# ---------- header ----------

st.title("Mortgage default risk agent")
st.caption(
    "LightGBM default model on Freddie Mac loan-level data (2012–2016 originations), "
    "with a LangGraph agent on Gemini. **Portfolio demo, not a credit decision tool.**"
)
status = [
    ("Public tables", DB is not None), ("Model files", HAS_MODELS), ("Gemini key", HAS_KEY),
]
st.caption(" · ".join(f"{'✅' if ok else '⚪'} {name}" for name, ok in status))

ask_tab, score_tab, explore_tab, about_tab = st.tabs(
    ["Ask the agent", "Score a loan", "Explore the data", "About"]
)


# ---------- ask ----------

def show_trace(result: dict) -> None:
    calls = result.get("tool_calls") or []
    if not calls:
        return
    with st.expander(f"How I got this ({len(calls)} tool call{'s' if len(calls) != 1 else ''})"):
        for c in calls:
            st.markdown(f"**{c['name']}**")
            if c["name"] == "run_sql":
                st.code(c["args"].get("query", ""), language="sql")
            elif c["name"] != "decline":
                st.json(c["args"], expanded=False)
            if c.get("error"):
                st.error(c["error"])
            elif isinstance(c.get("result"), dict) and "rows" in c["result"]:
                st.dataframe(pd.DataFrame(c["result"]["rows"], columns=c["result"]["columns"]),
                             hide_index=True)
            elif c.get("result") is not None:
                st.json(c["result"], expanded=False)
        usage = result.get("usage") or {}
        if usage:
            st.caption(f"Tokens: {usage.get('input_tokens', 0):,} in, {usage.get('output_tokens', 0):,} out")


with ask_tab:
    if DB is None or not HAS_KEY:
        missing = [n for n, ok in [("public tables (python -m scripts.build_public_tables)", DB is not None),
                                   ("GOOGLE_API_KEY", HAS_KEY)] if not ok]
        st.info("The agent needs " + " and ".join(missing) + ".")
    else:
        st.write("Ask about default rates, model performance or thresholds, or describe a loan to score. "
                 "Every number in an answer is checked against the tool outputs.")
        history = st.session_state.setdefault("history", [])
        cols = st.columns(2)
        clicked = None
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
                clicked = example
        for turn in history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                (st.warning if turn["result"]["refused"] else st.write)(turn["result"]["answer"])
                show_trace(turn["result"])
        question = st.chat_input("Ask a question") or clicked
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    from agent.graph import answer
                    try:
                        result = answer(question)
                    except Exception as e:  # network or quota errors
                        result = {"answer": f"The agent failed: {e}", "refused": True, "tool_calls": []}
                (st.warning if result["refused"] else st.write)(result["answer"])
                show_trace(result)
            history.append({"question": question, "result": result})


# ---------- score ----------

with score_tab:
    if not HAS_MODELS:
        st.info("Scoring needs the trained model files in models/ (lgbm_calibrated.joblib, lgbm.joblib).")
    else:
        defaults = predict.load_defaults()
        num = defaults["numeric"]

        def number(label, field, value, step):
            spec = num[field]
            return st.number_input(label, min_value=float(spec["min"]), max_value=float(spec["max"]),
                                   value=float(value), step=float(step))

        with st.form("score"):
            c1, c2, c3 = st.columns(3)
            with c1:
                credit_score = number("Credit score", "credit_score", 740, 10)
                ltv = number("Loan-to-value (%)", "orig_ltv", 80, 1)
                dti = number("Debt-to-income (%)", "orig_dti", 36, 1)
            with c2:
                upb = number("Loan amount ($)", "orig_upb", 250_000, 5_000)
                rate = number("Interest rate (%)", "orig_interest_rate", 4.25, 0.125)
                states = defaults["categorical"]["property_state"]["allowed"]
                state = st.selectbox("State", states, index=states.index("CA") if "CA" in states else 0)
            with c3:
                purpose = st.selectbox("Loan purpose", list(PURPOSES))
                occupancy = st.selectbox("Occupancy", list(OCCUPANCY))
                borrowers = st.selectbox("Borrowers", [2, 1])
            with st.expander("Optional details (typical values are used otherwise)"):
                o1, o2 = st.columns(2)
                property_type = o1.selectbox("Property type", ["(typical)", *PROPERTY_TYPES])
                first_time = o2.selectbox("First-time homebuyer", ["(typical)", "Y", "N"])
            submitted = st.form_submit_button("Score loan", type="primary")

        if submitted:
            features = {
                "credit_score": credit_score, "orig_ltv": ltv, "orig_dti": dti, "orig_upb": upb,
                "orig_interest_rate": rate, "loan_purpose": PURPOSES[purpose],
                "occupancy_status": OCCUPANCY[occupancy], "property_state": state,
                "num_borrowers": borrowers,
            }
            if property_type != "(typical)":
                features["property_type"] = PROPERTY_TYPES[property_type]
            if first_time != "(typical)":
                features["first_time_homebuyer_flag"] = first_time
            try:
                score = agent_tools.score_loan(features)
                explanation = agent_tools.explain_score(features)
            except agent_tools.ToolError as e:
                st.error(str(e))
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Default probability (24 months)", pct(score["probability"]))
                m2.metric("Decision at threshold", score["decision"].title())
                m3.metric("Threshold", pct(score["threshold"]),
                          help=f"Chosen on 2015 loans for an ASSUMED cost ratio of {score['cost_ratio_ASSUMED']}: "
                               "a missed default is assumed to cost that many times a wrongly declined good loan.")
                drivers = pd.DataFrame(explanation["drivers"])
                drivers["factor"] = explanation["summary"]
                chart = alt.Chart(drivers).mark_bar().encode(
                    x=alt.X("shap_log_odds:Q", title="Effect on risk (log-odds; right = riskier)"),
                    y=alt.Y("factor:N", sort=None, title=None),
                    color=alt.condition("datum.shap_log_odds > 0", alt.value("#c2410c"), alt.value("#2563eb")),
                ).properties(height=220)
                st.subheader("Top drivers (SHAP)")
                st.altair_chart(chart, width="stretch")
                if score["filled_with_typical_values"]:
                    st.caption("Filled with typical training values: "
                               + ", ".join(f"{k} = {v}" for k, v in score["filled_with_typical_values"].items()))
                st.caption(score["note"])


# ---------- explore ----------

def bucket_order(dimension: str, buckets) -> list:
    if dimension in NUMERIC_BANDS:
        order = list(NUMERIC_BANDS[dimension][2])
    else:
        order = sorted(b for b in buckets if b not in (MISSING, OTHER))
    return [b for b in order + [MISSING, OTHER] if b in set(buckets)]


with explore_tab:
    if DB is None:
        st.info("Build the public tables first: python -m scripts.build_public_tables")
    else:
        segments = load("segments")
        st.subheader("Default rate by segment")
        c1, c2 = st.columns([1, 2])
        dims = [d for d in DIMENSION_NAMES if d in set(segments["dimension"])]
        dimension = c1.selectbox("Breakdown", dims, format_func=DIMENSION_NAMES.get)
        vintages = sorted(segments["vintage"].unique())
        chosen = c2.multiselect("Vintages", vintages, default=vintages[-1:])
        view = segments[(segments["dimension"] == dimension) & segments["vintage"].isin(chosen)]
        if view.empty:
            st.write("Pick at least one vintage.")
        else:
            order = bucket_order(dimension, view["bucket"])
            chart = alt.Chart(view).mark_bar().encode(
                x=alt.X("bucket:N", sort=order, title=DIMENSION_NAMES[dimension], axis=alt.Axis(labelAngle=0)),
                xOffset="vintage:N",
                y=alt.Y("default_rate:Q", title="Default rate", axis=alt.Axis(format="%")),
                color=alt.Color("vintage:N", title="Vintage"),
                tooltip=["vintage", "bucket", "loans", "defaults",
                         alt.Tooltip("default_rate:Q", format=".2%"),
                         alt.Tooltip("mean_predicted:Q", format=".2%")],
            ).properties(height=320)
            st.altair_chart(chart, width="stretch")
            st.caption(f"Groups under {public_tables.MIN_CELL} loans are pooled into '{OTHER}'. "
                       "Predictions are only shown for the 2016 test vintage.")
            with st.expander("Table"):
                st.dataframe(view.sort_values(["vintage", "bucket"]), hide_index=True)

        st.subheader("2016 test loans by model score decile")
        deciles = load("deciles")
        long = deciles.melt(id_vars="decile", value_vars=["default_rate", "mean_predicted"],
                            var_name="series", value_name="rate")
        long["series"] = long["series"].map({"default_rate": "Actual", "mean_predicted": "Predicted"})
        chart = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("decile:O", title="Score decile (10 = riskiest)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("rate:Q", title="Default rate", axis=alt.Axis(format="%")),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["Actual", "Predicted"], range=["#f97316", "#3b82f6"])),
            tooltip=["decile", "series", alt.Tooltip("rate:Q", format=".2%")],
        ).properties(height=280)
        st.altair_chart(chart, width="stretch")

        st.subheader("Model metrics (2016 test vintage)")
        st.dataframe(load("model_metrics"), hide_index=True)


# ---------- about ----------

with about_tab:
    st.markdown(f"""
**What this is.** A mortgage default model (24-month, 90+ days delinquent) trained on Freddie Mac's
Single-Family Loan-Level Dataset: trained on 2012–2014 originations, tuned on 2015, tested once on 2016.
The agent answers questions through three tools (guarded read-only SQL, scoring, SHAP explanations) and
declines anything else.

**What data you're seeing.** Freddie Mac allows publishing noncommercial research results only if they can't
be used to recreate the dataset, so this demo never shows individual loans. The agent and the charts use
aggregate tables, one breakdown at a time, with groups under {public_tables.MIN_CELL} loans pooled.

**Caveats.** The decision threshold depends on an *assumed* cost ratio. Calibrated probabilities
under-predict the 2016 default level. The original interest rate already reflects the lender's own risk
pricing. Not for real credit decisions.

Code, model card and evaluation: [github.com/Loya-ar/credit-risk-agent](https://github.com/Loya-ar/credit-risk-agent)
""")
