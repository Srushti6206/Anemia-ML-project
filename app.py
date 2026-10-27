"""
AnemiaSense - Machine Learning-Based Anemia Classification Among Pregnant Women (NFHS-5, India)

A Streamlit app that REUSES the saved Balanced XGBoost model and median imputer.
Nothing is retrained and no metrics are invented: every number shown comes from your
saved files (model_comparison.csv, classification_report.txt) or from the real model.

Artifacts are loaded from the repository root by default. To use another folder,
set the environment variable ANEMIASENSE_ARTIFACT_DIR (e.g. "results").
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("ANEMIASENSE_ARTIFACT_DIR", APP_DIR))
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = APP_DIR / ARTIFACT_DIR

FILES = {
    "model": "balanced_xgboost_model.pkl",
    "imputer": "median_imputer.pkl",
    "features": "feature_names.pkl",
    "comparison": "model_comparison.csv",
    "report": "classification_report.txt",
    "shap": "shap_feature_importance.png",
}

# Target variable v457 (given in the project brief)
CLASS_LABELS = {1: "Severe anemia", 2: "Moderate anemia", 3: "Mild anemia", 4: "Not anemic"}
CLASS_COLORS = {1: "#C0392B", 2: "#E08E0B", 3: "#D4B106", 4: "#2E8B57"}

# Project-stated results, used ONLY as a fallback if model_comparison.csv cannot be parsed.
FALLBACK_RESULTS = pd.DataFrame(
    {
        "Model": ["Random Forest", "XGBoost", "Balanced XGBoost"],
        "Accuracy": [0.4722, 0.4974, 0.3648],
        "Macro F1": [0.2599, 0.2377, 0.2762],
        "Severe Anemia Recall": [0.0, 0.0, 0.2262],
    }
)

FEATURE_LABELS = {
    "v012": "Respondent age (years)",
    "v025": "Type of residence",
    "v104": "Years lived in current place",
    "v106": "Highest education level",
    "v133": "Years of education",
    "v190": "Household wealth index",
    "v208": "Births in the last five years",
    "v212": "Age at first birth (years)",
    "v219": "Living children plus current pregnancy",
    "v225": "Current pregnancy wanted",
    "v214": "Duration of current pregnancy (months)",
    "v445": "BMI (as encoded in dataset)",
    "v212_missing": "Age at first birth missing (indicator)",
}

# Standard DHS recode categories. These follow the public DHS recode manual; confirm them
# against the NFHS-5 data dictionary you used. The numeric survey code is always shown.
CATEGORICAL_CODES = {
    "v025": {1: "Urban", 2: "Rural"},
    "v106": {0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"},
    "v190": {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"},
    "v225": {1: "Wanted then", 2: "Wanted later", 3: "Wanted no more"},
}

# --------------------------------------------------------------------------------------
# Page setup and design system
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="AnemiaSense",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --navy:#0B2A3C; --teal:#0E8C8C; --blue:#2563A8; --bg:#F6F8FA; --card:#FFFFFF;
  --border:#E3E9EE; --muted:#5B6B78;
}
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; }
.stApp { background:var(--bg); }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1200px; }
h1,h2,h3,h4 { color:var(--navy); letter-spacing:-0.01em; }
h1 { font-size:1.8rem !important; font-weight:700 !important; }
h2 { font-size:1.3rem !important; font-weight:600 !important; margin-top:1.4rem; }
h3 { font-size:1.05rem !important; font-weight:600 !important; }

section[data-testid="stSidebar"] { background:var(--navy); }
section[data-testid="stSidebar"] * { color:#E8F1F5; }
section[data-testid="stSidebar"] .stRadio label p { font-size:0.95rem; }
.brand { font-size:1.5rem; font-weight:700; color:#FFFFFF; margin-bottom:0.1rem; }
.tagline { font-size:0.78rem; color:#9FC2CF; margin-bottom:1.2rem; line-height:1.35; }
.side-card { background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12);
  border-radius:12px; padding:0.8rem 0.9rem; font-size:0.8rem; line-height:1.6; margin-top:1rem; }
.side-card b { color:#FFFFFF; }

.card { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:1.1rem 1.25rem; box-shadow:0 1px 3px rgba(11,42,60,0.06); height:100%; }
.kpi-label { font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); font-weight:600; }
.kpi-value { font-size:1.9rem; font-weight:700; color:var(--navy); line-height:1.25; }
.kpi-note { font-size:0.8rem; color:var(--muted); }
.hero { background:linear-gradient(135deg,#0B2A3C 0%,#0E5E73 100%); color:#fff; border-radius:16px;
  padding:1.6rem 1.8rem; margin-bottom:1.2rem; }
.hero h1 { color:#fff; margin:0 0 0.3rem 0; }
.hero p { color:#CFE6EE; margin:0; font-size:0.97rem; max-width:62rem; }
.notice { border-left:4px solid #E08E0B; background:#FFF8E8; border-radius:8px;
  padding:0.8rem 1rem; color:#5A4106; font-size:0.9rem; }
.notice.red { border-left-color:#C0392B; background:#FDEEEC; color:#6B1F17; }
.notice.blue { border-left-color:var(--blue); background:#EAF2FB; color:#17375E; }
.pill { display:inline-block; padding:0.25rem 0.75rem; border-radius:999px; color:#fff; font-weight:600; font-size:0.9rem; }
div[data-testid="stForm"] { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.2rem; }
.stButton>button, .stFormSubmitButton>button { border-radius:10px; font-weight:600; }
.stFormSubmitButton>button[kind="primary"], .stButton>button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
@media (max-width:768px) { .block-container { padding-left:1rem; padding-right:1rem; } .kpi-value { font-size:1.5rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'
    )


def notice(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="notice {kind}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0B2A3C"),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor="#E3E9EE")
    fig.update_yaxes(gridcolor="#E3E9EE")
    return fig


# --------------------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------------------
class ArtifactError(Exception):
    """Raised when a required artifact is missing or incompatible."""


def artifact_path(key: str) -> Path:
    return ARTIFACT_DIR / FILES[key]


@st.cache_resource(show_spinner="Loading trained model and imputer...")
def load_artifacts():
    """Load model, imputer and feature names. Never substitutes a mock model."""
    missing = [FILES[k] for k in ("model", "imputer", "features") if not artifact_path(k).exists()]
    if missing:
        raise ArtifactError(
            f"Missing required file(s) in `{ARTIFACT_DIR}`: {', '.join(missing)}. "
            "Place them in the same folder as app.py (repository root) or set ANEMIASENSE_ARTIFACT_DIR."
        )
    try:
        model = joblib.load(artifact_path("model"))
        imputer = joblib.load(artifact_path("imputer"))
        features = joblib.load(artifact_path("features"))
    except Exception as exc:  # version mismatch, corrupt pickle, etc.
        raise ArtifactError(
            "Could not unpickle the saved artifacts. This is usually a library-version mismatch. "
            "Install the same scikit-learn / xgboost / numpy / pandas versions you used in the notebook "
            "(run `pip freeze` in the notebook environment and copy them into requirements.txt).\n\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    features = [str(f) for f in list(features)]

    # Compatibility checks between model, imputer and feature list
    model_n = getattr(model, "n_features_in_", None)
    if model_n is not None and model_n != len(features):
        raise ArtifactError(
            f"feature_names.pkl lists {len(features)} features but the model expects {model_n}."
        )
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None and [str(n) for n in model_names] != features:
        raise ArtifactError(
            "The model's stored feature names/order differ from feature_names.pkl.\n"
            f"Model: {list(model_names)}\nfeature_names.pkl: {features}"
        )
    if not hasattr(imputer, "transform"):
        raise ArtifactError("median_imputer.pkl does not look like a fitted imputer (no transform method).")

    return model, imputer, features


def imputer_columns(imputer, features: list[str]) -> list[str]:
    """Work out which columns, in which order, the fitted imputer expects."""
    names = getattr(imputer, "feature_names_in_", None)
    if names is not None:
        cols = [str(n) for n in names]
        unknown = [c for c in cols if c not in features]
        if unknown:
            raise ArtifactError(f"Imputer expects columns not in feature_names.pkl: {unknown}")
        return cols
    n = getattr(imputer, "n_features_in_", None)
    if n == len(features):
        return list(features)
    if n == len(features) - 1 and "v212_missing" in features:
        # Imputer was fit before the missingness flag was added
        return [f for f in features if f != "v212_missing"]
    raise ArtifactError(
        f"Cannot align imputer ({n} features) with feature_names.pkl ({len(features)} features)."
    )


def build_model_input(raw: dict, imputer, features: list[str]) -> pd.DataFrame:
    """
    Reproduce training preprocessing:
      1. v212_missing = 1 if age at first birth is missing, else 0 (created BEFORE imputation)
      2. median imputation using the saved, already-fitted imputer (transform only, never fit)
      3. columns ordered exactly as in feature_names.pkl
    """
    row = {k: (np.nan if v is None else float(v)) for k, v in raw.items() if k != "v212_missing"}
    row["v212_missing"] = 1.0 if raw.get("v212") is None else 0.0
    df = pd.DataFrame([row])

    for f in features:
        if f not in df.columns:
            raise ArtifactError(f"Feature `{f}` from feature_names.pkl has no input field in this app.")

    cols = imputer_columns(imputer, features)
    imputed = imputer.transform(df[cols])
    df_imp = pd.DataFrame(imputed, columns=cols, index=df.index)
    for c in features:
        if c not in df_imp.columns:  # e.g. v212_missing not seen by the imputer
            df_imp[c] = df[c]
    return df_imp[features].astype(float)


def decode_classes(model) -> list[int]:
    """Map model class indices to v457 codes (1-4). XGBoost needs 0-based labels, so 0..3 -> 1..4."""
    classes = [int(c) for c in getattr(model, "classes_", [0, 1, 2, 3])]
    return [c + 1 for c in classes] if min(classes) == 0 else classes


# --------------------------------------------------------------------------------------
# Metrics loading (from your saved files only)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_comparison():
    """Return (raw dataframe, tidy dataframe, source note)."""
    path = artifact_path("comparison")
    if not path.exists():
        return None, FALLBACK_RESULTS.copy(), "model_comparison.csv not found - showing project-stated results."
    raw = pd.read_csv(path)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    tidy = pd.DataFrame()
    cols = {c: str(c).lower() for c in raw.columns}

    def find(*keys, exclude=()):
        for c, lc in cols.items():
            if all(k in lc for k in keys) and not any(e in lc for e in exclude):
                return c
        return None

    name_col = next((c for c in raw.columns if raw[c].dtype == object), None)
    acc = find("acc")
    f1 = find("f1", "macro") or find("macro") or find("f1")
    sev = find("severe") or find("recall")
    if name_col is None or acc is None or f1 is None:
        return raw, FALLBACK_RESULTS.copy(), "Could not recognise columns in model_comparison.csv - showing project-stated results."
    tidy["Model"] = raw[name_col].astype(str)
    tidy["Accuracy"] = pd.to_numeric(raw[acc], errors="coerce")
    tidy["Macro F1"] = pd.to_numeric(raw[f1], errors="coerce")
    if sev is not None:
        tidy["Severe Anemia Recall"] = pd.to_numeric(raw[sev], errors="coerce")
    # Normalise percentages (e.g. 49.74) to fractions
    for c in ("Accuracy", "Severe Anemia Recall"):
        if c in tidy and tidy[c].dropna().max() > 1.0:
            tidy[c] = tidy[c] / 100.0
    return raw, tidy, "Loaded from model_comparison.csv"


@st.cache_data(show_spinner=False)
def load_report_text() -> str | None:
    path = artifact_path("report")
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def parse_report(text: str | None) -> pd.DataFrame | None:
    """Parse the per-class rows of a scikit-learn text classification report."""
    if not text:
        return None
    pat = re.compile(r"^\s*(.+?)\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d+)\s*$")
    rows = []
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            rows.append((m.group(1).strip(), *map(float, m.groups()[1:4]), int(m.group(5))))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1-score", "Support"])
    avg = df["Class"].str.contains("avg", case=False)
    per_class = df[~avg].copy()
    # Replace numeric class ids with readable names when they are recognisable
    ids = []
    for c in per_class["Class"]:
        try:
            ids.append(int(float(c)))
        except ValueError:
            ids.append(None)
    if all(i is not None for i in ids) and len(ids) == 4:
        base = 0 if min(ids) == 0 else 1
        per_class["Class"] = [CLASS_LABELS[i - base + 1] for i in ids]
    return pd.concat([per_class, df[avg]], ignore_index=True)


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
PAGES = ["Overview", "Live Prediction", "Model Performance", "SHAP Explainability", "About the Project"]

with st.sidebar:
    st.markdown('<div class="brand">AnemiaSense</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">Understand the model. Explore the data. Interpret predictions.</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown(
        """<div class="side-card"><b>Model information</b><br>
        Model: Balanced XGBoost<br>Data: NFHS-5 India (2019–21)<br>
        Target: v457 (4 classes)<br>Status: educational demo</div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------------------
comparison_raw, results, results_note = load_comparison()


def page_overview():
    st.markdown(
        """<div class="hero"><h1>AnemiaSense</h1>
        <p>Machine learning-based classification of anemia severity among pregnant women using NFHS-5
        (India) survey data. This is an educational demonstration of model behaviour, not an iron-deficiency
        diagnosis system or a clinical decision-support tool.</p></div>""",
        unsafe_allow_html=True,
    )

    best_acc = results.loc[results["Accuracy"].idxmax()]
    best_f1 = results.loc[results["Macro F1"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(card("Best observed accuracy", f"{best_acc['Accuracy']*100:.2f}%", best_acc["Model"]), unsafe_allow_html=True)
    k2.markdown(card("Highest macro F1", f"{best_f1['Macro F1']:.4f}", best_f1["Model"]), unsafe_allow_html=True)
    k3.markdown(card("Target categories", "4", "Severe, moderate, mild, not anemic"), unsafe_allow_html=True)
    k4.markdown(card("Dataset", "NFHS-5", "India, 2019–2021"), unsafe_allow_html=True)
    st.caption(results_note)

    st.markdown("## Model comparison")
    long = results.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1"], var_name="Metric", value_name="Score")
    fig = px.bar(
        long, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_map={"Accuracy": "#2563A8", "Macro F1": "#0E8C8C"}, text_auto=".3f",
        title="Accuracy vs macro F1-score",
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.markdown(
        "Accuracy rewards predicting the majority classes. Macro F1 averages F1 equally across all four "
        "classes, so it is more sensitive to how the minority (severe) class is handled. "
        "The highest-accuracy model is therefore **not necessarily best** for every use case."
    )

    c1, c2 = st.columns(2)
    with c1:
        if "Severe Anemia Recall" in results:
            fig = px.bar(
                results, x="Model", y="Severe Anemia Recall", text_auto=".3f",
                color_discrete_sequence=["#C0392B"], title="Severe anemia recall by model",
            )
            fig.update_yaxes(range=[0, max(0.3, results["Severe Anemia Recall"].max() * 1.3)])
            st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        report_df = parse_report(load_report_text())
        if report_df is not None:
            pc = report_df[~report_df["Class"].str.contains("avg|accuracy", case=False)]
            fig = px.bar(
                pc, x="Class", y="Recall", text_auto=".3f", color_discrete_sequence=["#0E8C8C"],
                title="Per-class recall - Balanced XGBoost",
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Add classification_report.txt to show per-class recall.")
    st.caption(
        "Per-class recall for Random Forest and standard XGBoost is not available because only the "
        "Balanced XGBoost classification report was saved."
    )

    st.markdown("## Why class balancing was used")
    st.markdown(
        """Anemia categories in the survey are imbalanced, so a standard classifier can score reasonable
        accuracy while almost never predicting the rarer, clinically more serious categories. Both Random Forest
        and standard XGBoost had **0% severe-anemia recall** in this project. Balanced XGBoost re-weights classes
        during training, which raised severe-anemia recall and macro F1, at the cost of overall accuracy."""
    )

    st.markdown("## Limitations")
    notice(
        "<b>Read before interpreting any output.</b> Overall accuracy is below 50% for every model. "
        "Balanced XGBoost has a severe-anemia precision of roughly 2%, meaning most severe-anemia predictions "
        "are false alarms. The target classifies anemia severity and does <b>not</b> independently confirm iron "
        "deficiency. Outputs are experimental and must not guide medical decisions.",
        "red",
    )


# --------------------------------------------------------------------------------------
# Page: Live Prediction
# --------------------------------------------------------------------------------------
FORM_KEYS = ["v012", "v025", "v104", "v106", "v133", "v190", "v208", "v212", "v219", "v225", "v214", "v445"]


def reset_form():
    for k in FORM_KEYS:
        st.session_state.pop(f"in_{k}", None)
    st.session_state.pop("prediction", None)


def code_select(key: str, help_text: str):
    options = list(CATEGORICAL_CODES[key].keys())
    return st.selectbox(
        FEATURE_LABELS[key],
        options,
        index=None,
        format_func=lambda c: f"{c} - {CATEGORICAL_CODES[key][c]}",
        placeholder="Select...",
        help=help_text,
        key=f"in_{key}",
    )


def page_prediction():
    st.title("Live Prediction")
    st.write("Enter survey-style values below. The Balanced XGBoost model you trained is called for every prediction.")
    notice(
        "Experimental educational tool. Predictions are not medical advice and must not be used for diagnosis, "
        "treatment or any health decision.",
        "blue",
    )

    try:
        model, imputer, features = load_artifacts()
    except ArtifactError as exc:
        st.error(str(exc))
        return

    with st.expander("Model input check", expanded=False):
        st.write(f"Feature order from `feature_names.pkl` ({len(features)} features):")
        st.code(", ".join(features))
        st.write(f"Imputer columns: `{', '.join(imputer_columns(imputer, features))}`")
        st.write(f"Class order: `{decode_classes(model)}` (v457 codes)")

    st.markdown("&nbsp;")
    with st.form("prediction_form"):
        st.markdown("### Respondent profile")
        a, b, c = st.columns(3)
        with a:
            v012 = st.number_input(FEATURE_LABELS["v012"], 15, 49, value=None, step=1, key="in_v012",
                                   help="Age in completed years (15–49 is the survey range).")
            v025 = code_select("v025", "Survey code (DHS recode): 1 = Urban, 2 = Rural.")
            v104 = st.number_input(FEATURE_LABELS["v104"], 0, 99, value=None, step=1, key="in_v104",
                                   help="Survey value. DHS also uses special codes (e.g. 95 = always, 96 = visitor, 97/98/99 = inconsistent/don't know/missing). Enter the code as it appears in your data.")
        with b:
            v106 = code_select("v106", "Survey code (DHS recode): 0 none, 1 primary, 2 secondary, 3 higher.")
            v133 = st.number_input(FEATURE_LABELS["v133"], 0, 30, value=None, step=1, key="in_v133",
                                   help="Total years of education. Survey special codes (e.g. 97–99) can be entered if that is how your data encodes them.")
            v190 = code_select("v190", "Wealth quintile code (DHS recode): 1 poorest … 5 richest.")
        with c:
            v445 = st.number_input(FEATURE_LABELS["v445"], 0, 9999, value=None, step=1, key="in_v445",
                                   help="Enter the raw encoded value from the dataset (DHS stores BMI × 100, e.g. 2150 = 21.50). Leave blank if unknown - it will be median-imputed exactly as in training.")
            v225 = code_select("v225", "Survey code (DHS recode): 1 wanted then, 2 wanted later, 3 wanted no more.")

        st.markdown("### Reproductive history")
        d, e, f, g = st.columns(4)
        with d:
            v208 = st.number_input(FEATURE_LABELS["v208"], 0, 10, value=None, step=1, key="in_v208",
                                   help="Number of births in the five years before the survey.")
        with e:
            v212 = st.number_input(FEATURE_LABELS["v212"], 10, 49, value=None, step=1, key="in_v212",
                                   help="Optional. Leave blank if there was no previous birth or it is unknown: the missingness indicator is set and the value is median-imputed, as in training.")
        with f:
            v219 = st.number_input(FEATURE_LABELS["v219"], 1, 20, value=None, step=1, key="in_v219",
                                   help="Total living children plus the current pregnancy (at least 1).")
        with g:
            v214 = st.number_input(FEATURE_LABELS["v214"], 1, 10, value=None, step=1, key="in_v214",
                                   help="Months of the current pregnancy at interview.")

        s1, s2, _ = st.columns([1, 1, 4])
        submitted = s1.form_submit_button("Predict", type="primary", use_container_width=True)
        s2.form_submit_button("Reset form", on_click=reset_form, use_container_width=True)

    if submitted:
        raw = dict(v012=v012, v025=v025, v104=v104, v106=v106, v133=v133, v190=v190, v208=v208,
                   v212=v212, v219=v219, v225=v225, v214=v214, v445=v445)
        required = [k for k in FORM_KEYS if k not in ("v212", "v445")]
        missing = [FEATURE_LABELS[k] for k in required if raw[k] is None]
        errors = []
        if missing:
            errors.append("Please fill in: " + ", ".join(missing) + ".")
        if v212 is not None and v012 is not None and v212 > v012:
            errors.append("Age at first birth cannot be greater than respondent age.")
        if errors:
            for msg in errors:
                st.error(msg)
            st.session_state.pop("prediction", None)
        else:
            try:
                with st.spinner("Running the Balanced XGBoost model..."):
                    X = build_model_input(raw, imputer, features)
                    codes = decode_classes(model)
                    if hasattr(model, "predict_proba"):
                        proba = np.asarray(model.predict_proba(X))[0]
                    else:
                        proba = None
                    pred_idx = int(np.argmax(proba)) if proba is not None else int(model.predict(X)[0])
                    pred_code = codes[pred_idx] if proba is not None else (pred_idx + 1 if min(getattr(model, "classes_", [0])) == 0 else pred_idx)
                st.session_state["prediction"] = dict(X=X, proba=proba, codes=codes, pred=pred_code)
            except Exception as exc:
                st.session_state.pop("prediction", None)
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

    pred = st.session_state.get("prediction")
    if pred:
        render_prediction(pred)


def render_prediction(pred: dict):
    code = pred["pred"]
    label = CLASS_LABELS.get(code, f"Class {code}")
    st.markdown("## Result")
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f'<div class="card"><div class="kpi-label">Predicted category (v457 = {code})</div>'
            f'<div style="margin:0.5rem 0"><span class="pill" style="background:{CLASS_COLORS.get(code, "#2563A8")}">{label}</span></div>'
            '<div class="kpi-note">Most likely class according to the model; not a diagnosis.</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        if pred["proba"] is not None:
            dfp = pd.DataFrame(
                {"Category": [CLASS_LABELS.get(c, str(c)) for c in pred["codes"]], "Probability": pred["proba"], "code": pred["codes"]}
            )
            fig = px.bar(
                dfp, x="Probability", y="Category", orientation="h", text=dfp["Probability"].map(lambda p: f"{p:.1%}"),
                color="Category", color_discrete_map={CLASS_LABELS[c]: CLASS_COLORS[c] for c in CLASS_LABELS},
                title="Model class probabilities",
            )
            fig.update_layout(showlegend=False)
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.update_yaxes(categoryorder="array", categoryarray=[CLASS_LABELS[c] for c in (4, 3, 2, 1)])
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        else:
            st.info("This model does not expose class probabilities.")

    st.markdown("### How to interpret this")
    st.markdown(
        "- The probabilities are the model's raw output scores. They are **not** clinically calibrated risk and not diagnostic certainty.\n"
        "- Overall accuracy of the Balanced XGBoost model is only about 36%, so any single prediction is frequently wrong.\n"
        "- The model uses a small set of survey variables and cannot see blood haemoglobin measurements, symptoms or clinical history."
    )
    if code == 1:
        notice(
            "<b>Severe anemia class predicted.</b> This model's severe-anemia precision is only about 2%, so this "
            "output cannot establish a diagnosis and most such predictions are false alarms. No treatment "
            "recommendation is made. If you are concerned about anemia, please seek assessment, including a blood "
            "test, from a qualified healthcare professional.",
            "red",
        )
    else:
        notice("Experimental prediction. If you have concerns about anemia, please consult a qualified healthcare professional.")
    with st.expander("Exact vector sent to the model"):
        st.dataframe(pred["X"], use_container_width=True)


# --------------------------------------------------------------------------------------
# Page: Model Performance
# --------------------------------------------------------------------------------------
def page_performance():
    st.title("Model Performance")
    st.caption(results_note)

    st.markdown("## Model comparison table")
    show = comparison_raw if comparison_raw is not None else results
    st.dataframe(show, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(results, x="Model", y="Accuracy", text_auto=".3f", color_discrete_sequence=["#2563A8"], title="Accuracy")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.bar(results, x="Model", y="Macro F1", text_auto=".4f", color_discrete_sequence=["#0E8C8C"], title="Macro F1-score")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    if "Severe Anemia Recall" in results:
        fig = px.bar(results, x="Model", y="Severe Anemia Recall", text_auto=".3f", color_discrete_sequence=["#C0392B"],
                     title="Severe anemia recall")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    st.markdown("## Balanced XGBoost - per-category metrics")
    text = load_report_text()
    rdf = parse_report(text)
    if rdf is not None:
        pc = rdf[~rdf["Class"].str.contains("avg", case=False)]
        long = pc.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1-score"], var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Class", y="Score", color="Metric", barmode="group", text_auto=".2f",
                     color_discrete_sequence=["#2563A8", "#0E8C8C", "#E08E0B"], title="Precision, recall and F1 by category")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
    if text:
        with st.expander("Raw classification_report.txt", expanded=rdf is None):
            st.code(text)
    elif rdf is None:
        st.warning("classification_report.txt was not found.")

    st.markdown("## Confusion matrix")
    yt, yp = ARTIFACT_DIR / "y_test.pkl", ARTIFACT_DIR / "y_pred.pkl"
    if yt.exists() and yp.exists():
        try:
            from sklearn.metrics import confusion_matrix

            y_true = np.asarray(joblib.load(yt)).ravel()
            y_pred = np.asarray(joblib.load(yp)).ravel()
            labs = sorted(set(y_true) | set(y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=labs)
            names = [CLASS_LABELS.get(int(l) + (1 if min(labs) == 0 else 0), str(l)) for l in labs]
            fig = px.imshow(cm, x=names, y=names, text_auto=True, color_continuous_scale="Teal",
                            labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (saved test predictions)")
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not build confusion matrix from y_test.pkl / y_pred.pkl: {exc}")
    else:
        notice(
            "No confusion matrix is shown because the original test labels and predictions were not saved with "
            "the model. To add one, export <code>y_test</code> and <code>y_pred</code> from the notebook with "
            "<code>joblib.dump(y_test, 'y_test.pkl')</code> and <code>joblib.dump(y_pred, 'y_pred.pkl')</code>, "
            "place them next to app.py and reload; the app will then draw the real matrix.",
            "blue",
        )

    st.markdown("## The accuracy / minority-class recall trade-off")
    st.markdown(
        "Standard XGBoost and Random Forest reach higher accuracy but detect **none** of the severe cases. Balanced "
        "XGBoost gives up accuracy to catch some severe cases (recall of roughly 22.6%), but with very low "
        "precision (about 2%) most of its severe-anemia flags are false positives. Which trade-off is acceptable "
        "depends on the cost of missed cases versus false alarms, which this educational project does not settle."
    )


# --------------------------------------------------------------------------------------
# Page: SHAP
# --------------------------------------------------------------------------------------
def page_shap():
    st.title("SHAP Explainability")
    p = artifact_path("shap")
    if p.exists():
        st.image(str(p), caption="Mean absolute SHAP value by feature (saved from your notebook)", use_container_width=True)
    else:
        st.warning(f"`{FILES['shap']}` was not found in `{ARTIFACT_DIR}`.")

    st.markdown("## How to read this plot")
    st.markdown(
        "- **Mean absolute SHAP value** is the average size of a feature's contribution to the model's output "
        "across the samples used. Longer bars mean the feature moved predictions more on average.\n"
        "- It shows importance **magnitude**, not direction, and not for any single person.\n"
        "- **Importance is not causation.** A high SHAP value does not prove that a feature causes anemia; "
        "it only shows the model relied on it, possibly because it correlates with other factors.\n"
        "- Only the saved plot is shown. Individual-prediction explanations are not generated, because the "
        "training data needed to compute them reliably is not bundled with the app."
    )


# --------------------------------------------------------------------------------------
# Page: About
# --------------------------------------------------------------------------------------
def page_about():
    st.title("About the Project")
    st.markdown("## Problem statement")
    st.write(
        "Anemia during pregnancy is a major public-health concern in India. This project explores whether "
        "routinely collected survey variables can classify the anemia category of pregnant women, and examines "
        "how class imbalance affects what a model learns."
    )
    st.markdown("## Objectives")
    st.markdown(
        "- Classify pregnant respondents into four anemia categories (v457).\n"
        "- Compare Random Forest, XGBoost and Balanced XGBoost.\n"
        "- Evaluate with accuracy, macro F1 and per-class recall, not accuracy alone.\n"
        "- Interpret the final model with SHAP."
    )
    st.markdown("## Dataset")
    st.write("National Family Health Survey-5 (NFHS-5), India, 2019–2021, restricted to currently pregnant respondents.")
    st.markdown("## Machine learning workflow")
    st.markdown(
        "1. Filter NFHS-5 records to currently pregnant respondents.\n"
        "2. Select 12 survey features and create a missingness indicator for age at first birth (`v212_missing`).\n"
        "3. Impute missing values with the median (saved `median_imputer.pkl`).\n"
        "4. Train and evaluate Random Forest, XGBoost and Balanced XGBoost.\n"
        "5. Save the Balanced XGBoost model and explain it with SHAP."
    )
    st.markdown("## Algorithms and metrics")
    st.write("Random Forest, XGBoost, and class-weighted (Balanced) XGBoost. Metrics: accuracy, macro F1-score, per-class precision/recall/F1.")
    st.markdown("## Features")
    fdf = pd.DataFrame({"Feature": list(FEATURE_LABELS), "Meaning": list(FEATURE_LABELS.values())})
    st.dataframe(fdf, use_container_width=True, hide_index=True)
    st.markdown("## Limitations")
    st.markdown(
        "- The target classifies anemia severity; it does **not** independently confirm iron deficiency.\n"
        "- Accuracy is below 50% for all models; severe-anemia precision is about 2%.\n"
        "- Features are limited survey variables, with no haemoglobin or clinical measurements.\n"
        "- Not validated clinically; not a diagnostic or clinical decision-support tool."
    )
    st.markdown("## Future improvements")
    st.markdown(
        "- Probability calibration and threshold tuning for the severe class.\n"
        "- Resampling (e.g. SMOTE) and cost-sensitive learning comparisons.\n"
        "- Cross-validation with confidence intervals; saving test predictions for a confusion matrix.\n"
        "- Richer feature sets and external validation."
    )
    st.markdown("## References and data access")
    st.markdown("- DHS Program: https://dhsprogram.com/\n- NFHS (IIPS): https://rchiips.org/nfhs/")


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------
{
    "Overview": page_overview,
    "Live Prediction": page_prediction,
    "Model Performance": page_performance,
    "SHAP Explainability": page_shap,
    "About the Project": page_about,
}[page]()
