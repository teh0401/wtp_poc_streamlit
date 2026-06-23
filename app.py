from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from src.adjustment_rules import describe_rules
from src.features import MODEL_FEATURES, build_features_from_raw
from src.model_service import ClarifierModelService, PredictionResult

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
SAMPLE_PATH = ARTIFACT_DIR / "sample_inputs.csv"

st.set_page_config(
    page_title="Clarifier pH Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #081f49;
        --muted: #516383;
        --line: #d7e2ee;
        --panel: #ffffff;
        --soft: #f4f8fc;
        --blue: #0d6efd;
        --blue-dark: #082f66;
        --navy: #061832;
        --green: #14903a;
        --green-soft: #edf9f1;
        --amber: #ef9700;
        --amber-soft: #fff8e8;
        --red: #e95c18;
        --red-soft: #fff3ed;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 70% 0%, rgba(13, 110, 253, 0.08), transparent 30rem),
            linear-gradient(135deg, #f7fbff 0%, #eef5fb 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061832 0%, #06264d 58%, #071427 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f4f9ff;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #d8e8ff;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #061832;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 251, 255, 0);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    .brand {
        display: flex;
        gap: 0.85rem;
        align-items: center;
        margin: 0.7rem 0 2.2rem 0;
    }

    .brand-mark {
        width: 3.2rem;
        height: 3.2rem;
        border: 2px solid #35b7ff;
        border-radius: 50% 50% 50% 8px;
        transform: rotate(-45deg);
        box-shadow: inset 0 0 0 8px rgba(53, 183, 255, 0.12);
    }

    .brand-text {
        font-size: 1.15rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .nav-item {
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.45rem;
        color: #e8f2ff;
        font-weight: 650;
    }

    .nav-item.active {
        background: linear-gradient(135deg, #1d6fff 0%, #0d56d9 100%);
        box-shadow: 0 12px 24px rgba(13, 86, 217, 0.28);
    }

    .plant-card {
        border: 1px solid rgba(205, 229, 255, 0.32);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 3rem;
        background: rgba(255, 255, 255, 0.06);
    }

    .plant-title {
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .plant-status {
        color: #73f5a4;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.4rem;
    }

    .topbar h1 {
        color: var(--ink);
        margin: 0 0 0.35rem 0;
        font-size: 2.05rem;
        line-height: 1.15;
        letter-spacing: 0;
    }

    .topbar p {
        color: var(--muted);
        margin: 0;
        font-size: 1rem;
    }

    .timestamp {
        color: var(--muted);
        white-space: nowrap;
        font-weight: 650;
        padding-top: 0.45rem;
    }

    .panel {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 14px 35px rgba(8, 31, 73, 0.09);
        padding: 1.2rem 1.25rem 1.3rem 1.25rem;
        margin-bottom: 1.15rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 850;
        margin-bottom: 1rem;
    }

    .section-note {
        color: var(--muted);
        font-size: 0.95rem;
        margin-top: 0.65rem;
    }

    .prediction-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 1rem;
    }

    .prediction-card {
        border-radius: 8px;
        border: 1px solid var(--line);
        background: #ffffff;
        padding: 1.3rem;
        min-height: 22rem;
    }

    .prediction-card.low {
        border-color: #8fd8aa;
        background: linear-gradient(135deg, var(--green-soft), #ffffff);
    }

    .prediction-card.medium {
        border-color: #f2c16a;
        background: linear-gradient(135deg, var(--amber-soft), #ffffff);
    }

    .prediction-card.high {
        border-color: #f2a47b;
        background: linear-gradient(135deg, var(--red-soft), #ffffff);
    }

    .prediction-time {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 1.18rem;
        font-weight: 850;
        margin-bottom: 1.35rem;
    }

    .prediction-time.low {
        color: var(--green);
    }

    .prediction-time.medium {
        color: var(--amber);
    }

    .prediction-time.high {
        color: var(--red);
    }

    .prediction-small-label {
        color: #111827;
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }

    .prediction-ph {
        font-size: 4.1rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: 0;
        margin-bottom: 1rem;
    }

    .prediction-ph.low {
        color: var(--green);
    }

    .prediction-ph.medium {
        color: var(--amber);
    }

    .prediction-ph.high {
        color: var(--red);
    }

    .prediction-rule {
        border-top: 1px solid rgba(8, 31, 73, 0.16);
        padding-top: 1rem;
        margin-top: 0.6rem;
    }

    .adjustment-box {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0.85rem;
        align-items: center;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-top: 0.85rem;
        border: 1px solid currentColor;
        background: rgba(255, 255, 255, 0.42);
    }

    .adjustment-icon {
        width: 2.65rem;
        height: 2.65rem;
        border-radius: 999px;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.45rem;
        font-weight: 900;
    }

    .adjustment-box.low,
    .adjustment-text.low {
        color: var(--green);
    }

    .adjustment-box.medium,
    .adjustment-text.medium {
        color: var(--amber);
    }

    .adjustment-box.high,
    .adjustment-text.high {
        color: var(--red);
    }

    .adjustment-icon.low {
        background: var(--green);
    }

    .adjustment-icon.medium {
        background: var(--amber);
    }

    .adjustment-icon.high {
        background: var(--red);
    }

    .adjustment-title {
        font-size: 1.15rem;
        font-weight: 900;
        margin-bottom: 0.15rem;
    }

    .adjustment-copy {
        color: #111827;
        font-size: 0.95rem;
    }

    .info-strip {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0.85rem;
        color: var(--muted);
        align-items: start;
        margin: 1.25rem 0 0.35rem 0;
        padding: 0 1rem;
    }

    .info-badge {
        width: 2.2rem;
        height: 2.2rem;
        border: 2px solid var(--blue);
        border-radius: 999px;
        color: var(--blue);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
    }

    @media (max-width: 900px) {
        .topbar {
            display: block;
        }

        .timestamp {
            margin-top: 0.7rem;
        }

        .prediction-ph {
            font-size: 3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_service() -> ClarifierModelService:
    return ClarifierModelService(ARTIFACT_DIR)


@st.cache_data
def load_samples() -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH, parse_dates=["Time_Stamp"])
    return pd.DataFrame()


def sample_label(samples: pd.DataFrame, index: int) -> str:
    if "Time_Stamp" in samples:
        return str(samples.loc[index, "Time_Stamp"])
    return f"Database row {index + 1}"


def result_theme(adjustment_level: str) -> dict[str, str]:
    themes = {
        "Low": {
            "class": "low",
            "title": "LOW",
            "icon": "OK",
            "message": "Low level of chemical adjustment needed",
            "detail": "Predicted pH is inside the preferred operating band.",
        },
        "Medium": {
            "class": "medium",
            "title": "MEDIUM",
            "icon": "!",
            "message": "Medium level of chemical adjustment needed",
            "detail": "Predicted pH is near the preferred band. Review recent process trend before action.",
        },
        "High": {
            "class": "high",
            "title": "HIGH",
            "icon": "!",
            "message": "High level of chemical adjustment needed",
            "detail": "Predicted pH is outside the preferred and neighboring bands. Treat this as advisory.",
        },
    }
    return themes.get(adjustment_level, themes["High"])


def render_sidebar(samples: pd.DataFrame) -> int:
    st.sidebar.markdown(
        """
        <div class="brand">
            <div class="brand-mark"></div>
            <div class="brand-text">Clarifier pH<br>Prediction</div>
        </div>
        <div class="nav-item active">Dashboard</div>
        <div class="nav-item">Predictions</div>
        <div class="nav-item">History</div>
        <div class="nav-item">Alerts</div>
        <div class="nav-item">Reports</div>
        <div class="nav-item">Settings</div>
        """,
        unsafe_allow_html=True,
    )

    if samples.empty:
        database_index = 0
        st.sidebar.warning("No simulated database rows found.")
    else:
        database_index = st.sidebar.selectbox(
            "Simulated database row",
            options=list(range(len(samples))),
            format_func=lambda i: sample_label(samples, i),
            help="Historical row used to fill lag, rolling, diff, and clarifier features.",
        )

    st.sidebar.markdown(
        """
        <div class="plant-card">
            <div class="plant-title">Treatment Plant 1</div>
            <div class="plant-status">Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return int(database_index)


def render_prediction_result(
    result: PredictionResult,
    metadata: dict[str, Any],
    rule_descriptions: dict[str, str],
) -> None:
    theme = result_theme(result.adjustment_level)
    class_name = theme["class"]
    st.markdown(
        f"""
        <div class="panel">
            <div class="section-title">Predicted Clarifier pH and Chemical Adjustment Need</div>
            <div class="prediction-layout">
                <div class="prediction-card {class_name}">
                    <div class="prediction-time {class_name}">{metadata['horizon_minutes']} Minutes</div>
                    <div class="prediction-small-label">Predicted Clarifier pH</div>
                    <div class="prediction-ph {class_name}">{result.predicted_clarifier_ph:.2f}</div>
                    <div class="prediction-rule">
                        <div class="prediction-small-label">Chemical Adjustment Needed</div>
                        <div class="adjustment-box {class_name}">
                            <div class="adjustment-icon {class_name}">{escape(theme['icon'])}</div>
                            <div>
                                <div class="adjustment-title adjustment-text {class_name}">{escape(theme['title'])}</div>
                                <div class="adjustment-copy">{escape(theme['message'])}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Model: {metadata['model_name']} | Low band: {rule_descriptions['Low']} | "
        f"Medium bands: {rule_descriptions['Medium']}"
    )
    st.info(theme["detail"])


def render_feature_audit(payload: dict[str, float], database_label: str) -> None:
    audit_df = (
        pd.DataFrame([payload], columns=MODEL_FEATURES)
        .T.reset_index()
        .rename(columns={"index": "Feature", 0: "Value"})
    )
    with st.expander("System-computed feature payload"):
        st.caption(f"Lag, rolling, diff, and clarifier fields came from simulated database row: {database_label}.")
        st.dataframe(audit_df, use_container_width=True, hide_index=True)


if not (ARTIFACT_DIR / "model.joblib").exists():
    st.error("Model artifacts are missing. Run `python scripts/export_model.py` first.")
    st.stop()

service = load_service()
samples = load_samples()
metadata = service.metadata
rule_descriptions = describe_rules(service.adjustment_rules)
selected_index = render_sidebar(samples)

if samples.empty:
    st.error("The simulated database file is missing or empty. Regenerate artifacts with `python scripts/export_model.py`.")
    st.stop()

database_row = samples.loc[selected_index, MODEL_FEATURES]
database_label = sample_label(samples, selected_index)
now = datetime.now().replace(second=0, microsecond=0)

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <h1>Clarifier pH Prediction Dashboard</h1>
            <p>Enter raw values to predict clarifier pH and chemical adjustment needs.</p>
        </div>
        <div class="timestamp">{now.strftime("%b %d, %Y %I:%M %p")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="section-title">Input Raw Values</div>', unsafe_allow_html=True)
    input_col1, input_col2, button_col = st.columns([1.2, 1.2, 0.65])
    with input_col1:
        raw_ntu = st.number_input(
            "Raw Turbidity (NTU)",
            min_value=0.0,
            value=float(database_row["NTU_9011_Raw"]),
            step=0.1,
            format="%.3f",
        )
    with input_col2:
        raw_ph = st.number_input(
            "Raw pH",
            min_value=0.0,
            max_value=14.0,
            value=float(database_row["pH_9011_Raw"]),
            step=0.01,
            format="%.3f",
        )
    with button_col:
        st.write("")
        st.write("")
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    st.markdown(
        '<div class="section-note">Provide the latest raw turbidity and pH values. '
        'The app fills historical and computed features from the simulated database.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    payload = build_features_from_raw(raw_ph, raw_ntu, database_row, now, service.feature_names)
    try:
        result = service.predict(payload)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        st.session_state.latest_prediction = result
        st.session_state.latest_payload = payload
        st.session_state.latest_database_label = database_label

if "latest_prediction" in st.session_state:
    render_prediction_result(st.session_state.latest_prediction, metadata, rule_descriptions)
    render_feature_audit(
        st.session_state.latest_payload,
        st.session_state.get("latest_database_label", database_label),
    )
else:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Predicted Clarifier pH and Chemical Adjustment Need</div>
            <div class="section-note">Run a prediction to display the 30-minute forecast and advisory adjustment level.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="info-strip">
        <div class="info-badge">i</div>
        <div>
            Predictions are based on historical data and the exported machine learning model.<br>
            Use the recommended chemical adjustment level as a guideline and adjust based on plant conditions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
