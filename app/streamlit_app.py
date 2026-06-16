# app/streamlit_app.py
# Hotel Cancellation Risk Assistant — Upgraded v2
# All 10 rectifications + UX + visual improvements applied

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hotel Cancellation Risk Assistant",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

APP_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "xgboost_cancellation_model_bundle.joblib"
DATA_PATH  = PROJECT_DIR / "data"  / "hotel_bookings_feature_engineered.csv"


# ============================================================
# CUSTOM CSS — SAGE / BLUSH / CREAM RESORT THEME (enhanced)
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg-main:        #101619;
        --bg-panel:       #182023;
        --bg-panel-soft:  #20292c;
        --sidebar-bg:     #252732;
        --sage:           #7f9a8a;
        --sage-light:     #dfe8e1;
        --sage-muted:     #b8c8bd;
        --cream:          #f3efe8;
        --sand:           #ded6ca;
        --blush:          #ead2c8;
        --rose:           #a4514a;
        --rose-deep:      #873732;
        --ink:            #25332c;
        --muted-text:     #cbd3cd;
        --amber:          #d4a44c;
    }

    .stApp {
        background: linear-gradient(180deg, #101619 0%, #151c1f 100%);
        color: var(--cream);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #252732 0%, #20232c 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * { color: #f3efe8 !important; }

    .block-container {
        max-width: 1350px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 { color: #f3efe8; }

    /* ── HERO ── */
    .hero-card {
        background: linear-gradient(135deg, #5e786b 0%, #8fa798 100%);
        border-radius: 30px;
        padding: 2.4rem 2.4rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.30);
        position: relative;
        overflow: hidden;
    }
    .hero-card::after {
        content: "🏨";
        position: absolute;
        right: 2rem; top: 1.5rem;
        font-size: 5rem;
        opacity: 0.12;
        pointer-events: none;
    }
    .hero-kicker {
        font-size: 0.80rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 700;
        color: #fff7ef;
        opacity: 0.85;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2.85rem;
        font-weight: 700;
        color: #fff8f0;
        line-height: 1.05;
        margin-bottom: 0.8rem;
    }
    .hero-subtitle {
        max-width: 900px;
        font-size: 1.05rem;
        line-height: 1.65;
        color: #fff8f0;
        opacity: 0.93;
    }

    /* ── SECTION HEADINGS ── */
    .section-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f3efe8;
        margin-top: 0.4rem;
        margin-bottom: 0.25rem;
    }
    .section-subtitle {
        color: var(--muted-text);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.35rem;
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        min-height: 160px;
        border-radius: 26px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
        border: 1px solid rgba(255,255,255,0.06);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 0.7rem;
    }
    .metric-card-light  { background: #f3efe8; color: #25332c; }
    .metric-card-sage   { background: #dfe8e1; color: #25332c; }
    .metric-card-blush  { background: #ead2c8; color: #25332c; }
    .metric-card-amber  { background: #f5e6c8; color: #25332c; }
    .metric-card-dark   { background: linear-gradient(180deg, #182023 0%, #20292c 100%); color: #f3efe8; }

    .metric-label   { font-size: 0.92rem; font-weight: 700; letter-spacing: 0.02em; opacity: 0.86; }
    .metric-value   { font-size: 2.1rem;  font-weight: 850; line-height: 1.05; margin: 0.3rem 0; }
    .metric-caption { font-size: 0.92rem; line-height: 1.45; opacity: 0.86; }

    /* ── RISK PILLS ── */
    .risk-pill {
        display: inline-block;
        padding: 0.52rem 1rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    .risk-high   { background: rgba(164,81,74,0.18);  color: #7a2f2a; border: 1px solid rgba(164,81,74,0.30); }
    .risk-medium { background: rgba(180,143,95,0.22); color: #6f5028; border: 1px solid rgba(180,143,95,0.30); }
    .risk-low    { background: rgba(95,122,109,0.22); color: #2d5241; border: 1px solid rgba(95,122,109,0.30); }

    /* ── CONTENT CARDS ── */
    .content-card {
        background: linear-gradient(180deg, #182023 0%, #20292c 100%);
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 12px 32px rgba(0,0,0,0.20);
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        color: #f3efe8;
    }
    .content-card-light {
        background: #f3efe8;
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        color: #25332c;
    }
    .content-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .content-text { font-size: 0.98rem; line-height: 1.65; }
    .content-card ul, .content-card-light ul { margin-top: 0.4rem; padding-left: 1.2rem; }
    .content-card li, .content-card-light li { margin-bottom: 0.58rem; line-height: 1.55; }

    /* ── VALIDATION BANNER ── */
    .validation-error {
        background: rgba(164,81,74,0.14);
        border: 1px solid rgba(164,81,74,0.35);
        border-radius: 16px;
        padding: 1rem 1.35rem;
        color: #c0504a;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }
    .validation-warning {
        background: rgba(212,164,76,0.14);
        border: 1px solid rgba(212,164,76,0.35);
        border-radius: 16px;
        padding: 1rem 1.35rem;
        color: #a07a2a;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }

    /* ── THRESHOLD BADGE ── */
    .threshold-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(127,154,138,0.18);
        border: 1px solid rgba(127,154,138,0.30);
        border-radius: 999px;
        padding: 0.4rem 0.9rem;
        font-size: 0.84rem;
        font-weight: 700;
        color: #b8c8bd;
        margin-bottom: 1rem;
    }

    /* ── SIDEBAR ── */
    .sidebar-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.1rem;
        margin-bottom: 1.1rem;
    }
    .sidebar-title { font-size: 1.18rem; font-weight: 800; margin-bottom: 0.6rem; }
    .sidebar-text  { font-size: 0.93rem; line-height: 1.6; opacity: 0.90; }

    /* ── MODEL PERF TABLE ── */
    .perf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        font-size: 0.94rem;
    }
    .perf-row:last-child { border-bottom: none; }
    .perf-key   { color: #b8c8bd; }
    .perf-value { font-weight: 700; color: #dfe8e1; }

    /* ── INPUTS ── */
    div[data-baseweb="select"] > div {
        background-color: #20292c !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 16px !important;
        color: #f3efe8 !important;
    }
    .stNumberInput input, .stTextInput input {
        background-color: #20292c !important;
        color: #f3efe8 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
    }
    .stFileUploader section {
        background: #182023 !important;
        border: 1px dashed rgba(255,255,255,0.18) !important;
        border-radius: 20px !important;
    }
    .stButton > button {
        border-radius: 14px !important;
        border: none !important;
        font-weight: 800 !important;
        padding: 0.7rem 1rem !important;
        background: linear-gradient(135deg, #7f9a8a 0%, #5f7a6d 100%) !important;
        color: #fff8f0 !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .stDownloadButton > button {
        border-radius: 14px !important;
        border: none !important;
        font-weight: 800 !important;
        padding: 0.7rem 1rem !important;
        background: linear-gradient(135deg, #ead2c8 0%, #d9b8ac 100%) !important;
        color: #25332c !important;
    }
    [data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .small-note {
        color: #cbd3cd;
        font-size: 0.86rem;
        line-height: 1.5;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE INITIALISATION
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "scored_batch" not in st.session_state:
    st.session_state["scored_batch"] = None
if "scored_segment" not in st.session_state:
    st.session_state["scored_segment"] = None
if "batch_source_label" not in st.session_state:
    st.session_state["batch_source_label"] = None


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model_bundle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        st.error(f"⚠️ Model bundle not found at: `{path}`. Make sure you have run notebook 04 to generate it.")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"⚠️ Dataset not found at: `{path}`. Make sure notebook 03 has been run.")
        st.stop()
    return pd.read_csv(path)


model_bundle = load_model_bundle(MODEL_PATH)
df_raw = load_data(DATA_PATH)


# ============================================================
# FLEXIBLE MODEL BUNDLE EXTRACTION
# ============================================================

def get_bundle_value(bundle: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in bundle:
            return bundle[key]
    return default


model_name = get_bundle_value(model_bundle, ["model_name"], "XGBoost")

model = get_bundle_value(
    model_bundle, ["model", "best_model", "xgb_model", "classifier"],
)
preprocessor = get_bundle_value(
    model_bundle, ["preprocessor", "transformer", "feature_pipeline"],
)
selected_threshold = float(get_bundle_value(
    model_bundle,
    ["selected_threshold", "threshold", "best_threshold", "decision_threshold"],
    0.55,
))
model_features = get_bundle_value(
    model_bundle,
    ["model_features", "feature_names", "selected_features", "input_features"],
    [],
)
target_col = get_bundle_value(model_bundle, ["target_column", "target_col"], "is_canceled")

# Model performance metrics saved in bundle (populated by notebook 04)
model_metrics: Dict[str, Any] = get_bundle_value(
    model_bundle,
    ["model_metrics", "metrics", "evaluation_metrics"],
    {},
)

for label, obj, key in [
    ("model",        model,        None),
    ("preprocessor", preprocessor, None),
]:
    if obj is None:
        st.error(f"⚠️ `{label}` object not found inside the saved bundle. Re-run notebook 04.")
        st.stop()

if not model_features:
    st.error("⚠️ Feature list not found inside the saved bundle. Re-run notebook 04.")
    st.stop()


# ============================================================
# COUNTRY CODE LOOKUP (ISO-3166 alpha-3 subset)
# ============================================================

COUNTRY_CODES: List[str] = [
    "PRT","GBR","FRA","ESP","DEU","ITA","BEL","NLD","USA","CHN",
    "BRA","RUS","IRL","POL","CHE","AUT","SWE","NOR","DNK","FIN",
    "GRC","TUR","ROU","HUN","CZE","SVK","HRV","BGR","ISR","ARG",
    "MEX","CAN","AUS","JPN","KOR","IND","ZAF","NGA","EGY","MAR",
    "Other",
]


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data
def enrich_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df.copy()
    if "total_nights" not in output_df.columns:
        weekend = output_df["stays_in_weekend_nights"] if "stays_in_weekend_nights" in output_df.columns else 0
        week    = output_df["stays_in_week_nights"]    if "stays_in_week_nights"    in output_df.columns else 0
        output_df["total_nights"] = weekend + week
    if "estimated_booking_value" not in output_df.columns:
        if "adr" in output_df.columns and "total_nights" in output_df.columns:
            output_df["estimated_booking_value"] = (
                output_df["adr"].fillna(0) * output_df["total_nights"].fillna(0)
            )
        else:
            output_df["estimated_booking_value"] = 0.0
    return output_df


df = enrich_dataframe(df_raw)


def get_default_value(column: str) -> Any:
    if column not in df.columns:
        return 0
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        value = series.median()
        return 0 if pd.isna(value) else value
    mode_values = series.dropna().mode()
    return mode_values.iloc[0] if len(mode_values) > 0 else "Unknown"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_currency(value: float) -> str:
    return f"€{value:,.0f}"


def safe_value(row: pd.Series, col: str, default: Any = "Unknown") -> Any:
    if col not in row.index:
        return default
    value = row[col]
    return default if pd.isna(value) else value


def classify_risk(probability: float) -> str:
    if probability >= max(0.75, selected_threshold + 0.12):
        return "High Risk"
    elif probability >= selected_threshold:
        return "Moderate Risk"
    return "Low Risk"


def risk_pill_class(risk_label: str) -> str:
    return {"High Risk": "risk-high", "Moderate Risk": "risk-medium"}.get(risk_label, "risk-low")


def borderline(probability: float) -> bool:
    """True when prediction is within 5 pp of the decision threshold."""
    return abs(probability - selected_threshold) <= 0.05


# ============================================================
# FEATURE ENGINEERING HELPERS
# ============================================================

def arrival_season_from_month(m: int) -> str:
    if m in [12, 1, 2]: return "Winter"
    if m in [3,  4, 5]: return "Spring"
    if m in [6,  7, 8]: return "Summer"
    return "Autumn"


MONTH_NAMES = {
    1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
    7:"July",8:"August",9:"September",10:"October",11:"November",12:"December",
}


def lead_time_bucket(lt: int) -> str:
    if lt <= 7:   return "Last minute"
    if lt <= 30:  return "Short lead"
    if lt <= 90:  return "Medium lead"
    if lt <= 180: return "Long lead"
    return "Very long lead"


def adr_bucket(adr: float) -> str:
    if adr <= 0:    return "Zero ADR"
    if adr < 75:    return "Low ADR"
    if adr < 150:   return "Medium ADR"
    return "High ADR"


def stay_length_category(nights: int) -> str:
    if nights == 0: return "Zero-night stay"
    if nights <= 2: return "Short stay"
    if nights <= 5: return "Medium stay"
    if nights <= 10: return "Long stay"
    return "Extended stay"


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_booking_inputs(
    adults: int,
    children: int,
    babies: int,
    stays_in_weekend_nights: int,
    stays_in_week_nights: int,
    arrival_year: int,
    arrival_month_number: int,
    arrival_day_of_month: int,
    adr: float,
    lead_time: int,
) -> Tuple[List[str], List[str]]:
    """Returns (errors, warnings). Errors block prediction; warnings are advisory."""
    errors:   List[str] = []
    warnings: List[str] = []

    if adults == 0 and children == 0 and babies == 0:
        errors.append("A booking must have at least one guest (adult, child, or baby).")

    if stays_in_weekend_nights == 0 and stays_in_week_nights == 0:
        errors.append("Total stay must be at least 1 night.")

    # Validate day-of-month against month
    import calendar
    max_day = calendar.monthrange(arrival_year, arrival_month_number)[1]
    if arrival_day_of_month > max_day:
        errors.append(
            f"{MONTH_NAMES[arrival_month_number]} {arrival_year} only has {max_day} days. "
            f"Arrival day {arrival_day_of_month} is not valid."
        )

    if adr < 0:
        errors.append("Average daily rate (ADR) cannot be negative.")

    if adr == 0:
        warnings.append("ADR is 0 — this may be a complimentary or test booking, which can affect the model signal.")

    if lead_time > 500:
        warnings.append(f"Lead time of {lead_time} days is unusually high. Verify this is correct.")

    if adults > 6:
        warnings.append(f"Booking has {adults} adults — this is unusually large. Verify before predicting.")

    return errors, warnings


# ============================================================
# BUILD MANUAL FEATURE ROW
# ============================================================

def build_manual_feature_row(
    hotel, lead_time, arrival_year, arrival_month_number, arrival_day_of_month,
    stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
    meal, country, market_segment, distribution_channel, is_repeated_guest,
    previous_cancellations, previous_bookings_not_canceled, reserved_room_type,
    assigned_room_type, booking_changes, deposit_type, agent, company,
    days_in_waiting_list, customer_type, adr, required_car_parking_spaces,
    total_of_special_requests,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    total_nights = stays_in_weekend_nights + stays_in_week_nights
    estimated_booking_value = adr * total_nights
    previous_total_bookings = previous_cancellations + previous_bookings_not_canceled
    previous_cancellation_rate = (
        previous_cancellations / previous_total_bookings if previous_total_bookings > 0 else 0.0
    )

    import calendar
    safe_day = min(arrival_day_of_month, calendar.monthrange(arrival_year, arrival_month_number)[1])
    arrival_date = pd.Timestamp(year=arrival_year, month=arrival_month_number, day=safe_day)

    row = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": MONTH_NAMES[arrival_month_number],
        "arrival_date_week_number": int(arrival_date.isocalendar().week),
        "arrival_date_day_of_month": arrival_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults, "children": children, "babies": babies,
        "meal": meal, "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent, "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        # engineered
        "lead_time_bucket": lead_time_bucket(lead_time),
        "is_last_minute_booking": 1 if lead_time <= 7 else 0,
        "is_long_lead_booking": 1 if lead_time >= 90 else 0,
        "arrival_month_number": arrival_month_number,
        "arrival_season": arrival_season_from_month(arrival_month_number),
        "is_peak_season": 1 if arrival_month_number in [6,7,8,9] else 0,
        "total_nights": total_nights,
        "is_zero_night_stay": 1 if total_nights == 0 else 0,
        "is_weekend_stay": 1 if stays_in_weekend_nights > 0 else 0,
        "stay_length_category": stay_length_category(total_nights),
        "previous_total_bookings": previous_total_bookings,
        "previous_cancellation_rate": previous_cancellation_rate,
        "has_previous_cancellation": 1 if previous_cancellations > 0 else 0,
        "has_previous_successful_booking": 1 if previous_bookings_not_canceled > 0 else 0,
        "has_booking_changes": 1 if booking_changes > 0 else 0,
        "has_special_requests": 1 if total_of_special_requests > 0 else 0,
        "is_waitlisted": 1 if days_in_waiting_list > 0 else 0,
        "repeat_guest_with_deposit": 1 if is_repeated_guest == 1 and deposit_type != "No Deposit" else 0,
        "repeat_guest_non_refund": 1 if is_repeated_guest == 1 and deposit_type == "Non Refund" else 0,
        "estimated_booking_value": estimated_booking_value,
        "adr_bucket": adr_bucket(adr),
        "has_invalid_adr": 1 if adr < 0 else 0,
        "is_zero_adr": 1 if adr == 0 else 0,
        "is_high_adr": 1 if adr >= 150 else 0,
        "is_high_value_booking": 1 if estimated_booking_value >= df["estimated_booking_value"].quantile(0.75) else 0,
        "room_type_changed": 1 if reserved_room_type != assigned_room_type else 0,
    }

    manual_df = pd.DataFrame([row])

    for feature in model_features:
        if feature not in manual_df.columns:
            manual_df[feature] = get_default_value(feature)

    for feature in model_features:
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                manual_df[feature] = pd.to_numeric(manual_df[feature], errors="coerce").fillna(get_default_value(feature))
            else:
                manual_df[feature] = manual_df[feature].astype(str)

    return manual_df[model_features].copy(), manual_df.copy()


# ============================================================
# PREDICTION FUNCTIONS (with error handling)
# ============================================================

def _align_features(working_df: pd.DataFrame) -> pd.DataFrame:
    """Add missing model features and align dtypes to training data."""
    for feature in model_features:
        if feature not in working_df.columns:
            working_df[feature] = get_default_value(feature)
    for feature in model_features:
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                working_df[feature] = pd.to_numeric(working_df[feature], errors="coerce").fillna(get_default_value(feature))
            else:
                working_df[feature] = working_df[feature].astype(str)
    return working_df


def predict_bookings(input_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Returns (scored_df, error_message). error_message is None on success."""
    try:
        working_df = enrich_dataframe(input_df.copy())
        working_df = _align_features(working_df)
        X_input = working_df[model_features].copy()
        X_processed = preprocessor.transform(X_input)
        probabilities = model.predict_proba(X_processed)[:, 1]
        predictions   = (probabilities >= selected_threshold).astype(int)
        scored = working_df.copy()
        scored["predicted_cancellation_probability"] = probabilities
        scored["predicted_is_cancelled"]             = predictions
        scored["risk_category"]                      = [classify_risk(p) for p in probabilities]
        scored["estimated_revenue_at_risk"]          = (
            scored["estimated_booking_value"].fillna(0) * probabilities
        )
        return scored, None
    except Exception as exc:
        return None, str(exc)


def predict_manual_booking(
    manual_model_df: pd.DataFrame,
    manual_full_df:  pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        X_processed  = preprocessor.transform(manual_model_df)
        probability  = float(model.predict_proba(X_processed)[:, 1][0])
        prediction   = int(probability >= selected_threshold)
        result_df    = manual_full_df.copy()
        result_df["predicted_cancellation_probability"] = probability
        result_df["predicted_is_cancelled"]             = prediction
        result_df["risk_category"]                      = classify_risk(probability)
        result_df["estimated_revenue_at_risk"]          = (
            result_df["estimated_booking_value"].fillna(0) * probability
        )
        return result_df, None
    except Exception as exc:
        return None, str(exc)


# ============================================================
# SHAP EXPLAINER (cached)
# ============================================================

@st.cache_resource
def get_shap_explainer():
    if not SHAP_AVAILABLE:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception:
        return None


def get_shap_drivers(manual_model_df: pd.DataFrame, top_n: int = 6) -> Optional[List[Tuple[str, float]]]:
    """Returns list of (feature_label, shap_value) for the top_n drivers, or None."""
    explainer = get_shap_explainer()
    if explainer is None:
        return None
    try:
        X_proc = preprocessor.transform(manual_model_df)
        shap_values = explainer.shap_values(X_proc)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
        feature_names = manual_model_df.columns.tolist()
        pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
        return [(name.replace("_", " ").title(), float(val)) for name, val in pairs[:top_n]]
    except Exception:
        return None


# ============================================================
# RULE-BASED RISK DRIVERS (fallback)
# ============================================================

def get_risk_drivers_rules(row: pd.Series) -> List[str]:
    drivers = []
    try:
        if float(safe_value(row, "lead_time", 0)) >= 60:
            drivers.append("Long lead time gives guests more time to change plans before arrival.")
        elif float(safe_value(row, "lead_time", 0)) <= 7:
            drivers.append("Very short lead time — may behave differently from normal booking patterns.")
    except Exception:
        pass
    if safe_value(row, "market_segment") in ["Online TA", "Groups", "Offline TA/TO"]:
        drivers.append(f"Booking is from the {safe_value(row, 'market_segment')} segment, which typically shows higher cancellation sensitivity.")
    if safe_value(row, "customer_type") in ["Transient", "Transient-Party"]:
        drivers.append(f"Customer type is {safe_value(row, 'customer_type')} — less commitment than contract or group demand.")
    if safe_value(row, "deposit_type") in ["No Deposit", "Refundable"]:
        drivers.append("Deposit profile allows easier cancellation compared to non-refundable commitment.")
    try:
        if float(safe_value(row, "total_of_special_requests", 1)) == 0:
            drivers.append("No special requests — may indicate lower guest engagement with the stay.")
    except Exception:
        pass
    try:
        if float(safe_value(row, "previous_cancellations", 0)) > 0:
            drivers.append("Guest has a previous cancellation in their booking history.")
    except Exception:
        pass
    try:
        if float(safe_value(row, "room_type_changed", 0)) == 1:
            drivers.append("Reserved and assigned room types differ — this is a meaningful model signal.")
    except Exception:
        pass
    try:
        if float(safe_value(row, "adr", 0)) >= 150:
            drivers.append("Higher ADR can increase price sensitivity when combined with other risk signals.")
    except Exception:
        pass
    if not drivers:
        drivers.append("Risk comes from a combination of model patterns rather than one dominant signal.")
    return drivers[:6]


def get_risk_drivers(row: pd.Series, manual_model_df: Optional[pd.DataFrame] = None) -> Tuple[List[str], bool]:
    """Returns (driver_list, is_shap_derived)."""
    if manual_model_df is not None and SHAP_AVAILABLE:
        shap_pairs = get_shap_drivers(manual_model_df)
        if shap_pairs:
            lines = []
            for name, val in shap_pairs:
                direction = "↑ increases" if val > 0 else "↓ reduces"
                lines.append(f"<strong>{name}</strong> — {direction} cancellation risk (SHAP: {val:+.3f})")
            return lines, True
    return get_risk_drivers_rules(row), False


def get_recommended_actions(risk_label: str, row: pd.Series) -> List[str]:
    actions = []
    if risk_label == "High Risk":
        actions.extend([
            "Send a personalised reconfirmation message to the guest.",
            "Review payment, deposit, and cancellation-policy exposure.",
            "Flag the reservation for closer monitoring by the revenue or reservations team.",
        ])
    elif risk_label == "Moderate Risk":
        actions.extend([
            "Send a pre-arrival engagement message or soft reminder.",
            "Monitor the booking closer to arrival before making inventory decisions.",
            "Consider a light upsell or add-on offer to increase guest commitment.",
        ])
    else:
        actions.extend([
            "Keep the booking under normal monitoring.",
            "No urgent intervention required based on the current model signal.",
            "Use standard pre-arrival communication.",
        ])
    if safe_value(row, "market_segment") == "Online TA":
        actions.append("For Online TA bookings, timely guest communication reduces passive cancellations.")
    try:
        if float(safe_value(row, "lead_time", 0)) >= 60:
            actions.append("Long lead-time booking — schedule a reminder closer to the arrival date.")
    except Exception:
        pass
    return actions[:5]


def create_manager_summary(row: pd.Series) -> str:
    hotel        = safe_value(row, "hotel", "the property")
    market       = safe_value(row, "market_segment", "Unknown")
    ctype        = safe_value(row, "customer_type", "Unknown")
    probability  = float(row["predicted_cancellation_probability"])
    risk_label   = row["risk_category"]
    bv           = float(safe_value(row, "estimated_booking_value", 0))
    rev_risk     = float(safe_value(row, "estimated_revenue_at_risk", 0))

    action_phrase = {
        "High Risk":      "should be reviewed immediately as part of the revenue-protection workflow",
        "Moderate Risk":  "should be monitored and lightly engaged before arrival",
    }.get(risk_label, "does not require urgent intervention")

    borderline_note = (
        " <em>Note: this prediction is close to the decision threshold — treat it with additional caution.</em>"
        if borderline(probability) else ""
    )

    return (
        f"This booking for <strong>{hotel}</strong> is classified as <strong>{risk_label}</strong>, "
        f"with an estimated cancellation probability of <strong>{format_percent(probability)}</strong>.{borderline_note} "
        f"It belongs to the <strong>{market}</strong> segment and the "
        f"<strong>{ctype}</strong> customer type. Estimated booking value: "
        f"<strong>{format_currency(bv)}</strong>, with approximately "
        f"<strong>{format_currency(rev_risk)}</strong> of probability-weighted revenue exposure. "
        f"Based on this profile, the booking {action_phrase}."
    )


# ============================================================
# PLOTLY CHART HELPERS (on-theme)
# ============================================================

PLOT_BG   = "#182023"
PLOT_PAPER = "#101619"
PLOT_FONT = "#f3efe8"
PLOT_GRID = "rgba(255,255,255,0.07)"

SEGMENT_COLORS = px.colors.sequential.Teal


def plotly_bar_h(chart_df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = px.bar(
        chart_df, x=x_col, y=y_col, orientation="h",
        color=x_col, color_continuous_scale="teal",
        title=title,
    )
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_FONT, family="Georgia, serif"),
        title_font=dict(size=16, color=PLOT_FONT),
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID),
        yaxis=dict(gridcolor=PLOT_GRID),
    )
    return fig


def plotly_donut(probability: float, risk_label: str) -> go.Figure:
    colours = {
        "High Risk":     ["#a4514a", "#2a1a1a"],
        "Moderate Risk": ["#d4a44c", "#2a2010"],
        "Low Risk":      ["#7f9a8a", "#151c1f"],
    }
    fill_pct = probability * 100
    c1, c2 = colours.get(risk_label, colours["Low Risk"])
    fig = go.Figure(go.Pie(
        values=[fill_pct, 100 - fill_pct],
        hole=0.72,
        marker=dict(colors=[c1, c2]),
        textinfo="none",
        hoverinfo="skip",
        sort=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text=f"<b>{fill_pct:.0f}%</b>",
            x=0.5, y=0.5, font_size=28, showarrow=False,
            font=dict(color=c1, family="Georgia, serif"),
        )],
        height=180,
    )
    return fig


def plotly_shap_bar(shap_pairs: List[Tuple[str, float]]) -> go.Figure:
    names = [p[0] for p in reversed(shap_pairs)]
    vals  = [p[1] for p in reversed(shap_pairs)]
    colors = ["#a4514a" if v > 0 else "#7f9a8a" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_FONT, family="Georgia, serif"),
        title=dict(text="SHAP Feature Contributions", font_size=15, font_color=PLOT_FONT),
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(title="SHAP value (positive = more cancellation risk)",
                   gridcolor=PLOT_GRID, zerolinecolor="#b8c8bd"),
        yaxis=dict(gridcolor=PLOT_GRID),
        height=320,
    )
    return fig


def plotly_monthly_trend(scored_df: pd.DataFrame) -> go.Figure:
    if "arrival_month_number" not in scored_df.columns:
        return None
    trend = (
        scored_df.groupby("arrival_month_number", dropna=False)
        .agg(
            avg_risk=("predicted_cancellation_probability", "mean"),
            bookings=("predicted_cancellation_probability", "count"),
            revenue_at_risk=("estimated_revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values("arrival_month_number")
    )
    trend["month"] = trend["arrival_month_number"].map(MONTH_NAMES)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["month"], y=trend["avg_risk"] * 100,
        mode="lines+markers",
        line=dict(color="#7f9a8a", width=3),
        marker=dict(size=8, color="#dfe8e1"),
        name="Avg Cancellation Risk (%)",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_FONT, family="Georgia, serif"),
        title=dict(text="Average Cancellation Risk by Arrival Month", font_size=15, font_color=PLOT_FONT),
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID),
        yaxis=dict(title="Avg Risk (%)", gridcolor=PLOT_GRID),
    )
    return fig


# ============================================================
# UI COMPONENT HELPERS
# ============================================================

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Revenue Protection · Resort Operations · Machine Learning</div>
            <div class="hero-title">Hotel Cancellation Risk Assistant</div>
            <div class="hero-subtitle">
                Predict cancellation risk for new bookings, understand the drivers behind each prediction,
                and support revenue-protection decisions with an explainable machine-learning assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, caption: str, card_type: str = "light") -> None:
    css = {"sage": "metric-card-sage", "blush": "metric-card-blush",
           "dark": "metric-card-dark", "amber": "metric-card-amber"}.get(card_type, "metric-card-light")
    st.markdown(
        f"""
        <div class="metric-card {css}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(risk_label: str, probability: float) -> None:
    pill_cls  = risk_pill_class(risk_label)
    card_type = {"High Risk": "metric-card-blush", "Moderate Risk": "metric-card-amber"}.get(risk_label, "metric-card-sage")
    caption   = {"High Risk": "Immediate manager attention recommended.",
                 "Moderate Risk": "Monitor and engage before arrival."}.get(risk_label, "Standard monitoring is sufficient.")
    st.markdown(
        f"""
        <div class="metric-card {card_type}">
            <div class="metric-label">Risk Assessment</div>
            <div class="metric-value">{format_percent(probability)}</div>
            <div><span class="risk-pill {pill_cls}">{risk_label}</span></div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list_card(title: str, items: List[str], light: bool = False) -> None:
    css = "content-card-light" if light else "content-card"
    list_html = "".join([f"<li>{item}</li>" for item in items])
    st.markdown(
        f'<div class="{css}"><div class="content-title">{title}</div><ul>{list_html}</ul></div>',
        unsafe_allow_html=True,
    )


def render_text_card(title: str, text: str, light: bool = False) -> None:
    css = "content-card-light" if light else "content-card"
    st.markdown(
        f'<div class="{css}"><div class="content-title">{title}</div><div class="content-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_error_banner(msg: str) -> None:
    st.markdown(f'<div class="validation-error">⛔ {msg}</div>', unsafe_allow_html=True)


def render_warning_banner(msg: str) -> None:
    st.markdown(f'<div class="validation-warning">⚠️ {msg}</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">🏨 Assistant Controls</div>
            <div class="sidebar-text">
                Predict cancellation risk for new bookings, review batch exposure,
                and analyse segment-level revenue risk.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Choose analysis mode",
        [
            "🔮 Predict New Booking",
            "🔍 Explain Existing Booking",
            "📦 Batch Risk Review",
            "📊 Segment Dashboard",
        ],
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-title">Model Information</div>
            <div class="sidebar-text">
                <strong>Model:</strong> {model_name}<br>
                <strong>Decision threshold:</strong> {selected_threshold:.2f}<br>
                <strong>Features used:</strong> {len(model_features)}<br>
                <strong>Reference rows:</strong> {len(df):,}<br>
                <strong>SHAP explanations:</strong> {"✅ Available" if SHAP_AVAILABLE else "⬜ Install shap"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model Performance Panel ──
    if model_metrics:
        with st.expander("📈 Model Performance", expanded=False):
            rows_html = ""
            metric_display = {
                "roc_auc": ("ROC-AUC", ".4f"),
                "f1":      ("F1 Score", ".4f"),
                "precision": ("Precision", ".4f"),
                "recall":    ("Recall",   ".4f"),
                "accuracy":  ("Accuracy", ".4f"),
                "threshold": ("Threshold", ".2f"),
            }
            for key, (label, fmt) in metric_display.items():
                if key in model_metrics:
                    val = model_metrics[key]
                    try:
                        display_val = f"{float(val):{fmt}}"
                    except Exception:
                        display_val = str(val)
                    rows_html += f'<div class="perf-row"><span class="perf-key">{label}</span><span class="perf-value">{display_val}</span></div>'
            if rows_html:
                st.markdown(
                    f'<div class="content-card" style="margin-top:0;">{rows_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Save model metrics in your bundle to see them here.")
    else:
        with st.expander("📈 Model Performance", expanded=False):
            st.markdown(
                """
                <div class="small-note">
                No metrics found in the model bundle.<br><br>
                In notebook 04, add this when saving:<br>
                <code>bundle['model_metrics'] = {'roc_auc': ..., 'f1': ..., 'precision': ..., 'recall': ...}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Threshold note ──
    st.markdown(
        f"""
        <div class="threshold-badge">
            ⚖️ Decision threshold: {selected_threshold:.2f}
            &nbsp;·&nbsp; bookings ≥ {selected_threshold:.2f} are flagged as likely cancellations
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Disclaimer (always visible) ──
    st.markdown(
        """
        <div class="sidebar-card" style="border-color:rgba(164,81,74,0.30);">
            <div class="sidebar-title" style="font-size:0.95rem;color:#c07a74;">⚠️ Disclaimer</div>
            <div class="sidebar-text">
                This tool provides model-based decision support only.
                Use as a prioritisation aid, not as a guaranteed prediction of guest behaviour.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN APP
# ============================================================

render_hero()


# ============================================================
# MODE 1 — PREDICT NEW BOOKING
# ============================================================

if mode == "🔮 Predict New Booking":
    render_section(
        "Predict New Booking",
        "Enter booking details below. The app validates inputs, builds all model features automatically, "
        "and estimates the cancellation probability with SHAP-derived explanations.",
    )

    with st.form("manual_booking_form"):

        # ── Section 1: Booking Profile ──
        st.markdown('<div class="section-title" style="font-size:1.4rem;">Booking Profile</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            hotel          = st.selectbox("Hotel type", ["City Hotel", "Resort Hotel"])
            market_segment = st.selectbox(
                "Market segment",
                ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"],
            )
            customer_type  = st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"])
            deposit_type   = st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])

        with col2:
            lead_time      = st.number_input("Lead time (days to arrival)", min_value=0, max_value=750, value=50)
            arrival_year   = st.number_input("Arrival year",  min_value=2015, max_value=2035, value=2026)
            arrival_month_number = st.selectbox(
                "Arrival month", list(range(1, 13)), index=7,
                format_func=lambda x: MONTH_NAMES[x],
            )
            arrival_day_of_month = st.number_input("Arrival day of month", min_value=1, max_value=31, value=15)

        with col3:
            stays_in_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=20, value=1)
            stays_in_week_nights    = st.number_input("Week nights",    min_value=0, max_value=50, value=2)
            adr = st.number_input("Average daily rate — ADR (€)", min_value=0.0, max_value=1000.0, value=120.0, step=5.0)
            total_of_special_requests = st.number_input("Number of special requests", min_value=0, max_value=10, value=0)

        # ── Section 2: Guest & Reservation Signals ──
        st.markdown('<div class="section-title" style="font-size:1.4rem;">Guest & Reservation Signals</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)

        with col4:
            adults   = st.number_input("Adults",   min_value=0, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
            babies   = st.number_input("Babies",   min_value=0, max_value=5,  value=0)
            meal = st.selectbox(
                "Meal plan",
                ["BB", "HB", "SC", "FB", "Undefined"],
                format_func=lambda x: {
                    "BB": "BB — Bed & Breakfast",
                    "HB": "HB — Half Board",
                    "SC": "SC — Self Catering",
                    "FB": "FB — Full Board",
                    "Undefined": "Undefined",
                }[x],
            )

        with col5:
            country = st.selectbox(
                "Guest country (ISO-3 code)",
                COUNTRY_CODES,
                index=COUNTRY_CODES.index("PRT") if "PRT" in COUNTRY_CODES else 0,
            )
            distribution_channel = st.selectbox(
                "Distribution channel",
                ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
            )
            is_repeated_guest       = st.selectbox("Repeated guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            previous_cancellations  = st.number_input("Previous cancellations",       min_value=0, max_value=30, value=0)

        with col6:
            previous_bookings_not_canceled = st.number_input("Previous successful stays", min_value=0, max_value=100, value=0)
            reserved_room_type = st.selectbox("Reserved room type", ["A","B","C","D","E","F","G","H","L","P"])
            assigned_room_type = st.selectbox("Assigned room type", ["A","B","C","D","E","F","G","H","L","P"])
            booking_changes    = st.number_input("Booking changes made", min_value=0, max_value=30, value=0)

        # ── Section 3: Operational Details (collapsed by default in the form) ──
        st.markdown('<div class="section-title" style="font-size:1.4rem;">Operational Details</div>', unsafe_allow_html=True)
        col7, col8 = st.columns(2)

        with col7:
            required_car_parking_spaces = st.number_input("Car parking spaces required", min_value=0, max_value=10, value=0)
            days_in_waiting_list        = st.number_input("Days in waiting list",         min_value=0, max_value=400, value=0)

        with col8:
            st.markdown(
                '<div class="small-note">Agent / Company IDs are internal codes used by the model. '
                'Use 0 if not applicable.</div>',
                unsafe_allow_html=True,
            )
            agent   = st.number_input("Agent ID (0 = none)", min_value=0, max_value=999, value=0)
            company = st.number_input("Company ID (0 = none)", min_value=0, max_value=999, value=0)

        submitted = st.form_submit_button("🔮 Predict Cancellation Risk")

    if submitted:
        # Validate
        errors, warnings = validate_booking_inputs(
            int(adults), int(children), int(babies),
            int(stays_in_weekend_nights), int(stays_in_week_nights),
            int(arrival_year), int(arrival_month_number), int(arrival_day_of_month),
            float(adr), int(lead_time),
        )
        for e in errors:
            render_error_banner(e)
        for w in warnings:
            render_warning_banner(w)

        if not errors:
            with st.spinner("Running prediction…"):
                manual_model_df, manual_full_df = build_manual_feature_row(
                    hotel=hotel, lead_time=int(lead_time),
                    arrival_year=int(arrival_year),
                    arrival_month_number=int(arrival_month_number),
                    arrival_day_of_month=int(arrival_day_of_month),
                    stays_in_weekend_nights=int(stays_in_weekend_nights),
                    stays_in_week_nights=int(stays_in_week_nights),
                    adults=int(adults), children=int(children), babies=int(babies),
                    meal=meal, country=country,
                    market_segment=market_segment,
                    distribution_channel=distribution_channel,
                    is_repeated_guest=int(is_repeated_guest),
                    previous_cancellations=int(previous_cancellations),
                    previous_bookings_not_canceled=int(previous_bookings_not_canceled),
                    reserved_room_type=reserved_room_type,
                    assigned_room_type=assigned_room_type,
                    booking_changes=int(booking_changes),
                    deposit_type=deposit_type, agent=int(agent), company=int(company),
                    days_in_waiting_list=int(days_in_waiting_list),
                    customer_type=customer_type, adr=float(adr),
                    required_car_parking_spaces=int(required_car_parking_spaces),
                    total_of_special_requests=int(total_of_special_requests),
                )

                result_df, err = predict_manual_booking(manual_model_df, manual_full_df)

            if err:
                render_error_banner(
                    f"Prediction failed. This usually means your CSV columns don't match what the model "
                    f"expects. Technical detail: {err}"
                )
            else:
                st.session_state["prediction_result"] = (result_df, manual_model_df)

    # Render cached result (persists across rerenders)
    if st.session_state["prediction_result"] is not None:
        result_df, manual_model_df = st.session_state["prediction_result"]
        result_row = result_df.iloc[0]

        probability = float(result_row["predicted_cancellation_probability"])
        risk_label  = result_row["risk_category"]
        decision    = "Likely Cancelled" if int(result_row["predicted_is_cancelled"]) == 1 else "Likely Retained"

        # Borderline warning
        if borderline(probability):
            render_warning_banner(
                f"This prediction ({format_percent(probability)}) is within 5 percentage points of "
                f"the decision threshold ({selected_threshold:.2f}). Treat this booking with extra caution."
            )

        # ── Gauge + key metrics row ──
        col_gauge, col_b, col_c, col_d = st.columns([1.1, 1, 1, 1])

        with col_gauge:
            st.markdown(
                f'<div class="metric-card metric-card-dark" style="align-items:center;padding-bottom:0.3rem;">'
                f'<div class="metric-label" style="text-align:center;">Cancellation Probability</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(plotly_donut(probability, risk_label), use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div class="metric-caption" style="text-align:center;padding-bottom:0.6rem;">'
                f'Threshold: {selected_threshold:.2f}</div></div>',
                unsafe_allow_html=True,
            )

        with col_b:
            render_risk_card(risk_label, probability)

        with col_c:
            render_metric_card(
                "Model Decision", decision,
                "Threshold-based operational classification.",
                "sage" if decision == "Likely Retained" else "blush",
            )

        with col_d:
            render_metric_card(
                "Revenue at Risk",
                format_currency(float(result_row["estimated_revenue_at_risk"])),
                "Probability-weighted revenue exposure.",
                "light",
            )

        # ── Drivers + Actions ──
        drivers, is_shap = get_risk_drivers(result_row, manual_model_df)
        driver_title = "Why the model is concerned (SHAP-derived)" if is_shap else "Why the booking may be risky"

        col_left, col_right = st.columns(2)
        with col_left:
            render_list_card(driver_title, drivers, light=False)
            if is_shap and SHAP_AVAILABLE:
                shap_pairs = get_shap_drivers(manual_model_df)
                if shap_pairs:
                    st.plotly_chart(plotly_shap_bar(shap_pairs), use_container_width=True, config={"displayModeBar": False})

        with col_right:
            render_list_card("Recommended manager actions", get_recommended_actions(risk_label, result_row), light=True)

        render_text_card("Manager Summary", create_manager_summary(result_row), light=False)

        # ── What-If Sensitivity ──
        with st.expander("🔄 What-If Analysis — adjust lead time & deposit to see risk change", expanded=False):
            wif_col1, wif_col2 = st.columns(2)
            with wif_col1:
                wif_lead  = st.slider("What-If lead time (days)", 0, 365, int(safe_value(result_row, "lead_time", 50)))
            with wif_col2:
                wif_dep   = st.selectbox("What-If deposit type", ["No Deposit", "Non Refund", "Refundable"], key="wif_dep")

            if st.button("Run What-If Prediction", key="run_whatif"):
                wif_model_df = manual_model_df.copy()
                if "lead_time" in wif_model_df.columns:
                    wif_model_df["lead_time"] = wif_lead
                if "lead_time_bucket" in wif_model_df.columns:
                    wif_model_df["lead_time_bucket"] = lead_time_bucket(wif_lead)
                if "is_last_minute_booking" in wif_model_df.columns:
                    wif_model_df["is_last_minute_booking"] = 1 if wif_lead <= 7 else 0
                if "is_long_lead_booking" in wif_model_df.columns:
                    wif_model_df["is_long_lead_booking"] = 1 if wif_lead >= 90 else 0
                if "deposit_type" in wif_model_df.columns:
                    wif_model_df["deposit_type"] = wif_dep

                wif_result, wif_err = predict_manual_booking(wif_model_df, manual_full_df.copy())
                if wif_err:
                    render_error_banner(f"What-If prediction failed: {wif_err}")
                else:
                    wif_prob  = float(wif_result.iloc[0]["predicted_cancellation_probability"])
                    wif_label = classify_risk(wif_prob)
                    delta     = wif_prob - probability
                    direction = "▲ higher" if delta > 0 else "▼ lower"
                    col_wif1, col_wif2 = st.columns(2)
                    with col_wif1:
                        st.plotly_chart(plotly_donut(wif_prob, wif_label), use_container_width=True, config={"displayModeBar": False})
                    with col_wif2:
                        render_metric_card(
                            "What-If Risk",
                            format_percent(wif_prob),
                            f"{direction} than original ({format_percent(abs(delta))} change) — {wif_label}",
                            "blush" if wif_prob >= selected_threshold else "sage",
                        )

        # ── Generated Booking Signals ──
        with st.expander("📋 Generated Booking Signals", expanded=False):
            signal_cols = [
                "hotel","market_segment","customer_type","lead_time","lead_time_bucket",
                "arrival_season","total_nights","stay_length_category","adr","adr_bucket",
                "deposit_type","estimated_booking_value","room_type_changed",
                "previous_cancellation_rate","has_special_requests","is_weekend_stay",
            ]
            avail = [c for c in signal_cols if c in result_df.columns]
            st.dataframe(result_df[avail].T.rename(columns={0: "Value"}), use_container_width=True)


# ============================================================
# MODE 2 — EXPLAIN EXISTING BOOKING
# ============================================================

elif mode == "🔍 Explain Existing Booking":
    render_section(
        "Explain Existing Booking",
        "Filter the dataset, select a booking, and get a manager-style risk explanation.",
    )

    # Filter controls
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        filter_hotel = st.multiselect(
            "Filter by hotel type",
            options=df["hotel"].dropna().unique().tolist() if "hotel" in df.columns else [],
            default=[],
        )
    with fcol2:
        filter_segment = st.multiselect(
            "Filter by market segment",
            options=df["market_segment"].dropna().unique().tolist() if "market_segment" in df.columns else [],
            default=[],
        )
    with fcol3:
        filter_risk_only = st.checkbox("Show only high-risk samples", value=False)

    # Build filtered sample
    sample_df = df.copy()
    if filter_hotel:
        sample_df = sample_df[sample_df["hotel"].isin(filter_hotel)]
    if filter_segment:
        sample_df = sample_df[sample_df["market_segment"].isin(filter_segment)]

    sample_df = sample_df.sample(min(500, len(sample_df)), random_state=42).reset_index(drop=True)

    with st.spinner("Scoring sample bookings…"):
        scored_sample, err = predict_bookings(sample_df)

    if err:
        render_error_banner(f"Scoring failed: {err}")
        st.stop()

    if filter_risk_only:
        scored_sample = scored_sample[scored_sample["risk_category"] == "High Risk"].reset_index(drop=True)

    if len(scored_sample) == 0:
        st.info("No bookings match the current filters. Try broadening the selection.")
        st.stop()

    # Meaningful booking label
    def booking_label(row: pd.Series) -> str:
        hotel_   = safe_value(row, "hotel", "?")
        seg_     = safe_value(row, "market_segment", "?")
        lt_      = safe_value(row, "lead_time", "?")
        adr_     = safe_value(row, "adr", "?")
        try:
            lt_str  = f"{int(lt_)}d"
            adr_str = f"€{float(adr_):.0f}/n"
        except Exception:
            lt_str, adr_str = str(lt_), str(adr_)
        return f"{hotel_} · {seg_} · {lt_str} lead · {adr_str}"

    label_map = {i: booking_label(scored_sample.loc[i]) for i in scored_sample.index}

    selected_index = st.selectbox(
        "Select a booking to explain",
        options=scored_sample.index.tolist(),
        format_func=lambda x: label_map[x],
    )

    selected_row = scored_sample.loc[selected_index]
    probability  = float(selected_row["predicted_cancellation_probability"])
    risk_label   = selected_row["risk_category"]
    decision     = "Likely Cancelled" if int(selected_row["predicted_is_cancelled"]) == 1 else "Likely Retained"

    if borderline(probability):
        render_warning_banner(
            f"This prediction ({format_percent(probability)}) is close to the decision threshold. "
            "Treat this booking with additional caution."
        )

    col_g, col_b, col_c = st.columns([1.1, 1, 1])
    with col_g:
        st.markdown(
            '<div class="metric-card metric-card-dark" style="align-items:center;padding-bottom:0.3rem;">'
            '<div class="metric-label" style="text-align:center;">Cancellation Probability</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(plotly_donut(probability, risk_label), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="metric-caption" style="text-align:center;padding-bottom:0.6rem;"></div></div>', unsafe_allow_html=True)
    with col_b:
        render_risk_card(risk_label, probability)
    with col_c:
        render_metric_card(
            "Model Decision", decision, "Threshold-based classification.",
            "sage" if decision == "Likely Retained" else "blush",
        )

    display_cols = [c for c in [
        "hotel","market_segment","customer_type","lead_time","adr",
        "total_nights","total_of_special_requests","deposit_type","estimated_booking_value",
    ] if c in scored_sample.columns]

    st.markdown('<div class="section-title" style="font-size:1.4rem;">Booking Details</div>', unsafe_allow_html=True)
    st.dataframe(selected_row[display_cols].to_frame(name="Value"), use_container_width=True)

    drivers, is_shap = get_risk_drivers(selected_row)
    col_a, col_b2 = st.columns(2)
    with col_a:
        render_list_card(
            "Why the model may be concerned" + (" (SHAP)" if is_shap else ""),
            drivers, light=False,
        )
    with col_b2:
        render_list_card("Recommended manager actions", get_recommended_actions(risk_label, selected_row), light=True)

    render_text_card("Manager Summary", create_manager_summary(selected_row), light=False)


# ============================================================
# MODE 3 — BATCH RISK REVIEW
# ============================================================

elif mode == "📦 Batch Risk Review":
    render_section(
        "Batch Booking Risk Review",
        "Upload a CSV file or use the available dataset. The app validates structure, "
        "scores all bookings, and surfaces the highest-risk reservations.",
    )

    uploaded_file = st.file_uploader("Upload booking CSV (optional)", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
        # Check for critical missing columns
        missing_cols = [c for c in model_features if c not in batch_df.columns]
        if missing_cols:
            render_warning_banner(
                f"Your CSV is missing {len(missing_cols)} model features. "
                f"They will be filled with dataset defaults, which may reduce accuracy. "
                f"First missing: {', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}."
            )
    else:
        batch_df = df.copy()
        source_label = "Feature-engineered dataset"

    # Only re-score if source changed
    if (
        st.session_state["scored_batch"] is None
        or st.session_state["batch_source_label"] != source_label
    ):
        with st.spinner("Scoring all bookings…"):
            scored_df, err = predict_bookings(batch_df)

        if err:
            render_error_banner(
                f"Batch scoring failed. Check that your CSV columns are compatible with the model. "
                f"Technical detail: {err}"
            )
            st.stop()

        st.session_state["scored_batch"]        = scored_df
        st.session_state["batch_source_label"]  = source_label
    else:
        scored_df = st.session_state["scored_batch"]

    total_bookings     = len(scored_df)
    avg_risk           = scored_df["predicted_cancellation_probability"].mean()
    high_risk_count    = (scored_df["risk_category"] == "High Risk").sum()
    at_risk_count      = int(scored_df["predicted_is_cancelled"].sum())
    total_rev_at_risk  = scored_df["estimated_revenue_at_risk"].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Dataset Source", source_label[:22] + "…" if len(source_label) > 22 else source_label,
                           "Current dataset being analysed.", "light")
    with col2:
        render_metric_card("Bookings Reviewed", f"{total_bookings:,}", "Total bookings scored.", "sage")
    with col3:
        render_metric_card("Predicted At-Risk", f"{at_risk_count:,}",
                           f"{format_percent(at_risk_count/total_bookings)} of total bookings.", "blush")
    with col4:
        render_metric_card("Revenue at Risk", format_currency(total_rev_at_risk),
                           "Probability-weighted exposure.", "light")

    render_text_card(
        "Batch Summary",
        f"The batch contains <strong>{total_bookings:,}</strong> bookings with an average predicted "
        f"cancellation probability of <strong>{format_percent(avg_risk)}</strong>. "
        f"The model flags <strong>{at_risk_count:,}</strong> bookings as likely cancellations, "
        f"including <strong>{int(high_risk_count):,}</strong> high-risk reservations. "
        f"Estimated revenue at risk: <strong>{format_currency(total_rev_at_risk)}</strong>.",
        light=False,
    )

    # Monthly trend chart
    trend_fig = plotly_monthly_trend(scored_df)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

    top_n = st.slider("Number of high-risk bookings to display", 5, 50, 15)

    preview_cols = [c for c in [
        "hotel","market_segment","customer_type","lead_time","adr",
        "estimated_booking_value","predicted_cancellation_probability",
        "risk_category","predicted_is_cancelled","estimated_revenue_at_risk",
    ] if c in scored_df.columns]

    top_risk_df = scored_df.sort_values(
        ["predicted_cancellation_probability", "estimated_revenue_at_risk"],
        ascending=[False, False],
    ).head(top_n)

    st.markdown('<div class="section-title" style="font-size:1.4rem;">Top Risk Bookings</div>', unsafe_allow_html=True)
    st.dataframe(top_risk_df[preview_cols], use_container_width=True)

    csv_output = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download scored booking file",
        data=csv_output,
        file_name="scored_hotel_bookings.csv",
        mime="text/csv",
    )


# ============================================================
# MODE 4 — SEGMENT DASHBOARD
# ============================================================

elif mode == "📊 Segment Dashboard":
    render_section(
        "Segment Dashboard",
        "Compare cancellation exposure across market segments, customer types, "
        "hotel categories, and arrival months.",
    )

    # Cache scored full dataset
    if st.session_state["scored_segment"] is None:
        with st.spinner("Scoring dataset for segment analysis…"):
            scored_seg, err = predict_bookings(df)
        if err:
            render_error_banner(f"Scoring failed: {err}")
            st.stop()
        st.session_state["scored_segment"] = scored_seg
    else:
        scored_seg = st.session_state["scored_segment"]

    segment_col = st.selectbox(
        "Choose segmentation dimension",
        ["market_segment", "customer_type", "hotel"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    if segment_col not in scored_seg.columns:
        st.warning(f"`{segment_col}` is not available in the dataset.")
        st.stop()

    segment_summary = (
        scored_seg.groupby(segment_col, dropna=False)
        .agg(
            booking_count=(segment_col, "count"),
            avg_predicted_cancellation_probability=("predicted_cancellation_probability", "mean"),
            predicted_cancellation_rate=("predicted_is_cancelled", "mean"),
            total_estimated_booking_value=("estimated_booking_value", "sum"),
            estimated_revenue_at_risk=("estimated_revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values("estimated_revenue_at_risk", ascending=False)
    )

    top_segment       = segment_summary.iloc[0]
    total_rev_at_risk = segment_summary["estimated_revenue_at_risk"].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            "Highest Revenue Exposure", str(top_segment[segment_col]),
            "Segment with the largest probability-weighted revenue at risk.", "blush",
        )
    with col2:
        render_metric_card(
            "Top Segment Avg Risk",
            format_percent(float(top_segment["avg_predicted_cancellation_probability"])),
            "Average predicted cancellation probability for top segment.", "light",
        )
    with col3:
        render_metric_card(
            "Total Revenue at Risk", format_currency(float(total_rev_at_risk)),
            "Estimated exposure across all segments.", "sage",
        )

    render_text_card(
        "Segment-Level Readout",
        f"Using <strong>{segment_col.replace('_',' ').title()}</strong> as the segmentation lens, "
        f"<strong>{top_segment[segment_col]}</strong> shows the highest revenue exposure. "
        f"This segment has <strong>{int(top_segment['booking_count']):,}</strong> bookings and "
        f"an estimated <strong>{format_currency(float(top_segment['estimated_revenue_at_risk']))}</strong> "
        f"probability-weighted revenue at risk.",
        light=False,
    )

    # Plotly bar chart (on-theme)
    chart_df = segment_summary.sort_values("estimated_revenue_at_risk", ascending=True)
    fig = plotly_bar_h(
        chart_df, x_col="estimated_revenue_at_risk", y_col=segment_col,
        title=f"Estimated Revenue at Risk by {segment_col.replace('_',' ').title()}",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Monthly trend for selected segment
    trend_fig = plotly_monthly_trend(scored_seg)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

    # Formatted summary table
    display_seg = segment_summary.copy()
    display_seg["avg_predicted_cancellation_probability"] = (
        display_seg["avg_predicted_cancellation_probability"] * 100
    ).round(1).astype(str) + "%"
    display_seg["predicted_cancellation_rate"] = (
        display_seg["predicted_cancellation_rate"] * 100
    ).round(1).astype(str) + "%"
    display_seg["total_estimated_booking_value"] = (
        display_seg["total_estimated_booking_value"].round(0).astype(int).apply(lambda x: f"€{x:,}")
    )
    display_seg["estimated_revenue_at_risk"] = (
        display_seg["estimated_revenue_at_risk"].round(0).astype(int).apply(lambda x: f"€{x:,}")
    )

    st.markdown('<div class="section-title" style="font-size:1.4rem;">Segment Risk Table</div>', unsafe_allow_html=True)
    st.dataframe(display_seg, use_container_width=True, hide_index=True)