# app/streamlit_app.py

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


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

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "xgboost_cancellation_model_bundle.joblib"
DATA_PATH = PROJECT_DIR / "data" / "hotel_bookings_feature_engineered.csv"


# ============================================================
# CUSTOM CSS — SAGE / BLUSH / CREAM RESORT THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg-main: #111719;
        --bg-panel: #182023;
        --bg-panel-soft: #20292c;
        --sidebar-bg: #252732;
        --sage: #7f9a8a;
        --sage-light: #dfe8e1;
        --sage-muted: #b8c8bd;
        --cream: #f3efe8;
        --sand: #ded6ca;
        --blush: #ead2c8;
        --rose: #a4514a;
        --rose-deep: #873732;
        --ink: #25332c;
        --muted-text: #cbd3cd;
    }

    .stApp {
        background: linear-gradient(180deg, #101619 0%, #151c1f 100%);
        color: var(--cream);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #252732 0%, #20232c 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f3efe8 !important;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f3efe8;
    }

    .hero-card {
        background: linear-gradient(135deg, #5e786b 0%, #8fa798 100%);
        border-radius: 30px;
        padding: 2.2rem 2.2rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
    }

    .hero-kicker {
        font-size: 0.82rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 700;
        color: #fff7ef;
        opacity: 0.9;
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
        max-width: 980px;
        font-size: 1.08rem;
        line-height: 1.65;
        color: #fff8f0;
        opacity: 0.96;
    }

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

    .metric-card-light {
        background: #f3efe8;
        color: #25332c;
    }

    .metric-card-sage {
        background: #dfe8e1;
        color: #25332c;
    }

    .metric-card-blush {
        background: #ead2c8;
        color: #25332c;
    }

    .metric-card-dark {
        background: linear-gradient(180deg, #182023 0%, #20292c 100%);
        color: #f3efe8;
    }

    .metric-label {
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        opacity: 0.86;
    }

    .metric-value {
        font-size: 2.1rem;
        font-weight: 850;
        line-height: 1.05;
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }

    .metric-caption {
        font-size: 0.92rem;
        line-height: 1.45;
        opacity: 0.86;
    }

    .risk-pill {
        display: inline-block;
        padding: 0.52rem 0.9rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
    }

    .risk-high {
        background: rgba(164,81,74,0.18);
        color: #7a2f2a;
        border: 1px solid rgba(164,81,74,0.25);
    }

    .risk-medium {
        background: rgba(180,143,95,0.20);
        color: #6f5028;
        border: 1px solid rgba(180,143,95,0.25);
    }

    .risk-low {
        background: rgba(95,122,109,0.20);
        color: #2d5241;
        border: 1px solid rgba(95,122,109,0.25);
    }

    .content-card {
        background: linear-gradient(180deg, #182023 0%, #20292c 100%);
        border-radius: 24px;
        padding: 1.35rem 1.45rem;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 12px 32px rgba(0,0,0,0.20);
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        color: #f3efe8;
    }

    .content-card-light {
        background: #f3efe8;
        border-radius: 24px;
        padding: 1.35rem 1.45rem;
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

    .content-text {
        font-size: 0.98rem;
        line-height: 1.65;
    }

    .content-card ul,
    .content-card-light ul {
        margin-top: 0.4rem;
        padding-left: 1.2rem;
    }

    .content-card li,
    .content-card-light li {
        margin-bottom: 0.58rem;
        line-height: 1.55;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.1rem;
        margin-bottom: 1.1rem;
    }

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }

    .sidebar-text {
        font-size: 0.95rem;
        line-height: 1.6;
        opacity: 0.92;
    }

    div[data-baseweb="select"] > div {
        background-color: #20292c !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 16px !important;
        color: #f3efe8 !important;
    }

    .stNumberInput input,
    .stTextInput input {
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

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px !important;
        border: none !important;
        font-weight: 800 !important;
        padding: 0.7rem 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7f9a8a 0%, #5f7a6d 100%) !important;
        color: #fff8f0 !important;
    }

    .stDownloadButton > button {
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
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model_bundle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        st.error(f"Model bundle not found at: {path}")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Dataset not found at: {path}")
        st.stop()
    return pd.read_csv(path)


model_bundle = load_model_bundle(MODEL_PATH)
df = load_data(DATA_PATH)


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
    model_bundle,
    ["model", "best_model", "xgb_model", "classifier"],
)

preprocessor = get_bundle_value(
    model_bundle,
    ["preprocessor", "transformer", "feature_pipeline"],
)

selected_threshold = float(
    get_bundle_value(
        model_bundle,
        ["selected_threshold", "threshold", "best_threshold", "decision_threshold"],
        0.55,
    )
)

model_features = get_bundle_value(
    model_bundle,
    ["model_features", "feature_names", "selected_features", "input_features"],
    [],
)

target_col = get_bundle_value(
    model_bundle,
    ["target_column", "target_col"],
    "is_canceled",
)

if model is None:
    st.error("Model object not found inside the saved bundle.")
    st.stop()

if preprocessor is None:
    st.error("Preprocessor object not found inside the saved bundle.")
    st.stop()

if not model_features:
    st.error("Model feature list not found inside the saved bundle.")
    st.stop()


# ============================================================
# DATA HELPERS
# ============================================================

def get_default_value(column: str) -> Any:
    if column not in df.columns:
        return 0

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        value = series.median()
        if pd.isna(value):
            return 0
        return value

    mode_values = series.dropna().mode()
    if len(mode_values) > 0:
        return mode_values.iloc[0]

    return "Unknown"


def enrich_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df.copy()

    if "total_nights" not in output_df.columns:
        weekend = output_df["stays_in_weekend_nights"] if "stays_in_weekend_nights" in output_df.columns else 0
        week = output_df["stays_in_week_nights"] if "stays_in_week_nights" in output_df.columns else 0
        output_df["total_nights"] = weekend + week

    if "estimated_booking_value" not in output_df.columns:
        if "adr" in output_df.columns and "total_nights" in output_df.columns:
            output_df["estimated_booking_value"] = output_df["adr"].fillna(0) * output_df["total_nights"].fillna(0)
        else:
            output_df["estimated_booking_value"] = 0.0

    return output_df


df = enrich_dataframe(df)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_currency(value: float) -> str:
    return f"€{value:,.2f}"


def safe_value(row: pd.Series, col: str, default: Any = "Unknown") -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return value


def classify_risk(probability: float) -> str:
    if probability >= max(0.75, selected_threshold + 0.12):
        return "High Risk"
    elif probability >= selected_threshold:
        return "Moderate Risk"
    else:
        return "Low Risk"


def risk_pill_class(risk_label: str) -> str:
    if risk_label == "High Risk":
        return "risk-high"
    elif risk_label == "Moderate Risk":
        return "risk-medium"
    return "risk-low"


# ============================================================
# FEATURE ENGINEERING FOR MANUAL BOOKING INPUT
# ============================================================

def arrival_season_from_month(month_number: int) -> str:
    if month_number in [12, 1, 2]:
        return "Winter"
    if month_number in [3, 4, 5]:
        return "Spring"
    if month_number in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def month_name_from_number(month_number: int) -> str:
    month_map = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    return month_map.get(month_number, "Unknown")


def lead_time_bucket_from_days(lead_time: int) -> str:
    if lead_time <= 7:
        return "Last minute"
    if lead_time <= 30:
        return "Short lead"
    if lead_time <= 90:
        return "Medium lead"
    if lead_time <= 180:
        return "Long lead"
    return "Very long lead"


def adr_bucket_from_value(adr: float) -> str:
    if adr <= 0:
        return "Zero ADR"
    if adr < 75:
        return "Low ADR"
    if adr < 150:
        return "Medium ADR"
    return "High ADR"


def stay_length_category_from_nights(total_nights: int) -> str:
    if total_nights == 0:
        return "Zero-night stay"
    if total_nights <= 2:
        return "Short stay"
    if total_nights <= 5:
        return "Medium stay"
    if total_nights <= 10:
        return "Long stay"
    return "Extended stay"


def build_manual_feature_row(
    hotel: str,
    lead_time: int,
    arrival_year: int,
    arrival_month_number: int,
    arrival_day_of_month: int,
    stays_in_weekend_nights: int,
    stays_in_week_nights: int,
    adults: int,
    children: int,
    babies: int,
    meal: str,
    country: str,
    market_segment: str,
    distribution_channel: str,
    is_repeated_guest: int,
    previous_cancellations: int,
    previous_bookings_not_canceled: int,
    reserved_room_type: str,
    assigned_room_type: str,
    booking_changes: int,
    deposit_type: str,
    agent: int,
    company: int,
    days_in_waiting_list: int,
    customer_type: str,
    adr: float,
    required_car_parking_spaces: int,
    total_of_special_requests: int,
) -> pd.DataFrame:

    total_nights = stays_in_weekend_nights + stays_in_week_nights
    estimated_booking_value = adr * total_nights
    previous_total_bookings = previous_cancellations + previous_bookings_not_canceled

    if previous_total_bookings > 0:
        previous_cancellation_rate = previous_cancellations / previous_total_bookings
    else:
        previous_cancellation_rate = 0.0

    arrival_date = pd.Timestamp(
        year=arrival_year,
        month=arrival_month_number,
        day=min(arrival_day_of_month, 28),
    )

    row = {
        # original / raw booking features
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": month_name_from_number(arrival_month_number),
        "arrival_date_week_number": int(arrival_date.isocalendar().week),
        "arrival_date_day_of_month": arrival_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,

        # engineered temporal features
        "lead_time_bucket": lead_time_bucket_from_days(lead_time),
        "is_last_minute_booking": 1 if lead_time <= 7 else 0,
        "is_long_lead_booking": 1 if lead_time >= 90 else 0,
        "arrival_month_number": arrival_month_number,
        "arrival_season": arrival_season_from_month(arrival_month_number),
        "is_peak_season": 1 if arrival_month_number in [6, 7, 8, 9] else 0,

        # engineered stay-duration features
        "total_nights": total_nights,
        "is_zero_night_stay": 1 if total_nights == 0 else 0,
        "is_weekend_stay": 1 if stays_in_weekend_nights > 0 else 0,
        "stay_length_category": stay_length_category_from_nights(total_nights),

        # engineered behavioural signals
        "previous_total_bookings": previous_total_bookings,
        "previous_cancellation_rate": previous_cancellation_rate,
        "has_previous_cancellation": 1 if previous_cancellations > 0 else 0,
        "has_previous_successful_booking": 1 if previous_bookings_not_canceled > 0 else 0,
        "has_booking_changes": 1 if booking_changes > 0 else 0,
        "has_special_requests": 1 if total_of_special_requests > 0 else 0,
        "is_waitlisted": 1 if days_in_waiting_list > 0 else 0,
        "repeat_guest_with_deposit": 1 if is_repeated_guest == 1 and deposit_type != "No Deposit" else 0,
        "repeat_guest_non_refund": 1 if is_repeated_guest == 1 and deposit_type == "Non Refund" else 0,

        # engineered revenue features
        "estimated_booking_value": estimated_booking_value,
        "adr_bucket": adr_bucket_from_value(adr),
        "has_invalid_adr": 1 if adr < 0 else 0,
        "is_zero_adr": 1 if adr == 0 else 0,
        "is_high_adr": 1 if adr >= 150 else 0,
        "is_high_value_booking": 1 if estimated_booking_value >= df["estimated_booking_value"].quantile(0.75) else 0,
        "room_type_changed": 1 if reserved_room_type != assigned_room_type else 0,
    }

    manual_df = pd.DataFrame([row])

    # add any missing model features using defaults from the training dataset
    for feature in model_features:
        if feature not in manual_df.columns:
            manual_df[feature] = get_default_value(feature)

    # align dtypes as much as possible to the training dataframe
    for feature in model_features:
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                manual_df[feature] = pd.to_numeric(manual_df[feature], errors="coerce").fillna(get_default_value(feature))
            else:
                manual_df[feature] = manual_df[feature].astype(str)

    return manual_df[model_features].copy(), manual_df.copy()


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_bookings(input_df: pd.DataFrame) -> pd.DataFrame:
    working_df = enrich_dataframe(input_df.copy())

    for feature in model_features:
        if feature not in working_df.columns:
            working_df[feature] = get_default_value(feature)

    X_input = working_df[model_features].copy()

    for feature in model_features:
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                X_input[feature] = pd.to_numeric(X_input[feature], errors="coerce").fillna(get_default_value(feature))
            else:
                X_input[feature] = X_input[feature].astype(str)

    X_processed = preprocessor.transform(X_input)

    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= selected_threshold).astype(int)

    scored_df = working_df.copy()
    scored_df["predicted_cancellation_probability"] = probabilities
    scored_df["predicted_is_cancelled"] = predictions
    scored_df["risk_category"] = scored_df["predicted_cancellation_probability"].apply(classify_risk)
    scored_df["estimated_revenue_at_risk"] = (
        scored_df["estimated_booking_value"].fillna(0)
        * scored_df["predicted_cancellation_probability"]
    )

    return scored_df


def predict_manual_booking(manual_model_df: pd.DataFrame, manual_full_df: pd.DataFrame) -> pd.DataFrame:
    X_processed = preprocessor.transform(manual_model_df)

    probability = model.predict_proba(X_processed)[:, 1][0]
    prediction = int(probability >= selected_threshold)

    result_df = manual_full_df.copy()
    result_df["predicted_cancellation_probability"] = probability
    result_df["predicted_is_cancelled"] = prediction
    result_df["risk_category"] = classify_risk(probability)
    result_df["estimated_revenue_at_risk"] = (
        result_df["estimated_booking_value"].fillna(0)
        * result_df["predicted_cancellation_probability"]
    )

    return result_df


# ============================================================
# BUSINESS EXPLANATION HELPERS
# ============================================================

def get_risk_drivers(row: pd.Series) -> List[str]:
    drivers = []

    lead_time = safe_value(row, "lead_time", 0)
    market_segment = safe_value(row, "market_segment", "Unknown")
    customer_type = safe_value(row, "customer_type", "Unknown")
    deposit_type = safe_value(row, "deposit_type", "Unknown")
    special_requests = safe_value(row, "total_of_special_requests", 0)
    previous_cancellations = safe_value(row, "previous_cancellations", 0)
    previous_cancellation_rate = safe_value(row, "previous_cancellation_rate", 0)
    room_type_changed = safe_value(row, "room_type_changed", 0)
    adr = safe_value(row, "adr", 0)
    total_nights = safe_value(row, "total_nights", 0)

    try:
        if float(lead_time) >= 60:
            drivers.append("Long lead time creates more opportunity for guest plans to change before arrival.")
        elif float(lead_time) <= 7:
            drivers.append("Very short lead time may behave differently from normal booking patterns.")
    except Exception:
        pass

    if market_segment in ["Online TA", "Groups", "Offline TA/TO"]:
        drivers.append(f"The booking comes from the {market_segment} segment, which can show higher cancellation sensitivity.")

    if customer_type in ["Transient", "Transient-Party"]:
        drivers.append(f"The customer type is {customer_type}, usually less contract-bound than group or contract demand.")

    if deposit_type in ["No Deposit", "Refundable"]:
        drivers.append("The deposit profile may allow easier cancellation compared with stronger payment commitment.")

    try:
        if float(special_requests) == 0:
            drivers.append("No special requests may indicate lower guest engagement with the stay.")
    except Exception:
        pass

    try:
        if float(previous_cancellations) > 0 or float(previous_cancellation_rate) > 0:
            drivers.append("Previous cancellation behaviour is present in the booking history.")
    except Exception:
        pass

    try:
        if float(room_type_changed) == 1:
            drivers.append("The reserved and assigned room types differ, which is a meaningful model signal.")
    except Exception:
        pass

    try:
        if float(adr) >= 150:
            drivers.append("Higher ADR can increase price sensitivity when combined with other risk signals.")
    except Exception:
        pass

    try:
        if float(total_nights) >= 7:
            drivers.append("Longer stays carry larger value exposure and may require closer monitoring.")
    except Exception:
        pass

    if not drivers:
        drivers.append("No single obvious business-rule driver dominates; the risk comes from the combined model pattern.")

    return drivers[:6]


def get_recommended_actions(risk_label: str, row: pd.Series) -> List[str]:
    actions = []

    if risk_label == "High Risk":
        actions.extend(
            [
                "Send a personalised reconfirmation message to the guest.",
                "Review payment, deposit, and cancellation-policy exposure.",
                "Flag the reservation for closer monitoring by the revenue or reservations team.",
            ]
        )
    elif risk_label == "Moderate Risk":
        actions.extend(
            [
                "Send a softer pre-arrival reminder or engagement message.",
                "Monitor the booking closer to arrival before making inventory decisions.",
                "Consider a light upsell or add-on offer to increase guest commitment.",
            ]
        )
    else:
        actions.extend(
            [
                "Keep the booking under normal monitoring.",
                "No urgent intervention is required based on the current model signal.",
                "Use standard pre-arrival communication.",
            ]
        )

    market_segment = safe_value(row, "market_segment", "Unknown")
    lead_time = safe_value(row, "lead_time", 0)

    if market_segment == "Online TA":
        actions.append("For Online TA bookings, use timely guest communication to reduce passive cancellations.")

    try:
        if float(lead_time) >= 60:
            actions.append("Because this is a longer lead-time booking, schedule a reminder closer to arrival.")
    except Exception:
        pass

    return actions[:5]


def create_manager_summary(row: pd.Series) -> str:
    hotel = safe_value(row, "hotel", "the property")
    market_segment = safe_value(row, "market_segment", "Unknown")
    customer_type = safe_value(row, "customer_type", "Unknown")
    probability = float(row["predicted_cancellation_probability"])
    risk_label = row["risk_category"]
    booking_value = float(safe_value(row, "estimated_booking_value", 0))
    revenue_at_risk = float(safe_value(row, "estimated_revenue_at_risk", 0))

    if risk_label == "High Risk":
        action_phrase = "should be reviewed immediately as part of the revenue-protection workflow"
    elif risk_label == "Moderate Risk":
        action_phrase = "should be monitored and lightly engaged before arrival"
    else:
        action_phrase = "does not require urgent intervention"

    return (
        f"This booking for <strong>{hotel}</strong> is classified as <strong>{risk_label}</strong>, "
        f"with an estimated cancellation probability of <strong>{format_percent(probability)}</strong>. "
        f"It belongs to the <strong>{market_segment}</strong> segment and the "
        f"<strong>{customer_type}</strong> customer type. The estimated booking value is "
        f"<strong>{format_currency(booking_value)}</strong>, with approximately "
        f"<strong>{format_currency(revenue_at_risk)}</strong> of probability-weighted revenue exposure. "
        f"Based on this profile, the booking {action_phrase}."
    )


# ============================================================
# UI COMPONENTS
# ============================================================

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Revenue Protection • Resort Operations • Machine Learning</div>
            <div class="hero-title">Hotel Cancellation Risk Assistant</div>
            <div class="hero-subtitle">
                Enter booking details, estimate cancellation risk, understand the likely drivers,
                and support hotel revenue decisions with a structured machine-learning assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, caption: str, card_type: str = "light") -> None:
    if card_type == "sage":
        css_class = "metric-card-sage"
    elif card_type == "blush":
        css_class = "metric-card-blush"
    elif card_type == "dark":
        css_class = "metric-card-dark"
    else:
        css_class = "metric-card-light"

    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(risk_label: str, probability: float) -> None:
    pill_class = risk_pill_class(risk_label)

    if risk_label == "High Risk":
        css_class = "metric-card-blush"
        caption = "Immediate manager attention recommended."
    elif risk_label == "Moderate Risk":
        css_class = "metric-card-sage"
        caption = "Monitor and engage before arrival."
    else:
        css_class = "metric-card-light"
        caption = "Standard monitoring is sufficient."

    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-label">Risk Assessment</div>
            <div class="metric-value">{format_percent(probability)}</div>
            <div><span class="risk-pill {pill_class}">{risk_label}</span></div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list_card(title: str, items: List[str], light: bool = False) -> None:
    css_class = "content-card-light" if light else "content-card"
    list_html = "".join([f"<li>{item}</li>" for item in items])

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="content-title">{title}</div>
            <ul>{list_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_card(title: str, text: str, light: bool = False) -> None:
    css_class = "content-card-light" if light else "content-card"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="content-title">{title}</div>
            <div class="content-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            "Predict New Booking",
            "Explain Existing Booking",
            "Batch Booking Risk Review",
            "Segment Dashboard",
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
                <strong>Reference rows:</strong> {len(df):,}
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
# MODE 1: PREDICT NEW BOOKING
# ============================================================

if mode == "Predict New Booking":
    render_section(
        "Predict New Booking",
        "Enter manager-friendly booking details. The app automatically creates the engineered model features and estimates cancellation risk.",
    )

    with st.form("manual_booking_form"):
        st.markdown('<div class="section-title" style="font-size:1.45rem;">Booking Profile</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            hotel = st.selectbox("Hotel type", ["City Hotel", "Resort Hotel"])
            market_segment = st.selectbox(
                "Market segment",
                ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"],
            )
            customer_type = st.selectbox(
                "Customer type",
                ["Transient", "Transient-Party", "Contract", "Group"],
            )
            deposit_type = st.selectbox(
                "Deposit type",
                ["No Deposit", "Non Refund", "Refundable"],
            )

        with col2:
            lead_time = st.number_input("Booking-to-arrival lead time in days", min_value=0, max_value=750, value=50, step=1)
            arrival_year = st.number_input("Arrival year", min_value=2015, max_value=2035, value=2026, step=1)
            arrival_month_number = st.selectbox(
                "Arrival month",
                list(range(1, 13)),
                index=7,
                format_func=lambda x: month_name_from_number(x),
            )
            arrival_day_of_month = st.number_input("Arrival day of month", min_value=1, max_value=31, value=15, step=1)

        with col3:
            stays_in_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=20, value=1, step=1)
            stays_in_week_nights = st.number_input("Week nights", min_value=0, max_value=50, value=2, step=1)
            adr = st.number_input("ADR / average daily rate (€)", min_value=0.0, max_value=1000.0, value=120.0, step=5.0)
            total_of_special_requests = st.number_input("Special requests", min_value=0, max_value=10, value=0, step=1)

        st.markdown('<div class="section-title" style="font-size:1.45rem;">Guest & Reservation Signals</div>', unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)

        with col4:
            adults = st.number_input("Adults", min_value=0, max_value=10, value=2, step=1)
            children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
            babies = st.number_input("Babies", min_value=0, max_value=5, value=0, step=1)
            meal = st.selectbox("Meal plan", ["BB", "HB", "SC", "FB", "Undefined"])

        with col5:
            country = st.text_input("Guest country code", value="PRT")
            distribution_channel = st.selectbox(
                "Distribution channel",
                ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
            )
            is_repeated_guest = st.selectbox("Repeated guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=30, value=0, step=1)

        with col6:
            previous_bookings_not_canceled = st.number_input("Previous successful bookings", min_value=0, max_value=100, value=0, step=1)
            reserved_room_type = st.selectbox("Reserved room type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
            assigned_room_type = st.selectbox("Assigned room type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
            booking_changes = st.number_input("Booking changes", min_value=0, max_value=30, value=0, step=1)

        st.markdown('<div class="section-title" style="font-size:1.45rem;">Operational Details</div>', unsafe_allow_html=True)

        col7, col8, col9 = st.columns(3)

        with col7:
            required_car_parking_spaces = st.number_input("Required car parking spaces", min_value=0, max_value=10, value=0, step=1)

        with col8:
            days_in_waiting_list = st.number_input("Days in waiting list", min_value=0, max_value=400, value=0, step=1)

        with col9:
            agent = st.number_input("Agent ID", min_value=0, max_value=999, value=0, step=1)
            company = st.number_input("Company ID", min_value=0, max_value=999, value=0, step=1)

        submitted = st.form_submit_button("Predict Cancellation Risk")

    if submitted:
        manual_model_df, manual_full_df = build_manual_feature_row(
            hotel=hotel,
            lead_time=int(lead_time),
            arrival_year=int(arrival_year),
            arrival_month_number=int(arrival_month_number),
            arrival_day_of_month=int(arrival_day_of_month),
            stays_in_weekend_nights=int(stays_in_weekend_nights),
            stays_in_week_nights=int(stays_in_week_nights),
            adults=int(adults),
            children=int(children),
            babies=int(babies),
            meal=meal,
            country=country.upper().strip(),
            market_segment=market_segment,
            distribution_channel=distribution_channel,
            is_repeated_guest=int(is_repeated_guest),
            previous_cancellations=int(previous_cancellations),
            previous_bookings_not_canceled=int(previous_bookings_not_canceled),
            reserved_room_type=reserved_room_type,
            assigned_room_type=assigned_room_type,
            booking_changes=int(booking_changes),
            deposit_type=deposit_type,
            agent=int(agent),
            company=int(company),
            days_in_waiting_list=int(days_in_waiting_list),
            customer_type=customer_type,
            adr=float(adr),
            required_car_parking_spaces=int(required_car_parking_spaces),
            total_of_special_requests=int(total_of_special_requests),
        )

        result_df = predict_manual_booking(manual_model_df, manual_full_df)
        result_row = result_df.iloc[0]

        probability = float(result_row["predicted_cancellation_probability"])
        risk_label = result_row["risk_category"]
        decision = "Likely Cancelled" if int(result_row["predicted_is_cancelled"]) == 1 else "Likely Retained"

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            render_metric_card(
                "Cancellation Probability",
                format_percent(probability),
                "Predicted likelihood that this booking will cancel.",
                "light",
            )

        with col_b:
            render_risk_card(risk_label, probability)

        with col_c:
            render_metric_card(
                "Model Decision",
                decision,
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

        col_left, col_right = st.columns(2)

        with col_left:
            render_list_card(
                "Why the booking may be risky",
                get_risk_drivers(result_row),
                light=False,
            )

        with col_right:
            render_list_card(
                "Recommended manager actions",
                get_recommended_actions(risk_label, result_row),
                light=True,
            )

        render_text_card(
            "Manager Summary",
            create_manager_summary(result_row),
            light=False,
        )

        st.markdown('<div class="section-title" style="font-size:1.55rem;">Generated Booking Signals</div>', unsafe_allow_html=True)

        signal_cols = [
            "hotel",
            "market_segment",
            "customer_type",
            "lead_time",
            "lead_time_bucket",
            "arrival_season",
            "total_nights",
            "stay_length_category",
            "adr",
            "adr_bucket",
            "deposit_type",
            "estimated_booking_value",
            "room_type_changed",
            "previous_cancellation_rate",
            "has_special_requests",
            "is_weekend_stay",
        ]

        available_signal_cols = [col for col in signal_cols if col in result_df.columns]
        st.dataframe(result_df[available_signal_cols].T.rename(columns={0: "Value"}), use_container_width=True)


# ============================================================
# MODE 2: EXPLAIN EXISTING BOOKING
# ============================================================

elif mode == "Explain Existing Booking":
    render_section(
        "Explain Existing Booking",
        "Select one booking from the feature-engineered dataset and generate a hotel-manager style risk explanation.",
    )

    sample_size = min(500, len(df))
    sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    scored_sample_df = predict_bookings(sample_df)

    selected_index = st.selectbox(
        "Select a sample booking",
        scored_sample_df.index,
        format_func=lambda x: f"Booking #{x}",
    )

    selected_row = scored_sample_df.loc[selected_index]
    probability = float(selected_row["predicted_cancellation_probability"])
    risk_label = selected_row["risk_category"]
    decision = "Likely Cancelled" if int(selected_row["predicted_is_cancelled"]) == 1 else "Likely Retained"

    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric_card(
            "Cancellation Probability",
            format_percent(probability),
            "Predicted likelihood that this reservation will cancel.",
            "light",
        )

    with col2:
        render_risk_card(risk_label, probability)

    with col3:
        render_metric_card(
            "Model Decision",
            decision,
            "Threshold-based operational classification.",
            "sage" if decision == "Likely Retained" else "blush",
        )

    display_cols = [
        col for col in [
            "hotel",
            "market_segment",
            "customer_type",
            "lead_time",
            "adr",
            "total_nights",
            "total_of_special_requests",
            "deposit_type",
            "estimated_booking_value",
        ]
        if col in scored_sample_df.columns
    ]

    st.markdown('<div class="section-title" style="font-size:1.55rem;">Booking Details</div>', unsafe_allow_html=True)
    st.dataframe(selected_row[display_cols].to_frame(name="Value"), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        render_list_card(
            "Why the model may be concerned",
            get_risk_drivers(selected_row),
            light=False,
        )

    with col_b:
        render_list_card(
            "Recommended manager actions",
            get_recommended_actions(risk_label, selected_row),
            light=True,
        )

    render_text_card(
        "Manager Summary",
        create_manager_summary(selected_row),
        light=False,
    )


# ============================================================
# MODE 3: BATCH BOOKING RISK REVIEW
# ============================================================

elif mode == "Batch Booking Risk Review":
    render_section(
        "Batch Booking Risk Review",
        "Upload a CSV file or use the available dataset to review cancellation risk across many bookings.",
    )

    uploaded_file = st.file_uploader("Upload booking CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
    else:
        batch_df = df.copy()
        source_label = "Feature-engineered dataset"

    scored_df = predict_bookings(batch_df)

    total_bookings = len(scored_df)
    avg_risk = scored_df["predicted_cancellation_probability"].mean()
    high_risk_count = (scored_df["risk_category"] == "High Risk").sum()
    at_risk_count = scored_df["predicted_is_cancelled"].sum()
    total_revenue_at_risk = scored_df["estimated_revenue_at_risk"].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Dataset Source", source_label, "Current dataset being analysed.", "light")

    with col2:
        render_metric_card("Bookings Reviewed", f"{total_bookings:,}", "Total bookings scored by the model.", "sage")

    with col3:
        render_metric_card("Predicted At-Risk", f"{int(at_risk_count):,}", "Bookings above the decision threshold.", "blush")

    with col4:
        render_metric_card("Revenue at Risk", format_currency(total_revenue_at_risk), "Probability-weighted booking value exposure.", "light")

    render_text_card(
        "Batch Summary",
        (
            f"The batch contains <strong>{total_bookings:,}</strong> bookings with an average predicted "
            f"cancellation probability of <strong>{format_percent(avg_risk)}</strong>. "
            f"The model flags <strong>{int(at_risk_count):,}</strong> bookings as likely cancellations, "
            f"including <strong>{int(high_risk_count):,}</strong> high-risk reservations. "
            f"The estimated revenue at risk is <strong>{format_currency(total_revenue_at_risk)}</strong>."
        ),
        light=False,
    )

    top_n = st.slider("Number of high-risk bookings to display", 5, 50, 15)

    preview_cols = [
        col for col in [
            "hotel",
            "market_segment",
            "customer_type",
            "lead_time",
            "adr",
            "estimated_booking_value",
            "predicted_cancellation_probability",
            "risk_category",
            "predicted_is_cancelled",
            "estimated_revenue_at_risk",
        ]
        if col in scored_df.columns
    ]

    top_risk_df = scored_df.sort_values(
        by=["predicted_cancellation_probability", "estimated_revenue_at_risk"],
        ascending=[False, False],
    ).head(top_n)

    st.markdown('<div class="section-title" style="font-size:1.55rem;">Top Risk Bookings</div>', unsafe_allow_html=True)
    st.dataframe(top_risk_df[preview_cols], use_container_width=True)

    csv_output = scored_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download scored booking file",
        data=csv_output,
        file_name="scored_hotel_bookings.csv",
        mime="text/csv",
    )


# ============================================================
# MODE 4: SEGMENT DASHBOARD
# ============================================================

elif mode == "Segment Dashboard":
    render_section(
        "Segment Dashboard",
        "Compare cancellation exposure across market segments, customer types, and hotel categories.",
    )

    scored_df = predict_bookings(df)

    segment_col = st.selectbox(
        "Choose segment dimension",
        ["market_segment", "customer_type", "hotel"],
    )

    if segment_col not in scored_df.columns:
        st.warning(f"`{segment_col}` is not available in the dataset.")
        st.stop()

    segment_summary = (
        scored_df.groupby(segment_col, dropna=False)
        .agg(
            booking_count=(segment_col, "count"),
            avg_predicted_cancellation_probability=("predicted_cancellation_probability", "mean"),
            predicted_cancellation_rate=("predicted_is_cancelled", "mean"),
            total_estimated_booking_value=("estimated_booking_value", "sum"),
            estimated_revenue_at_risk=("estimated_revenue_at_risk", "sum"),
        )
        .reset_index()
    )

    segment_summary = segment_summary.sort_values(
        by="estimated_revenue_at_risk",
        ascending=False,
    )

    top_segment = segment_summary.iloc[0]
    total_revenue_at_risk = segment_summary["estimated_revenue_at_risk"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric_card(
            "Highest Revenue Exposure",
            str(top_segment[segment_col]),
            "Segment with the highest estimated revenue at risk.",
            "blush",
        )

    with col2:
        render_metric_card(
            "Top Segment Risk",
            format_percent(float(top_segment["avg_predicted_cancellation_probability"])),
            "Average predicted cancellation probability.",
            "light",
        )

    with col3:
        render_metric_card(
            "Total Revenue at Risk",
            format_currency(float(total_revenue_at_risk)),
            "Estimated exposure across all segments.",
            "sage",
        )

    render_text_card(
        "Segment-Level Readout",
        (
            f"Using <strong>{segment_col}</strong> as the segmentation lens, "
            f"<strong>{top_segment[segment_col]}</strong> shows the highest revenue exposure. "
            f"This segment has <strong>{int(top_segment['booking_count']):,}</strong> bookings and "
            f"an estimated revenue at risk of "
            f"<strong>{format_currency(float(top_segment['estimated_revenue_at_risk']))}</strong>."
        ),
        light=False,
    )

    display_segment_summary = segment_summary.copy()
    display_segment_summary["avg_predicted_cancellation_probability"] = (
        display_segment_summary["avg_predicted_cancellation_probability"] * 100
    ).round(2)
    display_segment_summary["predicted_cancellation_rate"] = (
        display_segment_summary["predicted_cancellation_rate"] * 100
    ).round(2)
    display_segment_summary["total_estimated_booking_value"] = (
        display_segment_summary["total_estimated_booking_value"].round(2)
    )
    display_segment_summary["estimated_revenue_at_risk"] = (
        display_segment_summary["estimated_revenue_at_risk"].round(2)
    )

    st.markdown('<div class="section-title" style="font-size:1.55rem;">Segment Risk Table</div>', unsafe_allow_html=True)
    st.dataframe(display_segment_summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title" style="font-size:1.55rem;">Revenue at Risk by Segment</div>', unsafe_allow_html=True)

    chart_df = segment_summary.sort_values(
        by="estimated_revenue_at_risk",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    values = chart_df["estimated_revenue_at_risk"].values

    if values.max() == values.min():
        colors = plt.cm.coolwarm(np.linspace(0.35, 0.75, len(values)))
    else:
        normalized_values = (values - values.min()) / (values.max() - values.min())
        colors = plt.cm.coolwarm(normalized_values)

    ax.barh(chart_df[segment_col].astype(str), values, color=colors)
    ax.set_title("Estimated Revenue at Risk by Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Estimated Revenue at Risk")
    ax.set_ylabel(segment_col.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    st.pyplot(fig)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="small-note">
        Disclaimer: This assistant provides model-based decision support. It should be used as a prioritisation tool,
        not as a guaranteed prediction of guest behaviour or final revenue loss.
    </div>
    """,
    unsafe_allow_html=True,
)