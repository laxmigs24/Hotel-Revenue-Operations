# streamlit_app.py

import os
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
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f3ec;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #243b35;
        font-family: 'Georgia', serif;
    }

    .hero-card {
        background: linear-gradient(135deg, #243b35 0%, #5f7f6f 100%);
        padding: 2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0px 8px 28px rgba(0,0,0,0.12);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        font-family: Georgia, serif;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #efe7d7;
        max-width: 900px;
    }

    .metric-card {
        background-color: #ffffff;
        padding: 1.3rem;
        border-radius: 18px;
        border: 1px solid #e6ded1;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.06);
        text-align: center;
    }

    .risk-high {
        background-color: #7a2e2e;
        color: white;
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .risk-medium {
        background-color: #b87935;
        color: white;
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .risk-low {
        background-color: #3f6f5f;
        color: white;
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .assistant-box {
        background-color: #ffffff;
        border-left: 6px solid #b87935;
        padding: 1.4rem;
        border-radius: 14px;
        margin-top: 1rem;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.06);
    }

    .small-note {
        font-size: 0.85rem;
        color: #6d6d6d;
    }

    .stDataFrame {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

from pathlib import Path

# Base paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "xgboost_cancellation_model_bundle.joblib"
DATA_PATH = PROJECT_DIR / "data" / "hotel_bookings_feature_engineered.csv"


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model_bundle(path):
    if not path.exists():
        st.error(f"Model bundle not found at: {path}")
        st.stop()

    return joblib.load(path)


@st.cache_data
def load_data(path):
    if not path.exists():
        return None

    return pd.read_csv(path)


model_bundle = load_model_bundle(MODEL_PATH)
df = load_data(DATA_PATH)

model_name = model_bundle["model_name"]
model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
selected_threshold = model_bundle["selected_threshold"]
model_features = model_bundle["model_features"]
target_col = model_bundle["target_column"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_risk(probability: float) -> str:
    if probability >= 0.70:
        return "High Risk"
    elif probability >= selected_threshold:
        return "Medium Risk"
    else:
        return "Low Risk"


def risk_css_class(risk_label: str) -> str:
    if risk_label == "High Risk":
        return "risk-high"
    elif risk_label == "Medium Risk":
        return "risk-medium"
    return "risk-low"


def get_risk_drivers(row: pd.Series) -> list:
    drivers = []

    if "lead_time" in row and row["lead_time"] >= 90:
        drivers.append("Long lead time may increase cancellation uncertainty.")

    if "market_segment" in row and row["market_segment"] == "Online TA":
        drivers.append("Online travel agency bookings showed higher cancellation risk.")

    if "total_of_special_requests" in row and row["total_of_special_requests"] == 0:
        drivers.append("No special requests may indicate lower guest engagement.")

    if "previous_cancellation_rate" in row and row["previous_cancellation_rate"] > 0:
        drivers.append("Previous cancellation behaviour increases predicted risk.")

    if "deposit_type" in row and row["deposit_type"] == "No Deposit":
        drivers.append("No deposit may reduce booking commitment.")

    if "customer_type" in row and row["customer_type"] == "Transient":
        drivers.append("Transient customers showed higher cancellation concentration.")

    if "is_last_minute_booking" in row and row["is_last_minute_booking"] == 1:
        drivers.append("Last-minute booking behaviour is an important model signal.")

    if "room_type_changed" in row and row["room_type_changed"] == 1:
        drivers.append("Room type change is a strong model signal for this booking.")

    if len(drivers) == 0:
        drivers.append("No major individual risk driver was detected from the selected business rules.")

    return drivers[:5]


def get_recommended_actions(risk_label: str) -> list:
    if risk_label == "High Risk":
        return [
            "Send a personalised confirmation message immediately.",
            "Offer a small upgrade, add-on, or flexible stay incentive.",
            "Monitor the booking closer to arrival.",
            "Consider controlled overbooking only if operationally safe.",
            "Prioritise this booking in revenue-protection workflows."
        ]

    if risk_label == "Medium Risk":
        return [
            "Send a reminder or pre-arrival engagement message.",
            "Offer optional add-ons to increase guest commitment.",
            "Track the booking as part of the cancellation watchlist."
        ]

    return [
        "No urgent intervention required.",
        "Continue standard pre-arrival communication.",
        "Use this booking as part of normal occupancy planning."
    ]


def create_assistant_summary(probability: float, risk_label: str, row: pd.Series) -> str:
    hotel = row.get("hotel", "this hotel")
    market_segment = row.get("market_segment", "unknown segment")
    customer_type = row.get("customer_type", "unknown customer type")

    return f"""
    This booking is classified as **{risk_label}** with an estimated cancellation probability of **{probability:.1%}**.

    The booking belongs to **{hotel}**, comes through the **{market_segment}** segment, and is associated with the **{customer_type}** customer type.

    Based on the model output, this booking should be reviewed as part of the hotel's cancellation-risk monitoring process.
    """


def predict_single_booking(input_df: pd.DataFrame):
    missing_cols = [col for col in model_features if col not in input_df.columns]

    if missing_cols:
        st.error("The input data is missing required model features.")
        st.write(missing_cols)
        st.stop()

    X_input = input_df[model_features].copy()
    X_processed = preprocessor.transform(X_input)

    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= selected_threshold).astype(int)

    return probabilities, predictions


def format_currency(value):
    return f"€{value:,.2f}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏨 Assistant Controls")

st.sidebar.markdown(
    """
    This assistant uses a trained XGBoost model to estimate cancellation risk and support hotel revenue decisions.
    """
)

st.sidebar.divider()

mode = st.sidebar.radio(
    "Choose analysis mode",
    [
        "Single Booking Assistant",
        "Batch Booking Risk Review",
        "Segment Dashboard"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Decision threshold:** {selected_threshold}")
st.sidebar.write(f"**Features used:** {len(model_features)}")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Hotel Cancellation Risk Assistant</div>
        <div class="hero-subtitle">
            A decision-support tool for hotel managers to identify high-risk bookings,
            understand cancellation drivers, and protect revenue using machine learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODE 1: SINGLE BOOKING ASSISTANT
# ============================================================

if mode == "Single Booking Assistant":

    st.header("Single Booking Assistant")

    if df is None:
        st.warning("Feature-engineered dataset not found. Please use batch upload mode.")
        st.stop()

    st.markdown(
        """
        Select one booking from the test/sample dataset and generate a hotel-manager style risk explanation.
        """
    )

    sample_size = min(500, len(df))
    sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    selected_index = st.selectbox(
        "Select a sample booking",
        sample_df.index,
        format_func=lambda x: f"Booking #{x}"
    )

    selected_booking = sample_df.loc[[selected_index]].copy()
    selected_row = selected_booking.iloc[0]

    probabilities, predictions = predict_single_booking(selected_booking)

    probability = float(probabilities[0])
    prediction = int(predictions[0])
    risk_label = classify_risk(probability)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Cancellation Probability</h3>
                <h2>{probability:.1%}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="{risk_css_class(risk_label)}">
                {risk_label}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Model Decision</h3>
                <h2>{"Likely Cancelled" if prediction == 1 else "Likely Not Cancelled"}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Booking Details")

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
            "estimated_booking_value"
        ]
        if col in selected_booking.columns
    ]

    st.dataframe(selected_booking[display_cols], use_container_width=True)

    st.subheader("Assistant Recommendation")

    assistant_summary = create_assistant_summary(probability, risk_label, selected_row)

    st.markdown(
        f"""
        <div class="assistant-box">
        {assistant_summary}
        </div>
        """,
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Main Risk Drivers")
        for driver in get_risk_drivers(selected_row):
            st.write(f"• {driver}")

    with col_b:
        st.markdown("### Recommended Actions")
        for action in get_recommended_actions(risk_label):
            st.write(f"• {action}")


# ============================================================
# MODE 2: BATCH BOOKING RISK REVIEW
# ============================================================

elif mode == "Batch Booking Risk Review":

    st.header("Batch Booking Risk Review")

    st.markdown(
        """
        Upload a CSV file with the same feature-engineered structure used during model training.
        The assistant will score each booking and identify high-risk reservations.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload booking CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.write(f"Uploaded rows: **{batch_df.shape[0]:,}**")
        st.write(f"Uploaded columns: **{batch_df.shape[1]}**")

        missing_cols = [col for col in model_features if col not in batch_df.columns]

        if missing_cols:
            st.error("Uploaded file is missing required model features.")
            st.write(missing_cols)
            st.stop()

        probabilities, predictions = predict_single_booking(batch_df)

        scored_df = batch_df.copy()
        scored_df["predicted_cancellation_probability"] = probabilities
        scored_df["predicted_is_cancelled"] = predictions
        scored_df["risk_category"] = scored_df["predicted_cancellation_probability"].apply(classify_risk)

        if "estimated_booking_value" in scored_df.columns:
            scored_df["estimated_revenue_at_risk"] = (
                scored_df["predicted_cancellation_probability"] *
                scored_df["estimated_booking_value"]
            )
        else:
            scored_df["estimated_revenue_at_risk"] = np.nan

        high_risk_count = (scored_df["risk_category"] == "High Risk").sum()
        medium_risk_count = (scored_df["risk_category"] == "Medium Risk").sum()
        low_risk_count = (scored_df["risk_category"] == "Low Risk").sum()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("High Risk Bookings", f"{high_risk_count:,}")

        with col2:
            st.metric("Medium Risk Bookings", f"{medium_risk_count:,}")

        with col3:
            st.metric("Low Risk Bookings", f"{low_risk_count:,}")

        st.subheader("Scored Booking Preview")

        preview_cols = [
            col for col in [
                "hotel",
                "market_segment",
                "customer_type",
                "estimated_booking_value",
                "predicted_cancellation_probability",
                "risk_category",
                "estimated_revenue_at_risk"
            ]
            if col in scored_df.columns
        ]

        st.dataframe(
            scored_df[preview_cols].sort_values(
                by="predicted_cancellation_probability",
                ascending=False
            ),
            use_container_width=True
        )

        csv_output = scored_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download scored booking file",
            data=csv_output,
            file_name="scored_hotel_bookings.csv",
            mime="text/csv"
        )

    else:
        st.info("Upload a CSV file to begin batch risk scoring.")


# ============================================================
# MODE 3: SEGMENT DASHBOARD
# ============================================================

elif mode == "Segment Dashboard":

    st.header("Segment Dashboard")

    if df is None:
        st.warning("Feature-engineered dataset not found.")
        st.stop()

    st.markdown(
        """
        This dashboard scores the available feature-engineered dataset and summarises cancellation risk by hotel segment.
        """
    )

    probabilities, predictions = predict_single_booking(df)

    dashboard_df = df.copy()
    dashboard_df["predicted_cancellation_probability"] = probabilities
    dashboard_df["predicted_is_cancelled"] = predictions
    dashboard_df["risk_category"] = dashboard_df["predicted_cancellation_probability"].apply(classify_risk)

    if "estimated_booking_value" in dashboard_df.columns:
        dashboard_df["estimated_revenue_at_risk"] = (
            dashboard_df["predicted_cancellation_probability"] *
            dashboard_df["estimated_booking_value"]
        )
    else:
        dashboard_df["estimated_revenue_at_risk"] = np.nan

    total_bookings = dashboard_df.shape[0]
    avg_risk = dashboard_df["predicted_cancellation_probability"].mean()
    total_revenue_at_risk = dashboard_df["estimated_revenue_at_risk"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Bookings Scored", f"{total_bookings:,}")

    with col2:
        st.metric("Average Predicted Risk", f"{avg_risk:.1%}")

    with col3:
        st.metric("Estimated Revenue at Risk", format_currency(total_revenue_at_risk))

    st.subheader("Market Segment Risk Profile")

    if "market_segment" in dashboard_df.columns:
        segment_summary = (
            dashboard_df
            .groupby("market_segment")
            .agg(
                booking_count=("market_segment", "count"),
                avg_predicted_risk=("predicted_cancellation_probability", "mean"),
                predicted_at_risk_bookings=("predicted_is_cancelled", "sum"),
                estimated_revenue_at_risk=("estimated_revenue_at_risk", "sum")
            )
            .reset_index()
        )

        segment_summary["avg_predicted_risk"] = (
            segment_summary["avg_predicted_risk"] * 100
        ).round(2)

        segment_summary["estimated_revenue_at_risk"] = (
            segment_summary["estimated_revenue_at_risk"].round(2)
        )

        segment_summary = segment_summary.sort_values(
            by="estimated_revenue_at_risk",
            ascending=False
        )

        st.dataframe(segment_summary, use_container_width=True)

        plot_data = segment_summary.sort_values(
            by="estimated_revenue_at_risk",
            ascending=True
        )

        colors = plt.cm.coolwarm(
            (plot_data["estimated_revenue_at_risk"] - plot_data["estimated_revenue_at_risk"].min()) /
            (plot_data["estimated_revenue_at_risk"].max() - plot_data["estimated_revenue_at_risk"].min())
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            plot_data["market_segment"],
            plot_data["estimated_revenue_at_risk"],
            color=colors
        )

        ax.set_title(
            "Estimated Revenue at Risk by Market Segment",
            fontsize=14,
            fontweight="bold",
            pad=15
        )
        ax.set_xlabel("Estimated Revenue at Risk")
        ax.set_ylabel("Market Segment")

        st.pyplot(fig)

    st.subheader("Assistant Business Summary")

    if "market_segment" in dashboard_df.columns:
        top_segment = segment_summary.iloc[0]

        st.markdown(
            f"""
            <div class="assistant-box">
            The assistant identifies <b>{top_segment['market_segment']}</b> as the segment with the highest estimated revenue at risk.
            This segment contains <b>{int(top_segment['booking_count']):,}</b> bookings and an estimated revenue exposure of 
            <b>{format_currency(top_segment['estimated_revenue_at_risk'])}</b>.
            <br><br>
            Hotel managers should prioritise this segment for cancellation monitoring, guest re-confirmation campaigns,
            and revenue-protection actions.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <p class="small-note">
    Disclaimer: This assistant provides model-based decision support. It should be used as a prioritisation tool,
    not as a guaranteed prediction of guest behaviour or final revenue loss.
    </p>
    """,
    unsafe_allow_html=True
)
