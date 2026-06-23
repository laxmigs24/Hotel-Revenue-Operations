"""
Shared feature engineering for the Hotel Cancellation Risk project.

This module is the SINGLE SOURCE OF TRUTH for every engineered feature,
bucket definition, and category ordering used in this project. It is
imported identically by:

  - notebooks/03_feature_eng.ipynb   (creates the training dataset)
  - notebooks/04_modeling_evaluation.ipynb (builds the preprocessing pipeline)
  - app/streamlit_app.py             (scores new bookings at inference time)

Why this file exists
---------------------
The original version of this project defined bucket logic (lead_time_bucket,
adr_bucket, is_peak_season, etc.) independently in three different places —
once in the EDA notebook, once in the feature engineering notebook (with a
typo: en-dash characters instead of hyphens), and once again inside the
Streamlit app (with a completely different bucket scheme). The three
definitions disagreed with each other, which caused the trained model's
OrdinalEncoder to silently map real bookings to an "unknown" category at
both training time and inference time.

Centralizing the logic here means there is exactly one place to look, and
exactly one place to fix, if a feature definition ever needs to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS — bucket boundaries & category orderings
# ============================================================
#
# IMPORTANT: the *_ORDER lists below MUST exactly match the string labels
# produced by the corresponding bucket function (character-for-character,
# including dash type). They are consumed directly by sklearn's
# OrdinalEncoder(categories=...) in 04_modeling_evaluation.ipynb.
# A unit test in tests/test_features.py asserts this invariant on every run.

LEAD_TIME_BINS = [0, 30, 90, 180, 365, np.inf]
LEAD_TIME_LABELS = [
    "0-30 Days",
    "31-90 Days",
    "91-180 Days",
    "181-365 Days",
    "365+ Days",
]
LEAD_TIME_ORDER = LEAD_TIME_LABELS  # ordinal order == bucket order

ADR_BINS = [0, 50, 100, 150, 200, 300, np.inf]
ADR_LABELS = [
    "0-50",
    "51-100",
    "101-150",
    "151-200",
    "201-300",
    "300+",
]
# "Invalid ADR" is appended for negative-ADR rows; it sits below "0-50"
# in business meaning (data-quality flag, not a real price tier), but we
# put it at the END of the ordinal order so it doesn't distort the
# monotonic price ordering for the 99.999% of rows with valid ADR.
ADR_ORDER = ADR_LABELS + ["Invalid ADR"]

STAY_LENGTH_ORDER = [
    "Zero-night stay",
    "Short stay",
    "Medium stay",
    "Long stay",
    "Extended stay",
]

PEAK_SEASON_MONTHS = {"April", "May", "June", "July", "August"}

MONTH_MAPPING = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
MONTH_NAMES = {v: k for k, v in MONTH_MAPPING.items()}

LAST_MINUTE_LEAD_DAYS = 7
LONG_LEAD_DAYS = 90

# Leakage-prone columns that must never be used as model inputs.
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

# Excluded from the first modeling version due to overlap, high cardinality,
# or because they are not realistically knowable at prediction time.
# `room_type_changed` is intentionally EXCLUDED (see model card / README
# "Known Limitations" section) because `assigned_room_type` is typically
# only finalized by hotel staff at or near check-in, making it a likely
# proxy for "this booking was not cancelled" rather than a genuine
# ex-ante (booking-time) risk signal.
MODEL_EXCLUDE_FEATURES = [
    "arrival_date_month",   # captured by arrival_month_number, arrival_season, is_peak_season
    "country",              # high-cardinality; can be revisited later
    "reserved_room_type",   # not safely usable without assigned_room_type context
    "assigned_room_type",   # only known at/near check-in -> leakage risk
    "room_type_changed",    # derived from assigned_room_type -> leakage risk
    "meal",                 # optional; can be revisited later
    "agent",                # ID-like column, not a true numeric feature
    "company",              # ID-like column, not a true numeric feature
]


# ============================================================
# BUCKET / FLAG FUNCTIONS (vectorized, pandas-Series in -> Series out)
# ============================================================

def add_lead_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lead_time_bucket"] = pd.cut(
        df["lead_time"],
        bins=LEAD_TIME_BINS,
        labels=LEAD_TIME_LABELS,
        include_lowest=True,
    ).astype("object")
    return df


def add_adr_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_invalid_adr"] = np.where(df["adr"] < 0, 1, 0)
    bucket = pd.cut(
        df["adr"],
        bins=ADR_BINS,
        labels=ADR_LABELS,
        include_lowest=True,
    ).astype("object")
    # Negative ADR falls below the bucket range -> NaN from pd.cut.
    # Label it explicitly instead of dropping the row.
    df["adr_bucket"] = bucket.fillna("Invalid ADR")
    return df


def add_total_nights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_nights"] = (
        df["stays_in_weekend_nights"].fillna(0) + df["stays_in_week_nights"].fillna(0)
    )
    return df


def impute_children(df: pd.DataFrame) -> pd.DataFrame:
    """The raw dataset has 4 rows with a missing `children` value out of 119,390
    (and a handful remain after dedup). The original project never addressed
    this, which meant a few rows silently introduced NaN into the numeric
    feature matrix and broke any model that doesn't natively handle NaN
    (e.g. LogisticRegression). A missing children count overwhelmingly likely
    means zero children were on the booking (the dataset's other guest-count
    fields are always populated), so we impute with 0 rather than dropping
    rows or imputing with the mean/median, which would invent fractional
    children that don't make business sense.
    """
    df = df.copy()
    n_missing = df["children"].isna().sum()
    if n_missing > 0:
        df["children"] = df["children"].fillna(0)
    return df


def add_stay_length_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _categorize(nights: float) -> str:
        if nights == 0:
            return "Zero-night stay"
        elif nights <= 2:
            return "Short stay"
        elif nights <= 5:
            return "Medium stay"
        elif nights <= 10:
            return "Long stay"
        else:
            return "Extended stay"

    df["stay_length_category"] = df["total_nights"].apply(_categorize)
    df["is_zero_night_stay"] = np.where(df["total_nights"] == 0, 1, 0)
    df["is_weekend_stay"] = np.where(df["stays_in_weekend_nights"] > 0, 1, 0)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_last_minute_booking"] = np.where(df["lead_time"] <= LAST_MINUTE_LEAD_DAYS, 1, 0)
    df["is_long_lead_booking"] = np.where(df["lead_time"] >= LONG_LEAD_DAYS, 1, 0)
    df["arrival_month_number"] = df["arrival_date_month"].map(MONTH_MAPPING)
    df["arrival_season"] = df["arrival_date_month"].apply(assign_season)
    df["is_peak_season"] = np.where(
        df["arrival_date_month"].isin(PEAK_SEASON_MONTHS), 1, 0
    )
    return df


def assign_season(month_name: str) -> str:
    if month_name in ["December", "January", "February"]:
        return "Winter"
    elif month_name in ["March", "April", "May"]:
        return "Spring"
    elif month_name in ["June", "July", "August"]:
        return "Summer"
    else:
        return "Autumn"


def add_guest_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_previous_cancellation"] = np.where(df["previous_cancellations"] > 0, 1, 0)
    df["has_previous_successful_booking"] = np.where(
        df["previous_bookings_not_canceled"] > 0, 1, 0
    )
    df["has_booking_changes"] = np.where(df["booking_changes"] > 0, 1, 0)
    df["has_special_requests"] = np.where(df["total_of_special_requests"] > 0, 1, 0)
    df["is_waitlisted"] = np.where(df["days_in_waiting_list"] > 0, 1, 0)

    df["previous_total_bookings"] = (
        df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    )
    df["previous_cancellation_rate"] = np.where(
        df["previous_total_bookings"] > 0,
        df["previous_cancellations"] / df["previous_total_bookings"],
        0,
    )

    df["repeat_guest_with_deposit"] = np.where(
        (df["is_repeated_guest"] == 1) & (df["deposit_type"] != "No Deposit"), 1, 0
    )
    df["repeat_guest_non_refund"] = np.where(
        (df["is_repeated_guest"] == 1) & (df["deposit_type"] == "Non Refund"), 1, 0
    )
    return df


def add_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["estimated_booking_value"] = df["adr"] * df["total_nights"]
    df["is_zero_adr"] = np.where(df["adr"] == 0, 1, 0)
    adr_p75 = df["adr"].quantile(0.75)
    value_p75 = df["estimated_booking_value"].quantile(0.75)
    df["is_high_adr"] = np.where(df["adr"] >= adr_p75, 1, 0)
    df["is_high_value_booking"] = np.where(
        df["estimated_booking_value"] >= value_p75, 1, 0
    )
    return df


def add_room_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kept for EDA/business-narrative purposes only.

    NOTE: `room_type_changed` is intentionally excluded from
    MODEL_EXCLUDE_FEATURES above and must NOT be used as a model input.
    See the module docstring / README "Known Limitations" for rationale.
    """
    df = df.copy()
    df["room_type_changed"] = np.where(
        df["reserved_room_type"] != df["assigned_room_type"], 1, 0
    )
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline in the correct order.

    This is the ONE function that both the notebooks and the Streamlit app
    should call. Order matters: total_nights must exist before stay-length
    buckets, lead_time/adr buckets must exist before the candidate feature
    list is built, etc.
    """
    df = df.copy()
    df = impute_children(df)
    df = add_total_nights(df)
    df = add_lead_time_bucket(df)
    df = add_adr_bucket(df)
    df = add_stay_length_category(df)
    df = add_temporal_features(df)
    df = add_guest_behaviour_features(df)
    df = add_revenue_features(df)
    df = add_room_type_features(df)
    return df


# ============================================================
# FEATURE GROUPS for the sklearn preprocessing pipeline
# ============================================================

NUMERIC_FEATURES = [
    "lead_time",
    "adr",
    "total_nights",
    "estimated_booking_value",
    "total_of_special_requests",
    "booking_changes",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "previous_total_bookings",
    "previous_cancellation_rate",
    "required_car_parking_spaces",
    "arrival_month_number",
    "adults",
    "children",
    "babies",
]

NOMINAL_FEATURES = [
    "hotel",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
    "arrival_season",
]

ORDINAL_FEATURES = [
    "lead_time_bucket",
    "adr_bucket",
    "stay_length_category",
]

ORDINAL_CATEGORIES = [
    LEAD_TIME_ORDER,
    ADR_ORDER,
    STAY_LENGTH_ORDER,
]

BINARY_FEATURES = [
    "is_repeated_guest",
    "has_invalid_adr",
    "is_zero_night_stay",
    "is_weekend_stay",
    "is_last_minute_booking",
    "is_long_lead_booking",
    "is_peak_season",
    "has_previous_cancellation",
    "has_previous_successful_booking",
    "has_booking_changes",
    "has_special_requests",
    "is_waitlisted",
    "repeat_guest_with_deposit",
    "repeat_guest_non_refund",
    "is_zero_adr",
    "is_high_adr",
    "is_high_value_booking",
]

MODEL_FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
