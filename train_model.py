"""
train_model.py
──────────────
Rebuilds the model bundle from scratch using whatever scikit-learn
version is installed in the current environment.

Called automatically by streamlit_app.py on first launch if the
saved bundle is missing or incompatible. Also runnable directly:

    python3 train_model.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "hotel_bookings.csv")
MODEL_PATH = os.path.join(ROOT, "models", "cancellation_model.joblib")

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "lead_time", "adr", "adults", "children",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "total_of_special_requests",
    "required_car_parking_spaces", "days_in_waiting_list",
]

NOMINAL_FEATURES = [
    "hotel", "market_segment", "distribution_channel",
    "deposit_type", "customer_type",
]

BINARY_FEATURES = [
    "is_repeated_guest",
]

LEAD_TIME_LABELS  = ["0-30 Days","31-90 Days","91-180 Days","181-365 Days","365+ Days"]
STAY_LENGTH_ORDER = ["Zero-night stay","Short stay","Medium stay","Long stay","Extended stay"]

ORDINAL_FEATURES   = ["lead_time_bucket", "stay_length_category"]
ORDINAL_CATEGORIES = [LEAD_TIME_LABELS, STAY_LENGTH_ORDER]

MODEL_FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["children"] = df["children"].fillna(0)
    df["lead_time_bucket"] = pd.cut(
        df["lead_time"],
        bins=[0, 30, 90, 180, 365, np.inf],
        labels=LEAD_TIME_LABELS,
        include_lowest=True,
    ).astype("object")
    total = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["stay_length_category"] = pd.cut(
        total,
        bins=[-0.01, 0, 2, 5, 10, np.inf],
        labels=STAY_LENGTH_ORDER,
    ).astype("object")
    return df


def build_and_save():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    df = engineer_features(df)
    print(f"Rows after dedup: {len(df):,}")

    X = df[MODEL_FEATURES].copy()
    y = df["is_canceled"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", RobustScaler(), NUMERIC_FEATURES),
        ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
        ("ord", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ), ORDINAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE,
        )),
    ])

    print("Training model ...")
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")

    best_cost, best_thresh = float("inf"), 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = fn * 3.0 + fp * 1.0
        if cost < best_cost:
            best_cost, best_thresh = cost, t
    threshold = round(float(best_thresh), 2)
    print(f"Threshold: {threshold}")

    import sklearn
    bundle = {
        "pipeline":        pipeline,
        "threshold":       threshold,
        "model_features":  MODEL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "ordinal_categories": ORDINAL_CATEGORIES,
        "test_auc":        auc,
        "sklearn_version": sklearn.__version__,
        "excluded_features_leakage_risk": [
            "room_type_changed", "reserved_room_type", "assigned_room_type"
        ],
        "cost_assumptions": {
            "cost_per_missed_cancellation": 3.0,
            "cost_per_false_alarm": 1.0,
        },
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Bundle saved to {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    build_and_save()
