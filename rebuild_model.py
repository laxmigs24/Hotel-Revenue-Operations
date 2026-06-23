"""
rebuild_model.py
────────────────
Run this script ONCE from the root of your Hotel-Revenue-Operations folder:

    cd ~/Desktop/Hotel-Revenue-Operations
    python3 rebuild_model.py

It will:
  1. Load the raw hotel_bookings.csv from data/
  2. Run the full feature-engineering pipeline (via src/features.py)
  3. Train the same HistGradientBoostingClassifier used in notebook 04
  4. Save a NEW models/cancellation_model.joblib built with YOUR
     local scikit-learn and numpy versions — so the app can load it
     without any version-mismatch errors.

Requirements (all already in your Anaconda base env):
  pandas, numpy, scikit-learn, joblib
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from src import features as feat

RANDOM_STATE = 42

print("=" * 60)
print("Hotel Revenue Operations — Model Rebuild Script")
print(f"scikit-learn : {__import__('sklearn').__version__}")
print(f"numpy        : {np.__version__}")
print(f"pandas       : {pd.__version__}")
print("=" * 60)

# ── 1. Load raw data ──────────────────────────────────────────
RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "hotel_bookings.csv")
if not os.path.exists(RAW_PATH):
    print(f"\nERROR: Cannot find {RAW_PATH}")
    print("Make sure you are running this from the Hotel-Revenue-Operations root folder.")
    sys.exit(1)

print(f"\n[1/5] Loading raw data from {RAW_PATH} ...")
df = pd.read_csv(RAW_PATH)
rows_before = len(df)
df = df.drop_duplicates()
print(f"      Loaded {rows_before:,} rows → {len(df):,} after dedup")

# ── 2. Feature engineering ────────────────────────────────────
print("\n[2/5] Engineering features ...")
df = feat.engineer_all_features(df)

# Validate ordinal labels match category lists (the original bug check)
for col, order in zip(feat.ORDINAL_FEATURES, feat.ORDINAL_CATEGORIES):
    produced = set(df[col].dropna().unique())
    expected = set(order)
    unmatched = produced - expected
    if unmatched:
        print(f"  WARNING: {col} has unmatched labels: {unmatched}")
    else:
        print(f"  OK  {col}: labels match ordinal category list")

# ── 3. Train / test split ─────────────────────────────────────
print("\n[3/5] Splitting data ...")
X = df[feat.MODEL_FEATURES].copy()
y = df["is_canceled"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. Build & train pipeline ─────────────────────────────────
print("\n[4/5] Training model ...")

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", RobustScaler(), feat.NUMERIC_FEATURES),
        ("nominal", OneHotEncoder(handle_unknown="ignore"), feat.NOMINAL_FEATURES),
        (
            "ordinal",
            OrdinalEncoder(
                categories=feat.ORDINAL_CATEGORIES,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            feat.ORDINAL_FEATURES,
        ),
        ("binary", "passthrough", feat.BINARY_FEATURES),
    ],
    remainder="drop",
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE
    )),
])

pipeline.fit(X_train, y_train)
print("      Training complete.")

# ── 5. Evaluate & pick cost-weighted threshold ────────────────
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n      Test ROC-AUC: {test_auc:.4f}")

cost_per_fn = 3.0
cost_per_fp = 1.0
thresholds  = np.arange(0.05, 0.96, 0.01)
best_cost   = float("inf")
best_thresh = 0.5

for t in thresholds:
    preds = (y_pred_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    cost = fn * cost_per_fn + fp * cost_per_fp
    if cost < best_cost:
        best_cost   = cost
        best_thresh = t

FINAL_THRESHOLD = round(float(best_thresh), 2)
print(f"      Cost-weighted threshold: {FINAL_THRESHOLD}")

# ── 6. Save bundle ────────────────────────────────────────────
print("\n[5/5] Saving model bundle ...")

import sklearn
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)
out_path = os.path.join(models_dir, "cancellation_model.joblib")

bundle = {
    "pipeline":                      pipeline,
    "threshold":                     FINAL_THRESHOLD,
    "model_features":                feat.MODEL_FEATURES,
    "ordinal_features":              feat.ORDINAL_FEATURES,
    "ordinal_categories":            feat.ORDINAL_CATEGORIES,
    "test_auc":                      test_auc,
    "cost_assumptions": {
        "cost_per_missed_cancellation": cost_per_fn,
        "cost_per_false_alarm":         cost_per_fp,
    },
    "excluded_features_leakage_risk": [
        "room_type_changed", "reserved_room_type", "assigned_room_type"
    ],
    "sklearn_version": sklearn.__version__,
    "numpy_version":   np.__version__,
}

joblib.dump(bundle, out_path)
print(f"      Saved to: {out_path}")

print("\n" + "=" * 60)
print("Done! Now run:  streamlit run app/streamlit_app.py")
print("=" * 60)
