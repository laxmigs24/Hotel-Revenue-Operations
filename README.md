# Hotel Revenue Operations
<img width="1659" height="725" alt="Screenshot 2026-07-30 at 13 02 53" src="https://github.com/user-attachments/assets/56030652-8cb0-4d19-beb0-f4ce1365520f" />
<img width="1657" height="345" alt="Screenshot 2026-07-30 at 13 03 07" src="https://github.com/user-attachments/assets/c1242795-efe7-4da2-a380-4e09d1017797" />


### Cancellation Risk Prediction & Revenue Intelligence for Hotel Managers

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange?style=flat-square)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What This Project Does

Hotels lose significant revenue every year to booking cancellations. A booking made three months in advance through an online travel agency carries very different risk to a same-day direct booking — but most hotel teams treat every reservation the same way until it cancels.

This project builds an end-to-end machine learning system that:

- **Predicts** the probability a booking will cancel, using information available at booking time
- **Explains** why a booking is high-risk in plain business language
- **Quantifies** the revenue exposure attached to predicted cancellations
- **Deploys** as an interactive Streamlit app for hotel revenue managers

> **Dataset:** Hotel Booking Demand (Antonio, de Almeida & Nunes, 2019) — 119,390 bookings across a City Hotel and Resort Hotel in Portugal (2015–2017)

---

## Live Demo

> 🚀 **[Launch the App → ](https://hotel-revenue-operations-gepejcj4wtb5gahz6des3f.streamlit.app/)**


---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 87,396 bookings (after deduplication) |
| Cancellation rate | 27.5% |
| Model | Gradient Boosting (`HistGradientBoostingClassifier`) |
| Test ROC-AUC | **0.840** |
| Decision threshold | **0.26** (cost-weighted — missed cancellation assumed 3× costlier than false alarm) |
| Total booking value | €34.5M |
| Revenue at risk (canceled) | €11.5M (33.3% of total) |
| Features used | 41 (after excluding 3 leakage-risk columns) |

---

## Project Structure

```
Hotel-Revenue-Operations/
│
├── app/
│   └── streamlit_app.py          # 4-mode Streamlit decision-support app
│
├── data/
│   ├── hotel_bookings.csv                    # Raw dataset
│   ├── hotel_bookings_clean.csv              # After dedup + audit
│   ├── hotel_bookings_eda.csv                # EDA-enriched version
│   └── hotel_bookings_feature_engineered.csv # Final modelling dataset
│
├── models/
│   └── cancellation_model.joblib  # Trained pipeline + threshold + metadata
│
├── notebooks/
│   ├── 01_data_audit.ipynb        # Load, dedup, audit raw data
│   ├── 02_eda.ipynb               # Exploratory analysis + business insights
│   ├── 03_feature_eng.ipynb       # Feature engineering (via src/features.py)
│   └── 04_modeling_evaluation.ipynb  # Model training, threshold, explainability
│
├── src/
│   └── features.py               # Single source of truth for all feature logic
│
├── tests/
│   └── test_features.py          # 16 regression tests — run without pytest
│
├── requirements.txt
├── LICENSE
└── README.md
```

### Why `src/features.py` matters

All feature engineering logic — bucket definitions, category orderings, season assignments — lives in one shared module imported identically by the notebooks **and** the Streamlit app. This eliminates the train/serve skew that affects most portfolio projects in this category (where the app re-implements feature logic slightly differently from the notebook and the model silently receives wrong inputs).

---

## Notebook Workflow

Run the four notebooks in order. Each one saves its output for the next.

### `01_data_audit.ipynb`
- Loads 119,390 raw bookings
- Documents and removes 31,994 duplicate rows (27% of data) with explicit justification
- Audits missing values across all 32 columns
- Flags 180 zero-guest bookings as a known data quality issue
- Saves `hotel_bookings_clean.csv` (87,396 rows)

### `02_eda.ipynb`
- Cancellation patterns across hotel type, market segment, deposit type, customer type
- Lead time and ADR relationship to cancellation risk
- Monthly and seasonal cancellation trends
- Revenue impact framing using `estimated_booking_value = adr × total_nights`
- All undefined variable bugs from the original version are fixed

### `03_feature_eng.ipynb`
- Imports `src/features.py` — no feature logic is duplicated here
- Validates that all ordinal bucket labels match the `OrdinalEncoder` category lists (the en-dash bug check)
- Documents the **leakage decision** for `room_type_changed` (see Known Limitations)
- Saves `hotel_bookings_feature_engineered.csv`

**Engineered feature groups:**

| Group | Features |
|-------|----------|
| Stay duration | `total_nights`, `stay_length_category`, `is_zero_night_stay`, `is_weekend_stay` |
| Lead time | `lead_time_bucket`, `is_last_minute_booking`, `is_long_lead_booking` |
| Revenue | `estimated_booking_value`, `adr_bucket`, `is_zero_adr`, `is_high_adr`, `is_high_value_booking` |
| Guest behaviour | `previous_cancellation_rate`, `has_previous_cancellation`, `has_booking_changes`, `has_special_requests`, `is_waitlisted` |
| Seasonality | `arrival_month_number`, `arrival_season`, `is_peak_season` |

### `04_modeling_evaluation.ipynb`
- 5-fold stratified cross-validation across Logistic Regression, Random Forest, and Gradient Boosting
- Final model trained on the full training set and evaluated on a held-out test set
- **Cost-weighted threshold selection** — missed cancellations assumed 3× costlier than false alarms; threshold chosen to minimise total business cost, not F1
- Permutation importance for feature explainability
- Segment-level cancellation risk profiling
- Revenue-at-risk estimation per hotel and segment
- Saves `models/cancellation_model.joblib`

---

## The Streamlit App

The app has four modes, accessible from the sidebar:

| Mode | What it does |
|------|-------------|
| **Manual Prediction** | Score a single new booking — fill in booking details, get a risk probability and plain-language explanation |
| **Explain a Booking** | Upload an existing booking and see which factors are driving its risk score |
| **Batch Scoring** | Upload a CSV of bookings and score them all at once, with revenue-at-risk estimates per row |
| **Segment Dashboard** | Portfolio-level risk view — cancellation rates and revenue exposure by segment, hotel, and season |

---

## Known Limitations

These are disclosed here proactively — a reviewer should not have to find them:

- **`room_type_changed` excluded as leakage risk.** `assigned_room_type` is typically only finalised at or near check-in. A cancelled booking may never reach the assignment step, making this feature a proxy for "booking was not cancelled" rather than a genuine ex-ante signal. It was the top-ranked feature in earlier, unaudited versions of this kind of model. Removing it is a deliberate trade-off documented in notebook 03 and `src/features.py`.

- **Cost ratio is illustrative.** The 3:1 ratio (missed cancellation vs false alarm) used for threshold selection is an assumption, not derived from real hotel financials. Replace with real numbers before operational use.

- **No temporal validation.** The train/test split is random (stratified), not time-based. In production, a model should only be evaluated on data it could not have seen during training.

- **Single-market dataset.** Both hotels are in Portugal. Performance on properties in other markets is unverified.

- **Library substitution.** The original project intended XGBoost/LightGBM/SHAP. This version uses `HistGradientBoostingClassifier` (sklearn) and permutation importance due to environment constraints. Architecture and feature set transfer directly if you reinstall those libraries.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/laxmigs24/Hotel-Revenue-Operations.git
cd Hotel-Revenue-Operations
pip install -r requirements.txt
```

### 2. Run the test suite

```bash
python3 tests/test_features.py
```

All 16 checks should pass. These tests guard specifically against the ordinal-encoding bug class (en-dash vs hyphen in category labels) that silently broke the original version of this project.

### 3. Rebuild the model (optional)

If the saved model bundle is incompatible with your local library versions:

```bash
python3 rebuild_model.py
```

This retrains the full pipeline on your machine and saves a compatible bundle.

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## SQL Layer

A standalone SQL database (`hotel_revenue.sql`) is included for analytical querying.

```bash
# Open interactively in SQLite
sqlite3 hotel_revenue.db
.read hotel_revenue.sql
```

The file demonstrates: normalised schema design with FK constraints, views as feature-engineering objects, CTEs, window functions (`LAG`, `RANK`, `PARTITION BY`, `SUM OVER`), lift calculations, confusion matrix logic in SQL, and YoY comparison via `LAG()`.

---

## Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML | scikit-learn (`HistGradientBoostingClassifier`, `OrdinalEncoder`, `ColumnTransformer`) |
| Explainability | `sklearn.inspection.permutation_importance` |
| App | Streamlit, Plotly |
| Serialisation | Joblib |
| Database | SQLite |
| Visualisation | Matplotlib, Seaborn |
| Testing | pytest (or standalone `python3 tests/test_features.py`) |
| CI | GitHub Actions (`.github/workflows/tests.yml`) |

---

## Dataset

**Hotel Booking Demand**
Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.

Available via [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and the original publication.

---

## License

MIT — see [LICENSE](LICENSE).
Dataset has its own terms; see the license file for details.

---

## Author

**Laxmi Gupte**
[LinkedIn](https://www.linkedin.com/in/laxmigupte24) · [GitHub](https://github.com/laxmigs24)
