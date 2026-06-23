# Hotel Revenue Operations — Cancellation Risk Assistant

A cancellation-risk model and Streamlit decision-support app for hotel revenue managers,
built on the public [Hotel Booking Demand dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191)
(Antonio, de Almeida & Nunes, 2019).

**This is a corrected, audited version of the original project.** A full review surfaced
several real bugs and a leakage risk in the original notebooks and app — all are fixed here,
verified by execution, and documented transparently below rather than hidden. See
[`CHANGELOG.md`](CHANGELOG.md) for the full list of what changed and why.

---

## What this project does

- Predicts the probability that a hotel booking will be cancelled before arrival, using
  information available at booking time (lead time, deposit type, prior guest history,
  price, stay length, etc.)
- Surfaces a cost-weighted risk threshold, not just a statistically "optimal" one — missing
  a real cancellation is treated as costlier than a false alarm, and the threshold reflects
  that
- Ships as an interactive Streamlit app with four modes: manual single-booking scoring,
  explaining an existing booking, batch CSV scoring, and a portfolio-level risk dashboard

## Results (current model)

| Metric | Value |
|---|---|
| Test ROC-AUC | **0.840** |
| Model | Gradient Boosting (`HistGradientBoostingClassifier`) |
| Decision threshold | **0.26** (cost-weighted; missed cancellation assumed 3x costlier than a false alarm) |
| Features used | 41 (after excluding 3 leakage-risk columns — see below) |

This AUC is honestly lower than what an unaudited version of this kind of project might
report, for two reasons: (1) the model no longer uses `room_type_changed`, which was the
top-ranked feature in the original version but is a plausible leakage signal (see below),
and (2) two ordinal features (`lead_time_bucket`, `adr_bucket`) are now correctly encoded,
whereas a bug in the original version caused them to silently carry no information at all.
A lower, trustworthy number beats a higher, broken one.

## Project structure

```
.
├── notebooks/
│   ├── 01_data_audit.ipynb           # load, dedup, audit raw data
│   ├── 02_eda.ipynb                  # exploratory analysis & business insights
│   ├── 03_feature_eng.ipynb          # canonical feature engineering (via src/features.py)
│   └── 04_modeling_evaluation.ipynb  # model comparison, cost-weighted threshold, model card
├── src/
│   └── features.py                   # SINGLE SOURCE OF TRUTH for all feature logic —
│                                      # imported by notebooks AND the app, so they can't drift apart
├── tests/
│   └── test_features.py              # regression tests guarding against the original bug class
├── app/
│   └── streamlit_app.py              # the decision-support app
├── models/
│   └── cancellation_model.joblib     # trained pipeline + threshold + metadata bundle
├── data/                             # raw + intermediate + processed CSVs
├── .github/workflows/tests.yml       # CI: runs tests + executes all notebooks on every push
├── requirements.txt
├── LICENSE
└── CHANGELOG.md                      # what was fixed, and why
```

## Getting started

```bash
git clone <repo-url>
cd Hotel-Revenue-Operations
pip install -r requirements.txt

# Run the test suite (fast, no notebook execution needed)
pytest tests/ -v

# Reproduce the full pipeline from raw data
jupyter notebook notebooks/01_data_audit.ipynb   # then 02, 03, 04 in order

# Launch the app (requires models/cancellation_model.joblib to exist —
# produced by running notebook 04)
streamlit run app/streamlit_app.py
```

## Known limitations (read before trusting this in production)

This section exists so a reviewer doesn't have to find these on their own — they're
disclosed here, with reasoning, the same way they're documented in
`04_modeling_evaluation.ipynb`'s model card section.

- **`room_type_changed` is excluded as a leakage risk.** It compares `reserved_room_type`
  to `assigned_room_type`. In most hotel systems, the assigned room is only finalized by
  staff at or near check-in — a cancelled booking may never reach that step. This makes the
  feature a plausible proxy for "this booking was not cancelled" rather than a genuine,
  booking-time risk signal, and it isn't something a manager could honestly provide when
  scoring a new, not-yet-arrived booking. It was the #1 ranked feature in earlier,
  unaudited iterations of this kind of model — removing it is a deliberate, documented
  trade-off, not an oversight.
- **The cost ratio behind the decision threshold (3:1, missed cancellation : false alarm)
  is an illustrative assumption**, not derived from real hotel financials. Replace it with
  actual numbers from a real property before using the threshold operationally — see the
  "Cost-Weighted Threshold Selection" section of notebook 04.
- **No temporal validation.** The train/test split is a random stratified split, not a
  time-based one. In production, a booking made in January should only be scored by a model
  trained on data available before January. A time-based split is a natural next iteration.
- **Single-market generalization is untested.** The dataset covers two hotels (one City,
  one Resort) in one market. Performance elsewhere is unverified.
- **Library substitution.** This version was built and verified in an environment without
  XGBoost, LightGBM, or SHAP available, so it uses scikit-learn's
  `HistGradientBoostingClassifier` (algorithmically comparable) and permutation importance
  in place of SHAP for feature importance. The app still supports SHAP if it's installed
  locally (`pip install shap`) and degrades gracefully if not.

## Data

This project uses the public **Hotel Booking Demand dataset**
(Antonio, Almeida & Nunes, 2019), available via Kaggle and the original publication. It
covers ~119K bookings across a City Hotel and a Resort Hotel in Portugal (2015–2017).
The raw CSV is included under `data/` for reproducibility.

## License

MIT — see [`LICENSE`](LICENSE). The dataset itself has its own terms; see the license file
for details.
