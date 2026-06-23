"""
Tests for src/features.py.

These tests exist specifically to prevent regressions of the bug that
affected the original version of this project: ordinal category lists
(used by sklearn's OrdinalEncoder) silently drifting out of sync with the
string labels actually produced by the bucket functions. That bug caused
every row to be encoded as "unknown" (-1) without raising any error.

Run with:  pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    # Minimal stand-in so `@pytest.fixture` decorators below don't crash
    # on import in environments without pytest. The standalone runner at
    # the bottom of this file calls the underlying test functions directly
    # and never goes through the fixture machinery, so this stub is only
    # here to make the module importable.
    class _FixtureStub:
        def __call__(self, fn):
            return fn

    class _PytestStub:
        fixture = _FixtureStub()

        @staticmethod
        def main(*args, **kwargs):
            pass

    pytest = _PytestStub()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as feat


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A small synthetic raw booking dataframe covering edge cases:
    zero-night stay, negative ADR, last-minute lead time, long lead time,
    peak and non-peak season, repeat guest with/without deposit.
    """
    return pd.DataFrame({
        "hotel": ["City Hotel", "Resort Hotel", "City Hotel", "Resort Hotel", "City Hotel"],
        "is_canceled": [0, 1, 0, 1, 0],
        "lead_time": [5, 45, 120, 400, 0],
        "arrival_date_month": ["July", "January", "March", "November", "June"],
        "stays_in_weekend_nights": [0, 1, 2, 0, 0],
        "stays_in_week_nights": [0, 2, 3, 5, 0],
        "adults": [2, 1, 2, 3, 1],
        "children": [0, 0, 1, 0, 0],
        "babies": [0, 0, 0, 0, 0],
        "adr": [-6.38, 45.0, 110.0, 320.0, 0.0],
        "previous_cancellations": [0, 1, 0, 0, 0],
        "previous_bookings_not_canceled": [0, 0, 2, 0, 0],
        "booking_changes": [0, 1, 0, 0, 0],
        "total_of_special_requests": [0, 2, 1, 0, 0],
        "days_in_waiting_list": [0, 0, 0, 10, 0],
        "is_repeated_guest": [0, 1, 1, 0, 0],
        "deposit_type": ["No Deposit", "Refundable", "Non Refund", "No Deposit", "No Deposit"],
        "market_segment": ["Online TA", "Direct", "Corporate", "Groups", "Online TA"],
        "distribution_channel": ["TA/TO", "Direct", "Corporate", "TA/TO", "TA/TO"],
        "customer_type": ["Transient", "Contract", "Transient", "Group", "Transient"],
        "reserved_room_type": ["A", "B", "A", "C", "A"],
        "assigned_room_type": ["A", "B", "C", "C", "A"],
        "required_car_parking_spaces": [0, 1, 0, 0, 0],
    })


@pytest.fixture
def engineered_df(sample_raw_df) -> pd.DataFrame:
    return feat.engineer_all_features(sample_raw_df)


# ============================================================
# Bug-regression tests: ordinal category alignment
# ============================================================

class TestOrdinalCategoryAlignment:
    """The exact bug class that broke the original project."""

    def test_lead_time_bucket_labels_match_order_exactly(self, engineered_df):
        produced = set(engineered_df["lead_time_bucket"].dropna().unique())
        expected = set(feat.LEAD_TIME_ORDER)
        unmatched = produced - expected
        assert not unmatched, (
            f"lead_time_bucket produced labels not present in LEAD_TIME_ORDER: "
            f"{unmatched}. This is the exact bug that caused silent -1 encoding."
        )

    def test_adr_bucket_labels_match_order_exactly(self, engineered_df):
        produced = set(engineered_df["adr_bucket"].dropna().unique())
        expected = set(feat.ADR_ORDER)
        unmatched = produced - expected
        assert not unmatched, (
            f"adr_bucket produced labels not present in ADR_ORDER: {unmatched}"
        )

    def test_stay_length_category_labels_match_order_exactly(self, engineered_df):
        produced = set(engineered_df["stay_length_category"].dropna().unique())
        expected = set(feat.STAY_LENGTH_ORDER)
        unmatched = produced - expected
        assert not unmatched, (
            f"stay_length_category produced labels not present in STAY_LENGTH_ORDER: {unmatched}"
        )

    def test_no_dash_character_mismatch(self):
        """Defends specifically against en-dash vs hyphen drift."""
        for label in feat.LEAD_TIME_ORDER + feat.ADR_ORDER:
            assert "\u2013" not in label, (
                f"Found an en-dash (–) in category label {label!r}. "
                "Bucket labels must use a plain hyphen (-)."
            )

    def test_ordinal_encoder_does_not_silently_drop_known_categories(self, engineered_df):
        """End-to-end check using the real sklearn OrdinalEncoder, on the
        real data produced by engineer_all_features — this is the test that
        would have caught the original production bug.
        """
        from sklearn.preprocessing import OrdinalEncoder

        encoder = OrdinalEncoder(
            categories=feat.ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        encoded = encoder.fit_transform(engineered_df[feat.ORDINAL_FEATURES])

        # If the bug were present, every row would be -1 for the affected
        # column. Assert that at least one row encodes to something other
        # than -1 for each ordinal column.
        for i, col in enumerate(feat.ORDINAL_FEATURES):
            assert (encoded[:, i] != -1).any(), (
                f"All rows for ordinal feature '{col}' encoded to -1 (unknown). "
                "This means the category list does not match the produced labels."
            )


# ============================================================
# Feature correctness tests
# ============================================================

class TestFeatureCorrectness:

    def test_total_nights_is_sum_of_weekend_and_week_nights(self, engineered_df):
        expected = (
            engineered_df["stays_in_weekend_nights"] + engineered_df["stays_in_week_nights"]
        )
        pd.testing.assert_series_equal(
            engineered_df["total_nights"], expected, check_names=False
        )

    def test_negative_adr_flagged_and_bucketed_as_invalid(self, engineered_df):
        invalid_row = engineered_df[engineered_df["adr"] < 0]
        assert len(invalid_row) == 1
        assert invalid_row["has_invalid_adr"].iloc[0] == 1
        assert invalid_row["adr_bucket"].iloc[0] == "Invalid ADR"

    def test_zero_night_stay_flag(self, engineered_df):
        zero_night = engineered_df[engineered_df["total_nights"] == 0]
        assert (zero_night["is_zero_night_stay"] == 1).all()
        assert (zero_night["stay_length_category"] == "Zero-night stay").all()

    def test_peak_season_matches_constant_definition(self, engineered_df):
        for _, row in engineered_df.iterrows():
            expected = 1 if row["arrival_date_month"] in feat.PEAK_SEASON_MONTHS else 0
            assert row["is_peak_season"] == expected

    def test_no_adr_bucket_is_null(self, engineered_df):
        assert engineered_df["adr_bucket"].isna().sum() == 0

    def test_estimated_booking_value_formula(self, engineered_df):
        expected = engineered_df["adr"] * engineered_df["total_nights"]
        pd.testing.assert_series_equal(
            engineered_df["estimated_booking_value"], expected, check_names=False
        )

    def test_previous_cancellation_rate_handles_zero_total(self, engineered_df):
        # Rows with 0 previous bookings should have rate 0, not NaN/inf
        no_history = engineered_df[engineered_df["previous_total_bookings"] == 0]
        assert (no_history["previous_cancellation_rate"] == 0).all()
        assert not engineered_df["previous_cancellation_rate"].isna().any()
        assert np.isfinite(engineered_df["previous_cancellation_rate"]).all()

    def test_children_nan_imputed_to_zero(self):
        """The raw dataset has a handful of rows with a missing `children`
        value. The original project never handled this, which broke any
        model (e.g. LogisticRegression) that doesn't natively accept NaN.
        """
        raw = pd.DataFrame({
            "hotel": ["City Hotel", "City Hotel"],
            "is_canceled": [0, 1],
            "lead_time": [10, 20],
            "arrival_date_month": ["July", "August"],
            "stays_in_weekend_nights": [1, 0],
            "stays_in_week_nights": [2, 1],
            "adults": [2, 1],
            "children": [np.nan, 1.0],
            "babies": [0, 0],
            "adr": [100.0, 80.0],
            "previous_cancellations": [0, 0],
            "previous_bookings_not_canceled": [0, 0],
            "booking_changes": [0, 0],
            "total_of_special_requests": [0, 0],
            "days_in_waiting_list": [0, 0],
            "is_repeated_guest": [0, 0],
            "deposit_type": ["No Deposit", "No Deposit"],
            "market_segment": ["Online TA", "Online TA"],
            "distribution_channel": ["TA/TO", "TA/TO"],
            "customer_type": ["Transient", "Transient"],
            "reserved_room_type": ["A", "A"],
            "assigned_room_type": ["A", "A"],
            "required_car_parking_spaces": [0, 0],
        })
        engineered = feat.engineer_all_features(raw)
        assert engineered["children"].isna().sum() == 0
        assert engineered["children"].iloc[0] == 0


# ============================================================
# Leakage guard tests
# ============================================================

class TestLeakageGuards:

    def test_leakage_columns_not_in_model_features(self):
        for col in feat.LEAKAGE_COLUMNS:
            assert col not in feat.MODEL_FEATURES

    def test_room_type_changed_excluded_from_model_features(self):
        """room_type_changed is the highest-importance feature in the
        original model and is suspected of leaking post-booking
        information (assigned_room_type is typically only known at/near
        check-in). It must not be a model input.
        """
        assert "room_type_changed" not in feat.MODEL_FEATURES
        assert "reserved_room_type" not in feat.MODEL_FEATURES
        assert "assigned_room_type" not in feat.MODEL_FEATURES

    def test_model_features_have_no_duplicates(self):
        assert len(feat.MODEL_FEATURES) == len(set(feat.MODEL_FEATURES))


# (entry point is unified at the bottom of this file)


# ============================================================
# Standalone runner (no pytest required)
# ============================================================
#
# This block lets the test file double as a plain `python3 test_features.py`
# script in environments without pytest installed. It instantiates each
# Test* class and calls every test_* method manually, raising on the first
# failure. CI (.github/workflows/tests.yml) uses real pytest; this is a
# convenience fallback for local sanity checks.

def _run_standalone() -> None:
    raw = pd.DataFrame({
        "hotel": ["City Hotel", "Resort Hotel", "City Hotel", "Resort Hotel", "City Hotel"],
        "is_canceled": [0, 1, 0, 1, 0],
        "lead_time": [5, 45, 120, 400, 0],
        "arrival_date_month": ["July", "January", "March", "November", "June"],
        "stays_in_weekend_nights": [0, 1, 2, 0, 0],
        "stays_in_week_nights": [0, 2, 3, 5, 0],
        "adults": [2, 1, 2, 3, 1],
        "children": [0, 0, 1, 0, 0],
        "babies": [0, 0, 0, 0, 0],
        "adr": [-6.38, 45.0, 110.0, 320.0, 0.0],
        "previous_cancellations": [0, 1, 0, 0, 0],
        "previous_bookings_not_canceled": [0, 0, 2, 0, 0],
        "booking_changes": [0, 1, 0, 0, 0],
        "total_of_special_requests": [0, 2, 1, 0, 0],
        "days_in_waiting_list": [0, 0, 0, 10, 0],
        "is_repeated_guest": [0, 1, 1, 0, 0],
        "deposit_type": ["No Deposit", "Refundable", "Non Refund", "No Deposit", "No Deposit"],
        "market_segment": ["Online TA", "Direct", "Corporate", "Groups", "Online TA"],
        "distribution_channel": ["TA/TO", "Direct", "Corporate", "TA/TO", "TA/TO"],
        "customer_type": ["Transient", "Contract", "Transient", "Group", "Transient"],
        "reserved_room_type": ["A", "B", "A", "C", "A"],
        "assigned_room_type": ["A", "B", "C", "C", "A"],
        "required_car_parking_spaces": [0, 1, 0, 0, 0],
    })
    eng = feat.engineer_all_features(raw)

    checks = [
        TestOrdinalCategoryAlignment().test_lead_time_bucket_labels_match_order_exactly,
        TestOrdinalCategoryAlignment().test_adr_bucket_labels_match_order_exactly,
        TestOrdinalCategoryAlignment().test_stay_length_category_labels_match_order_exactly,
        TestOrdinalCategoryAlignment().test_no_dash_character_mismatch,
        TestOrdinalCategoryAlignment().test_ordinal_encoder_does_not_silently_drop_known_categories,
        TestFeatureCorrectness().test_total_nights_is_sum_of_weekend_and_week_nights,
        TestFeatureCorrectness().test_negative_adr_flagged_and_bucketed_as_invalid,
        TestFeatureCorrectness().test_zero_night_stay_flag,
        TestFeatureCorrectness().test_peak_season_matches_constant_definition,
        TestFeatureCorrectness().test_no_adr_bucket_is_null,
        TestFeatureCorrectness().test_estimated_booking_value_formula,
        TestFeatureCorrectness().test_previous_cancellation_rate_handles_zero_total,
        TestFeatureCorrectness().test_children_nan_imputed_to_zero,
        TestLeakageGuards().test_leakage_columns_not_in_model_features,
        TestLeakageGuards().test_room_type_changed_excluded_from_model_features,
        TestLeakageGuards().test_model_features_have_no_duplicates,
    ]

    passed = 0
    for check in checks:
        import inspect
        sig = inspect.signature(check)
        args = {}
        if "engineered_df" in sig.parameters:
            args["engineered_df"] = eng
        if "sample_raw_df" in sig.parameters:
            args["sample_raw_df"] = raw
        check(**args)
        print(f"PASS: {check.__qualname__}")
        passed += 1
    print(f"\n{passed}/{len(checks)} checks passed.")


if __name__ == "__main__":
    if _HAS_PYTEST:
        raise SystemExit(pytest.main([__file__, "-v"]))
    else:
        _run_standalone()
