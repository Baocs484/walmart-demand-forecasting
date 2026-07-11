# tests/test_data_processor.py
"""
Unit tests for src.data_processor.DataProcessor.

All tests use synthetically generated data (via conftest fixtures) so
the real Walmart dataset is never required.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processor import DataProcessor


# ── helpers ──────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    """Fresh DataProcessor instance."""
    return DataProcessor()


# ── test_load_and_process_data ──────────────────────────────────────────

class TestLoadAndProcessData:
    """Tests around the top-level load_and_process_data pipeline."""

    def test_load_and_process_data(self, sample_df, processor, tmp_path):
        """Save sample_df to CSV, load it back, verify more features are created."""
        csv_path = tmp_path / "walmart_test.csv"
        sample_df.to_csv(csv_path, index=False)

        result = processor.load_and_process_data(str(csv_path))

        # Feature engineering should have added columns
        assert result.shape[1] > sample_df.shape[1], (
            f"Expected more columns after processing; got {result.shape[1]} "
            f"vs original {sample_df.shape[1]}"
        )
        # Row count may shrink (inf-row removal) but should not grow
        assert result.shape[0] <= sample_df.shape[0]


# ── test_create_features ────────────────────────────────────────────────

class TestCreateFeatures:
    """Tests that verify individual feature groups produced by create_features."""

    def test_create_features_adds_time_features(self, sample_df, processor):
        """Verify Month, Quarter, DayOfWeek, WeekOfYear, IsWeekend columns exist."""
        df = processor.create_features(sample_df.copy())
        for col in ["Month", "Quarter", "DayOfWeek", "WeekOfYear", "IsWeekend"]:
            assert col in df.columns, f"Missing time feature: {col}"

    def test_create_features_adds_lag_features(self, sample_df, processor):
        """Verify Sales_lag_1, Sales_lag_4, Sales_lag_52 exist."""
        df = processor.create_features(sample_df.copy())
        for col in ["Sales_lag_1", "Sales_lag_4", "Sales_lag_52"]:
            assert col in df.columns, f"Missing lag feature: {col}"

    def test_create_features_adds_rolling_features(self, sample_df, processor):
        """Verify Sales_roll_mean_4, Sales_roll_mean_13, Sales_roll_std_4 exist."""
        df = processor.create_features(sample_df.copy())
        for col in ["Sales_roll_mean_4", "Sales_roll_mean_13", "Sales_roll_std_4"]:
            assert col in df.columns, f"Missing rolling feature: {col}"

    def test_create_features_adds_cyclical_encoding(self, sample_df, processor):
        """Verify Month_sin, Month_cos, etc. exist and values are in [-1, 1]."""
        df = processor.create_features(sample_df.copy())
        cyclic_cols = ["Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos"]
        for col in cyclic_cols:
            assert col in df.columns, f"Missing cyclical column: {col}"
            assert df[col].min() >= -1.0 - 1e-9, f"{col} has values < -1"
            assert df[col].max() <= 1.0 + 1e-9, f"{col} has values > 1"

    def test_create_features_adds_markdown_features(self, sample_df, processor):
        """Verify Markdown_Total and Has_Markdown columns exist."""
        df = processor.create_features(sample_df.copy())
        assert "Markdown_Total" in df.columns
        assert "Has_Markdown" in df.columns
        # Has_Markdown must be 0/1
        assert set(df["Has_Markdown"].unique()).issubset({0, 1})

    def test_create_features_adds_regional_features(self, sample_df, processor):
        """Verify Region and Regional_Sales_Lag columns are created."""
        df = processor.create_features(sample_df.copy())
        assert "Region" in df.columns
        assert "Regional_Sales_Lag" in df.columns

    def test_create_features_adds_dept_specific_features(self, sample_df, processor):
        """Verify department-specific features exist."""
        df = processor.create_features(sample_df.copy())
        dept_cols = [
            "Dept_Avg_Sales",
            "Sales_vs_Dept_Avg",
            "Dept_CV",
            "Dept_Zero_Freq",
            "Store_Rank_in_Dept",
            "Dept_Size_Indicator",
        ]
        for col in dept_cols:
            assert col in df.columns, f"Missing dept feature: {col}"


# ── test_handle_missing_values ──────────────────────────────────────────

class TestHandleMissingValues:
    """Tests for the NaN / missing-value handling pipeline."""

    def test_handle_missing_values_no_nan(self, processed_df):
        """After full processing, lag/rolling columns must have no NaN."""
        lag_roll_cols = [
            c for c in processed_df.columns
            if "lag" in c.lower() or "roll" in c.lower()
        ]
        for col in lag_roll_cols:
            nan_count = processed_df[col].isna().sum()
            assert nan_count == 0, f"{col} still has {nan_count} NaN values"


# ── test_validate_and_clean ─────────────────────────────────────────────

class TestValidateAndClean:
    """Tests for the final data validation / cleaning step."""

    def test_validate_and_clean_no_inf(self, processed_df):
        """After processing, no Inf values should remain in float columns."""
        float_cols = processed_df.select_dtypes(include=[np.floating]).columns
        if len(float_cols) > 0:
            inf_count = np.isinf(processed_df[float_cols].values).sum()
            assert inf_count == 0, f"Found {inf_count} Inf values"

    def test_validate_and_clean_clips_extreme_values(self, sample_df, processor):
        """Verify extreme values are clipped by _validate_and_clean."""
        df = sample_df.copy()
        # Inject an extreme Temperature value
        df.loc[0, "Temperature"] = 1e10

        df = processor.create_features(df)
        df = processor._handle_missing_values(df)
        df = processor._validate_and_clean(df)

        # After clipping to the 1st–99th percentile the extreme should be gone
        assert df["Temperature"].max() < 1e10, "Extreme Temperature was not clipped"


# ── test type coercion ──────────────────────────────────────────────────

class TestTypeCoercion:
    """Tests ensuring specific columns have the expected dtype after processing."""

    def test_isholiday_is_numeric(self, processed_df):
        """After processing, IsHoliday should be an integer type (0/1)."""
        assert pd.api.types.is_integer_dtype(processed_df["IsHoliday"]), (
            f"IsHoliday dtype is {processed_df['IsHoliday'].dtype}, expected int"
        )

    def test_weekofyear_is_int(self, processed_df):
        """After processing, WeekOfYear should be int64."""
        assert processed_df["WeekOfYear"].dtype == np.int64 or pd.api.types.is_integer_dtype(processed_df["WeekOfYear"]), (
            f"WeekOfYear dtype is {processed_df['WeekOfYear'].dtype}, expected int64"
        )
