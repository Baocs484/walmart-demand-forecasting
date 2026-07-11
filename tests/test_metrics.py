# tests/test_metrics.py
"""
Unit tests for src.metrics.MetricsCalculator.

Covers both accuracy metrics (MAE, RMSE, MAPE, sMAPE, WMAE) and
business metrics (stockout rate, service level, costs).
"""

import sys
import os
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import MetricsCalculator


# ── accuracy metrics ────────────────────────────────────────────────────

class TestCalculateMetrics:
    """Tests for MetricsCalculator.calculate_metrics."""

    def test_calculate_metrics_perfect_prediction(self):
        """When actual == predicted, MAE and RMSE must be exactly 0."""
        actual = [1000, 2000, 3000, 4000, 5000]
        predicted = [1000, 2000, 3000, 4000, 5000]
        m = MetricsCalculator.calculate_metrics(actual, predicted)

        assert m["MAE"] == pytest.approx(0.0)
        assert m["RMSE"] == pytest.approx(0.0)

    def test_calculate_metrics_with_error(self):
        """Known actual/predicted → verify MAE and RMSE values."""
        actual = [1000, 2000, 3000]
        predicted = [1100, 1900, 3200]
        m = MetricsCalculator.calculate_metrics(actual, predicted)

        # MAE = mean(|100|, |100|, |200|) = 400/3 ≈ 133.33
        expected_mae = np.mean([100, 100, 200])
        assert m["MAE"] == pytest.approx(expected_mae, rel=1e-4)

        # RMSE = sqrt(mean(10000, 10000, 40000)) = sqrt(20000) ≈ 141.42
        expected_rmse = np.sqrt(np.mean([100**2, 100**2, 200**2]))
        assert m["RMSE"] == pytest.approx(expected_rmse, rel=1e-4)

    def test_calculate_metrics_with_weights(self):
        """With sample_weight provided, WMAE should differ from MAE."""
        actual = [1000, 2000, 3000]
        predicted = [1100, 1900, 3200]
        weights = np.array([5, 1, 1])  # heavily weight the first sample

        m = MetricsCalculator.calculate_metrics(actual, predicted, sample_weight=weights)

        # WMAE should differ from MAE because weights are non-uniform
        assert m["WMAE"] != pytest.approx(m["MAE"], rel=1e-4)

    def test_calculate_metrics_without_weights(self):
        """Without weights, WMAE must equal MAE."""
        actual = [500, 1500, 2500]
        predicted = [600, 1400, 2600]
        m = MetricsCalculator.calculate_metrics(actual, predicted)

        assert m["WMAE"] == pytest.approx(m["MAE"])

    def test_calculate_metrics_mape_filters_small_values(self):
        """MAPE should only consider values > 100; small actual values are filtered."""
        actual = [10, 20, 5000, 6000]      # first two are < 100
        predicted = [50, 50, 5500, 5500]

        m = MetricsCalculator.calculate_metrics(actual, predicted)

        # Manually: only indices 2 & 3 used
        # |(5000-5500)/5000| = 0.1, |(6000-5500)/6000| ≈ 0.0833
        expected_mape = np.mean([0.1, abs(500 / 6000)]) * 100
        assert m["MAPE"] == pytest.approx(expected_mape, rel=1e-3)

    def test_calculate_metrics_smape_bounded(self):
        """sMAPE should always be between 0 and 200."""
        rng = np.random.RandomState(0)
        actual = rng.uniform(100, 10000, size=200)
        predicted = rng.uniform(100, 10000, size=200)

        m = MetricsCalculator.calculate_metrics(actual, predicted)
        assert 0 <= m["sMAPE"] <= 200, f"sMAPE out of bounds: {m['sMAPE']}"


# ── business metrics ────────────────────────────────────────────────────

class TestBusinessMetrics:
    """Tests for MetricsCalculator.calculate_business_metrics."""

    def test_business_metrics_no_stockout(self):
        """If predicted > actual for all, stockout_rate must be 0."""
        actual = [100, 200, 300]
        predicted = [200, 300, 400]  # all higher than actual
        # With safety_factor=1.0 so predicted is used as-is
        bm = MetricsCalculator.calculate_business_metrics(
            actual, predicted, safety_factor=1.0
        )
        assert bm["Stockout_Rate"] == pytest.approx(0.0)

    def test_business_metrics_all_stockout(self):
        """predicted=0 with actual>0 → stockout_rate near 100 (considering safety_factor)."""
        actual = [1000, 2000, 3000]
        predicted = [0, 0, 0]
        bm = MetricsCalculator.calculate_business_metrics(
            actual, predicted, safety_factor=1.2
        )
        # Even with safety_factor = 1.2, 0 * 1.2 = 0 < actual → stockout 100%
        assert bm["Stockout_Rate"] == pytest.approx(100.0)

    def test_business_metrics_service_level_range(self):
        """Service level must be between 0 and 100."""
        rng = np.random.RandomState(7)
        actual = rng.uniform(100, 5000, size=100)
        predicted = rng.uniform(100, 5000, size=100)
        bm = MetricsCalculator.calculate_business_metrics(actual, predicted)

        assert 0 <= bm["Service_Level"] <= 100

    def test_business_metrics_total_cost(self):
        """Total cost = overstock_cost + stockout_cost."""
        actual = [1000, 2000, 3000]
        predicted = [1500, 1500, 3500]
        bm = MetricsCalculator.calculate_business_metrics(
            actual, predicted, safety_factor=1.0
        )
        assert bm["Total_Cost"] == pytest.approx(
            bm["Overstock_Cost"] + bm["Stockout_Cost"]
        )

    def test_business_metrics_safety_factor(self):
        """Higher safety_factor → lower (or equal) stockout rate."""
        actual = [500, 1000, 1500, 2000, 2500]
        predicted = [400, 900, 1600, 1800, 2600]

        bm_low = MetricsCalculator.calculate_business_metrics(
            actual, predicted, safety_factor=1.0
        )
        bm_high = MetricsCalculator.calculate_business_metrics(
            actual, predicted, safety_factor=2.0
        )

        assert bm_high["Stockout_Rate"] <= bm_low["Stockout_Rate"], (
            "Higher safety_factor should reduce stockout rate"
        )
