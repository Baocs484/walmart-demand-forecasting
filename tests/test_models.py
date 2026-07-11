# tests/test_models.py
"""
Unit tests for all model classes under src.models.

Uses small synthetic data from conftest fixtures to keep tests fast.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.base_model import BaseModel
from src.models.baseline_model import BaselineModel
from src.models.rf_model import RFModel
from src.models.xgb_model import XGBModel
from src.models.gb_model import GBModel
from src.models.ensemble_model import EnsembleModel
from src.store_clustering import StoreClustering


# ── BaseModel (abstract, tested via concrete helpers) ───────────────────

class TestBaseModel:
    """Tests for shared BaseModel utility methods."""

    def test_base_model_get_features(self, processed_df):
        """_get_features should return only columns that actually exist in the df."""
        # Use a concrete subclass to access _get_features
        model = GBModel()
        features = model._get_features(processed_df)

        assert isinstance(features, list)
        assert len(features) > 0, "Expected at least some features to match"

        for f in features:
            assert f in processed_df.columns, f"Feature {f} not in DataFrame"

    def test_base_model_clip_negative(self):
        """_clip_negative should set all negative values to 0."""
        model = GBModel()
        preds = np.array([100, -50, 200, -1, 0, 300])
        clipped = model._clip_negative(preds)

        assert (clipped >= 0).all(), "Negative values were not clipped to 0"
        assert clipped[0] == 100
        assert clipped[1] == 0
        assert clipped[3] == 0
        assert clipped[4] == 0


# ── Baseline Model ──────────────────────────────────────────────────────

class TestBaselineModel:
    """Tests for the BaselineModel (historical average with seasonal adjustment)."""

    def test_baseline_model_train_predict(self, train_val_test_split):
        """Train and predict; output shape must match test data length."""
        splits = train_val_test_split
        model = BaselineModel()
        model.train(splits["train"])
        preds = model.predict(splits["test"])

        assert len(preds) == len(splits["test"]), (
            f"Prediction length {len(preds)} != test length {len(splits['test'])}"
        )

    def test_baseline_model_predictions_non_negative(self, train_val_test_split):
        """All baseline predictions must be >= 0."""
        splits = train_val_test_split
        model = BaselineModel()
        model.train(splits["train"])
        preds = model.predict(splits["test"])

        assert (preds >= 0).all(), "Baseline predictions contain negative values"


# ── Random Forest Model ─────────────────────────────────────────────────

class TestRFModel:
    """Tests for the Random Forest model."""

    def test_rf_model_train_predict(self, train_val_test_split):
        """Train and predict; output shape must match test data length."""
        splits = train_val_test_split
        model = RFModel()
        model.train(splits["train"])
        preds = model.predict(splits["test"])

        assert len(preds) == len(splits["test"])

    def test_rf_model_predictions_non_negative(self, train_val_test_split):
        """All RF predictions must be >= 0."""
        splits = train_val_test_split
        model = RFModel()
        model.train(splits["train"])
        preds = model.predict(splits["test"])

        assert (preds >= 0).all(), "RF predictions contain negative values"


# ── XGBoost Model ───────────────────────────────────────────────────────

class TestXGBModel:
    """Tests for the XGBoost model."""

    def test_xgb_model_train_predict(self, train_val_test_split):
        """Train and predict; output shape must match test data length."""
        splits = train_val_test_split
        model = XGBModel()
        model.train(splits["train"])
        preds = model.predict(splits["test"])

        assert len(preds) == len(splits["test"])

    def test_xgb_model_with_validation(self, train_val_test_split):
        """Train with validation data (early stopping path); should still produce output."""
        splits = train_val_test_split
        model = XGBModel()
        model.train(splits["train"], validation_data=splits["val"])
        preds = model.predict(splits["test"])

        assert len(preds) == len(splits["test"])
        assert (preds >= 0).all()


# ── Gradient Boosting Model ─────────────────────────────────────────────

class TestGBModel:
    """Tests for the sklearn GradientBoosting model."""

    def test_gb_model_train_predict(self, train_val_test_split):
        """Train and predict; output shape must match test data length."""
        splits = train_val_test_split
        model = GBModel()
        model.train(splits["train"])
        preds = model.predict(splits["test"])

        assert len(preds) == len(splits["test"])


# ── Ensemble Model ──────────────────────────────────────────────────────

class TestEnsembleModel:
    """Tests for the Ensemble model (GB + optional LightGBM)."""

    def test_ensemble_model_train_predict(self, train_val_test_split):
        """Train ensemble and predict; output shape must match test data length.

        If LightGBM is not installed the ensemble gracefully falls back to
        GradientBoosting-only mode.
        """
        splits = train_val_test_split
        model = EnsembleModel()
        model.train(splits["train"], validation_data=splits["val"])
        preds = model.predict(splits["test"])

        assert len(preds) == len(splits["test"])
        assert (preds >= 0).all()


# ── Store Clustering ────────────────────────────────────────────────────

class TestStoreClustering:
    """Tests for StoreClustering (KMeans-based store grouping)."""

    @pytest.fixture
    def fitted_clustering(self, sample_df):
        """Fit StoreClustering on the raw sample_df and return it."""
        sc = StoreClustering(n_clusters=2)
        sc.fit(sample_df)
        return sc

    def test_store_clustering_fit(self, sample_df, fitted_clustering):
        """Clustering should assign a cluster to every store."""
        sc = fitted_clustering
        unique_stores = sample_df["Store"].unique()

        for store in unique_stores:
            assert store in sc.store_clusters, f"Store {store} has no cluster"

    def test_store_clustering_get_cluster(self, fitted_clustering):
        """get_cluster should return a valid integer cluster id."""
        sc = fitted_clustering
        for store_id in sc.store_clusters:
            cluster = sc.get_cluster(store_id)
            assert isinstance(cluster, (int, np.integer))
            assert cluster >= 0

    def test_store_clustering_find_similar_stores(self, fitted_clustering):
        """find_similar_stores should return a list of store ids."""
        sc = fitted_clustering
        first_store = list(sc.store_clusters.keys())[0]
        similar = sc.find_similar_stores(first_store, top_k=2)

        assert isinstance(similar, list)
        # The returned ids should be actual store ids, not the query store
        for s in similar:
            assert s != first_store or len(sc.store_clusters) == 1
