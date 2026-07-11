# src/models/rf_model.py - IMPROVED: Hyperparams + Type Hints
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ..utils import logger
from .base_model import BaseModel


class RFModel(BaseModel):
    """
    Random Forest Regressor cho dự báo nhu cầu.
    
    Sử dụng sklearn RandomForestRegressor với hyperparameters
    được tối ưu cho bài toán retail forecasting.
    """

    def __init__(self) -> None:
        super().__init__()
        # IMPROVED: Tăng complexity để model đủ mạnh
        self.model = RandomForestRegressor(
            n_estimators=100,        # Tăng từ 30 → 100
            max_depth=12,            # Tăng từ 8 → 12
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',     # Thêm feature subsampling
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

    def train(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None) -> None:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        features = self._get_features(train_data)
        self.feature_names_ = features
        X_train = train_data[features].fillna(0)
        y_train = train_data['Weekly_Sales']
        # Trọng số ngày lễ x5 - khớp với metric WMAE
        w_train = self._holiday_weights(train_data)
        self.model.fit(X_train, y_train, sample_weight=w_train)
        logger.info("  ✓ Random Forest trained")

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Dự báo với Random Forest model."""
        features = self._get_features(test_data)
        X_test = test_data[features].fillna(0)
        preds = self.model.predict(X_test)
        return self._clip_negative(preds)