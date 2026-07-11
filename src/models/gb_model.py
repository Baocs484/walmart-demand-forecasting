# src/models/gb_model.py - IMPROVED: Hyperparams + Type Hints
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from ..utils import logger
from .base_model import BaseModel


class GBModel(BaseModel):
    """
    Gradient Boosting Regressor cho dự báo nhu cầu.
    
    Sử dụng sklearn GradientBoostingRegressor với hyperparameters
    được tối ưu cho bài toán time series retail.
    """

    def __init__(self) -> None:
        super().__init__()
        # IMPROVED: Tăng complexity để model đủ mạnh
        self.model = GradientBoostingRegressor(
            n_estimators=100,        # Tăng từ 20 → 100
            learning_rate=0.05,      # Thêm learning rate thấp
            max_depth=6,             # Tăng từ 4 → 6
            subsample=0.8,           # Thêm subsampling chống overfit
            min_samples_split=5,     # Thêm regularization
            min_samples_leaf=3,      # Thêm regularization
            random_state=42,
            verbose=0
        )

    def train(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None) -> None:
        """Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting model...")
        features = self._get_features(train_data)
        self.feature_names_ = features
        X_train = train_data[features].fillna(0)
        y_train = train_data['Weekly_Sales']
        # Trọng số ngày lễ x5 - khớp với metric WMAE
        w_train = self._holiday_weights(train_data)
        self.model.fit(X_train, y_train, sample_weight=w_train)
        logger.info("  ✓ Gradient Boosting trained")

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Dự báo với Gradient Boosting model."""
        features = self._get_features(test_data)
        X_test = test_data[features].fillna(0)
        preds = self.model.predict(X_test)
        return self._clip_negative(preds)