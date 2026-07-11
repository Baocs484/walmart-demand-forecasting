# src/models/catboost_model.py - IMPROVED: Proper ImportError + Type Hints
import numpy as np
import pandas as pd
import time
from ..utils import logger
from .base_model import BaseModel

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    CatBoostRegressor = None
    Pool = None


class CatBoostModel(BaseModel):
    """
    CatBoost Regressor cho dự báo nhu cầu.
    
    CatBoost xử lý tốt categorical features tự động,
    phù hợp cho bài toán có Store/Dept/Type.
    
    Raises:
        ImportError: Nếu thư viện catboost chưa được cài đặt.
    """

    def __init__(self) -> None:
        super().__init__()
        # IMPROVED: Raise ImportError rõ ràng giống LightGBM
        if CatBoostRegressor is None:
            raise ImportError(
                "Thư viện CatBoost chưa được cài đặt. "
                "Hãy chạy: pip install catboost"
            )

        # Loss MAE khớp với metric WMAE; nhiều iterations + early stopping
        self.model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            loss_function='MAE',
            random_seed=42,
            verbose=0,
            allow_writing_files=False
        )

        # Các cột cần chuyển sang dạng Category
        self.cat_features_names = ['Store', 'Dept', 'IsHoliday', 'IsWeekend', 'Cluster']

    def train(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None) -> None:
        """Train CatBoost model."""
        logger.info("Training CatBoost model...")
        start = time.time()

        features = self._get_features(train_data)
        self.feature_names_ = features

        # Tạo bản sao độc lập
        X_train = train_data[features].copy()
        y_train = train_data['Weekly_Sales']

        # Trọng số ngày lễ x5 - khớp với metric WMAE
        w_train = self._holiday_weights(train_data)

        # Tìm index của các cột category
        cat_indices = []
        for col_name in self.cat_features_names:
            if col_name in X_train.columns:
                X_train[col_name] = X_train[col_name].astype(str)
                cat_indices.append(X_train.columns.get_loc(col_name))

        train_pool = Pool(X_train, y_train, cat_features=cat_indices, weight=w_train)

        # Xử lý Validation Data (Pool có trọng số để early stopping theo weighted MAE)
        eval_pool = None
        if validation_data is not None:
            X_val = validation_data[features].copy()
            y_val = validation_data['Weekly_Sales']
            w_val = self._holiday_weights(validation_data)

            for col_name in self.cat_features_names:
                if col_name in X_val.columns:
                    X_val[col_name] = X_val[col_name].astype(str)

            eval_pool = Pool(X_val, y_val, cat_features=cat_indices, weight=w_val)

        # Train model
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=50,
            use_best_model=eval_pool is not None
        )

        elapsed = time.time() - start
        logger.info(f"  ✓ CatBoost training complete in {elapsed:.1f}s")

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Dự báo với CatBoost model."""
        features = self._get_features(test_data)
        X_test = test_data[features].copy()

        # Xử lý category giống lúc train
        for col_name in self.cat_features_names:
            if col_name in X_test.columns:
                X_test[col_name] = X_test[col_name].astype(str)

        preds = self.model.predict(X_test)
        return self._clip_negative(preds)