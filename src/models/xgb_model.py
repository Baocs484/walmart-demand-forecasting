# src/models/xgb_model.py - REFACTORED: Kế thừa BaseModel
import pandas as pd
from xgboost import XGBRegressor
from ..utils import logger
from .base_model import BaseModel


class XGBModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Cấu hình tối ưu cho bài toán hồi quy
        # eval_metric MAE khớp với metric WMAE
        self.model = XGBRegressor(
            n_estimators=2000,    # Nhiều cây + early stopping sẽ tự chọn số cây tối ưu
            learning_rate=0.05,   # Tốc độ học chậm -> chính xác hơn
            max_depth=6,          # Độ sâu để học pattern phức tạp
            subsample=0.8,        # 80% dữ liệu mỗi lần train để chống overfit
            colsample_bytree=0.8, # 80% features mỗi cây
            eval_metric='mae',
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )

    def train(self, train_data, validation_data=None):
        logger.info("Training Improved XGBoost model...")
        features = self._get_features(train_data)
        self.feature_names_ = features
        X_train = train_data[features].fillna(0)
        y_train = train_data['Weekly_Sales']

        # Trọng số ngày lễ x5 - khớp với metric WMAE
        w_train = self._holiday_weights(train_data)

        if validation_data is not None:
            X_val = validation_data[features].fillna(0)
            y_val = validation_data['Weekly_Sales']
            w_val = self._holiday_weights(validation_data)
            self.model.set_params(early_stopping_rounds=50)
            self.model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[w_val],
                verbose=False
            )
            logger.info(f"  ✓ XGBoost trained (best_iteration={self.model.best_iteration})")
        else:
            # Không có validation -> tắt early stopping, giới hạn số cây
            self.model.set_params(early_stopping_rounds=None, n_estimators=200)
            self.model.fit(X_train, y_train, sample_weight=w_train)
            logger.info("  ✓ XGBoost trained")

    def predict(self, test_data):
        features = self._get_features(test_data)
        X_test = test_data[features].fillna(0)
        preds = self.model.predict(X_test)
        return self._clip_negative(preds)
