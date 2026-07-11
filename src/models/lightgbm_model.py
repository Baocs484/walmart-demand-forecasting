# src/models/lightgbm_model.py - REFACTORED: Kế thừa BaseModel
import time
from ..utils import logger
from .base_model import BaseModel

# Import LightGBM an toàn
try:
    from lightgbm import LGBMRegressor, early_stopping
except ImportError:
    LGBMRegressor = None
    early_stopping = None


class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__()
        if LGBMRegressor is None:
            raise ImportError("Thư viện LightGBM chưa được cài đặt. Hãy chạy: pip install lightgbm")

        # Cấu hình tối ưu cho Time Series
        # objective L1 (MAE) khớp với metric WMAE hơn L2 mặc định
        self.model = LGBMRegressor(
            n_estimators=2000,        # Nhiều cây + early stopping sẽ tự chọn số cây tối ưu
            learning_rate=0.05,       # Tốc độ học chậm để chính xác hơn
            objective='regression_l1',# MAE objective - khớp metric WMAE
            num_leaves=31,            # Độ phức tạp của cây
            max_depth=-1,             # Không giới hạn độ sâu (dựa vào num_leaves)
            subsample=0.8,            # Chống overfitting
            colsample_bytree=0.8,     # Chống overfitting
            random_state=42,
            n_jobs=-1,
            verbose=-1                # Tắt log nhiễu
        )

    def train(self, train_data, validation_data=None):
        logger.info("Training LightGBM model...")
        start = time.time()

        features = self._get_features(train_data)
        self.feature_names_ = features

        X_train = train_data[features]
        y_train = train_data['Weekly_Sales']

        # Trọng số ngày lễ x5 - khớp với metric WMAE
        w_train = self._holiday_weights(train_data)

        fit_kwargs = {'sample_weight': w_train}
        if validation_data is not None:
            X_val = validation_data[features]
            y_val = validation_data['Weekly_Sales']
            w_val = self._holiday_weights(validation_data)
            fit_kwargs.update(
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[w_val],
                eval_metric='l1',
                callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
            )

        # Train model
        # Lưu ý: LightGBM có thể tự xử lý NaN, không cần fillna(0)
        self.model.fit(X_train, y_train, **fit_kwargs)

        elapsed = time.time() - start
        best_iter = getattr(self.model, 'best_iteration_', None)
        if best_iter:
            logger.info(f"  ✓ LightGBM training complete in {elapsed:.1f}s (best_iteration={best_iter})")
        else:
            logger.info(f"  ✓ LightGBM training complete in {elapsed:.1f}s")

    def predict(self, test_data):
        features = self._get_features(test_data)
        X_test = test_data[features]
        preds = self.model.predict(X_test)
        return self._clip_negative(preds)
