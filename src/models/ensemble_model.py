# src/models/ensemble_model.py - REFACTORED: Kế thừa BaseModel + Warning rõ ràng
import numpy as np
from scipy.optimize import minimize
from ..utils import logger
from .base_model import BaseModel

# Import sub-models
from .gb_model import GBModel
try:
    from .lightgbm_model import LightGBMModel
except ImportError:
    LightGBMModel = None


class EnsembleModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Khởi tạo sub-models
        self.gb = GBModel()
        self.lgbm = LightGBMModel() if LightGBMModel else None
        
        # FIXED: Warning rõ ràng khi thiếu LightGBM
        if self.lgbm is None:
            logger.warning("⚠ Ensemble: LightGBM not available! "
                          "Falling back to GradientBoosting only. "
                          "Install: pip install lightgbm")
            self.global_weights = {'lgbm': 0.0, 'gb': 1.0}
        else:
            # Chiến thuật mặc định: 60% LightGBM + 40% GB
            self.global_weights = {'lgbm': 0.6, 'gb': 0.4}

    def train(self, train_data, validation_data=None):
        if self.lgbm:
            logger.info("Training Optimized Ensemble (LightGBM + GradientBoosting)...")
        else:
            logger.info("Training Ensemble (GradientBoosting only - LightGBM unavailable)...")
        
        # 1. Train GB
        self.gb.train(train_data)
        
        # 2. Train LightGBM
        if self.lgbm: 
            self.lgbm.train(train_data, validation_data)

        # 3. Tự động tìm tỷ lệ vàng (nếu có validation data)
        if validation_data is not None and self.lgbm:
            self._optimize_weights(validation_data)
            
    def _optimize_weights(self, val_data):
        logger.info("Optimizing weights between LightGBM and GB...")

        # Lấy dự đoán
        pred_gb = self.gb.predict(val_data)
        pred_lgbm = self.lgbm.predict(val_data)

        y_true = val_data['Weekly_Sales'].values
        sample_w = self._holiday_weights(val_data)

        # Hàm mục tiêu: minimize WMAE (tuần lễ x5) - khớp với metric đánh giá
        def objective(w):
            w_lgbm = w[0]
            w_gb = 1 - w_lgbm
            final_pred = w_lgbm * pred_lgbm + w_gb * pred_gb
            return np.sum(sample_w * np.abs(y_true - final_pred)) / np.sum(sample_w)  # WMAE
            
        # Bắt đầu thử từ tỷ lệ 50-50
        res = minimize(objective, [0.5], bounds=[(0, 1)], method='L-BFGS-B')
        
        if res.success:
            w_lgbm = res.x[0]
            self.global_weights = {'lgbm': w_lgbm, 'gb': 1 - w_lgbm}
            logger.info(f"  ✓ OPTIMIZED WEIGHTS: LightGBM: {w_lgbm:.2f}, GradientBoosting: {1-w_lgbm:.2f}")

    def predict(self, test_data):
        pred_gb = self.gb.predict(test_data)
        
        if self.lgbm:
            pred_lgbm = self.lgbm.predict(test_data)
            # Tổng hợp kết quả
            preds = (pred_lgbm * self.global_weights['lgbm'] + 
                     pred_gb * self.global_weights['gb'])
        else:
            preds = pred_gb
        
        return self._clip_negative(preds)