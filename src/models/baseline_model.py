# src/models/baseline_model.py - REFACTORED: Kế thừa BaseModel + Vectorized predict
import numpy as np
import pandas as pd
import time
from ..utils import logger
from .base_model import BaseModel


class BaselineModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.store_baselines = {}
        self.dept_baselines = {}
        self.monthly_patterns = {}
        self.global_mean = 0
    
    def train(self, df_train, validation_data=None):
        """Train baseline model (Tính trung bình lịch sử)"""
        logger.info("Training Baseline model...")
        
        start_total = time.time()
        
        # 1. Global mean
        self.global_mean = df_train['Weekly_Sales'].mean()
        
        # 2. Store baselines
        self.store_baselines = df_train.groupby('Store')['Weekly_Sales'].mean().to_dict()
        
        # 3. Dept baselines (Store + Dept)
        self.dept_baselines = df_train.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().to_dict()
        
        # 4. Seasonal patterns (Mùa vụ)
        # Tính tỷ lệ so với trung bình theo Store + Month
        df_train = df_train.copy()
        df_train['Month'] = df_train['Date'].dt.month
        monthly_avg = df_train.groupby(['Store', 'Month'])['Weekly_Sales'].mean()
        
        for store in df_train['Store'].unique():
            store_mean = self.store_baselines.get(store, self.global_mean)
            if store_mean > 0:
                try:
                    store_months = monthly_avg.loc[store]
                    # Pattern = Doanh số tháng / Doanh số trung bình store
                    self.monthly_patterns[store] = (store_months / store_mean).to_dict()
                except KeyError:
                    self.monthly_patterns[store] = {}
        
        logger.info(f"  ✓ Baseline calculated in {time.time() - start_total:.1f}s")
    
    def predict(self, df_test):
        """FIXED: Dự báo vectorized (không dùng iterrows - nhanh hơn 100x)"""
        df_test = df_test.copy()
        
        if 'Month' not in df_test.columns:
            df_test['Month'] = df_test['Date'].dt.month
        
        # 1. Lấy base prediction bằng vectorized map
        # Ưu tiên Dept > Store > Global
        dept_keys = list(zip(df_test['Store'], df_test['Dept']))
        base_preds = np.array([
            self.dept_baselines.get(key, 
                self.store_baselines.get(key[0], self.global_mean))
            for key in dept_keys
        ])
        
        # 2. Nhân hệ số mùa vụ (Seasonal Adjustment)
        season_factors = np.array([
            self.monthly_patterns.get(store, {}).get(month, 1.0)
            for store, month in zip(df_test['Store'], df_test['Month'])
        ])
        
        preds = base_preds * season_factors
        return self._clip_negative(preds)