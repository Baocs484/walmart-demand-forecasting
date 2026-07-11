# src/models/base_model.py - IMPROVED: Type Hints + Docstrings
"""
Base class cho tất cả models - Loại bỏ code trùng lặp
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional
from ..utils import logger
from ..config import CONFIG


class BaseModel(ABC):
    """
    Abstract base class cho tất cả forecasting models.
    Chứa feature list chung và interface thống nhất.
    
    Mọi model mới phải kế thừa class này và implement:
    - train(): Huấn luyện model
    - predict(): Dự báo
    """

    # Feature list CHUNG cho tất cả models (bao gồm dept-specific features)
    FEATURE_LIST: List[str] = [
        # Identifiers
        'Store', 'Dept',
        # Time features
        'Month', 'Quarter', 'DayOfWeek', 'IsWeekend', 'IsHoliday',
        # Cyclical encoding
        'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
        # Lag features
        'Sales_lag_1', 'Sales_lag_4', 'Sales_lag_52',
        # Rolling features
        'Sales_roll_mean_4', 'Sales_roll_mean_13', 'Sales_roll_std_4',
        # Multi-store features
        'Regional_Sales_Lag', 'Regional_Market_Share_Lag',
        'National_Market_Share_Lag', 'Store_Rank_Lag',
        # Growth features
        'Sales_Growth_4w', 'Sales_Growth_YoY',
        # Dept features
        'Dept_Share_Lag', 'Dept_Rank_Lag',
        # Dept-specific features (FIXED: now leak-free)
        'Dept_Avg_Sales', 'Sales_vs_Dept_Avg', 'Dept_CV',
        'Dept_Zero_Freq', 'Store_Rank_in_Dept', 'Dept_Size_Indicator',
        # Markdown features
        'Markdown_Total', 'Has_Markdown',
        # Store info
        'Size', 'Type_encoded', 'Cluster'
    ]

    def __init__(self) -> None:
        self.feature_names_: List[str] = []

    def _get_features(self, df: pd.DataFrame) -> List[str]:
        """
        Lấy danh sách features có sẵn trong dataframe.
        
        Args:
            df: DataFrame chứa data
            
        Returns:
            Danh sách tên features có trong cả FEATURE_LIST và df.columns
        """
        return [f for f in self.FEATURE_LIST if f in df.columns]

    @abstractmethod
    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> None:
        """
        Train model.
        
        Args:
            train_data: DataFrame chứa data train
            validation_data: DataFrame chứa data validation (optional)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Predict.
        
        Args:
            test_data: DataFrame chứa data cần dự báo
            
        Returns:
            Array chứa kết quả dự báo
        """
        raise NotImplementedError

    @staticmethod
    def _holiday_weights(df: pd.DataFrame) -> np.ndarray:
        """
        Trọng số mẫu khớp với metric WMAE: tuần lễ = 5, tuần thường = 1.

        Args:
            df: DataFrame có cột IsHoliday

        Returns:
            Array trọng số cùng độ dài với df
        """
        if 'IsHoliday' in df.columns:
            w = float(CONFIG['metrics']['holiday_weight'])
            return np.where(df['IsHoliday'].values.astype(bool), w, 1.0)
        return np.ones(len(df))

    def _clip_negative(self, preds: np.ndarray) -> np.ndarray:
        """
        Đảm bảo predictions không âm.
        
        Args:
            preds: Array chứa predictions
            
        Returns:
            Array đã clip giá trị âm thành 0
        """
        preds = np.array(preds)
        preds[preds < 0] = 0
        return preds
