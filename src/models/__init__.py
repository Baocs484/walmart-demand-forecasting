# src/models/__init__.py - UPDATED
"""
Models package - All ML models
"""

from .base_model import BaseModel
from .baseline_model import BaselineModel
from .ensemble_model import EnsembleModel

# Sub-models
from .rf_model import RFModel
from .xgb_model import XGBModel
from .gb_model import GBModel

# Advanced models (optional)
try:
    from .lightgbm_model import LightGBMModel
except ImportError:
    LightGBMModel = None

try:
    from .catboost_model import CatBoostModel
except ImportError:
    CatBoostModel = None

__all__ = [
    'BaseModel',
    'BaselineModel',
    'EnsembleModel',
    'RFModel',
    'XGBModel', 
    'GBModel',
    'LightGBMModel',
    'CatBoostModel'
]