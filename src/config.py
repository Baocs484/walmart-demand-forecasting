# src/config.py
"""Load config.yaml with safe defaults - the file is optional."""

import copy
from pathlib import Path
from .utils import logger

try:
    import yaml
except ImportError:
    yaml = None

DEFAULTS = {
    'data': {'processed_path': 'data/processed/walmart_clean.csv'},
    'split': {'train_ratio': 0.70, 'val_ratio': 0.15},
    'clustering': {'n_clusters': 5},
    'metrics': {'holiday_weight': 5},
    'inventory': {
        'min_weeks': 4,
        'lead_time_weeks': 1,
        'z_scores': {'AX': 2.576, 'AY': 2.33, 'AZ': 2.05,
                     'BX': 2.17, 'BY': 2.05, 'BZ': 1.88,
                     'CX': 1.88, 'CY': 1.645, 'CZ': 1.28},
        'default_z': 1.645,
        'costs': {'stockout_penalty': 50, 'unit_cost': 10, 'holding_rate': 0.1},
    },
    'forecast': {'horizon_weeks': 4},
}


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path='config.yaml'):
    """Return DEFAULTS merged with config.yaml (if present and parseable)."""
    p = Path(path)
    if yaml is None or not p.exists():
        return copy.deepcopy(DEFAULTS)
    try:
        with open(p, encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULTS, user_cfg)
    except Exception as e:
        logger.warning(f'Could not parse {path} ({e}) - using defaults.')
        return copy.deepcopy(DEFAULTS)


# Singleton, loaded once at import time
CONFIG = load_config()
