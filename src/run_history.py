# src/run_history.py
"""Append-only run log (results/runs.jsonl) - a lightweight experiment tracker."""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from .utils import logger

RUNS_FILE = Path('results/runs.jsonl')


def log_run(model_name, metrics, run_mode='train', notes=''):
    """Append one line per pipeline run. Never raises - tracking must not break runs."""
    try:
        RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'model': model_name,
            'run_mode': run_mode,
            'WMAE': round(float(metrics['WMAE']), 2),
            'MAE': round(float(metrics['MAE']), 2),
            'RMSE': round(float(metrics['RMSE']), 2),
            'MAPE': round(float(metrics.get('MAPE', float('nan'))), 2),
            'notes': notes,
        }
        with open(RUNS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        logger.info(f'  ✓ Run logged to {RUNS_FILE}')
    except Exception as e:
        logger.warning(f'Could not log run: {e}')


def load_runs():
    """Return run history as a DataFrame (empty if no history)."""
    if not RUNS_FILE.exists():
        return pd.DataFrame()
    try:
        rows = [json.loads(line) for line in
                RUNS_FILE.read_text(encoding='utf-8').splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
