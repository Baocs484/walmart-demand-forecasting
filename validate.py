# validate.py
"""
Rolling-origin cross-validation (time-series CV).

Why: a single fixed test window (Jun-Oct 2012) contains only Labor Day, while
WMAE weighs holiday weeks 5x. Evaluating on several rolling windows - one of
which covers Thanksgiving + Christmas 2011 - gives a far more trustworthy
picture of model quality.

Scheme (expanding window over the last `folds * test_weeks` weeks):

    |--------- train ---------|-- val --|-- test 1 --|
    |------------- train -------------|-- val --|-- test 2 --|
    ...

For each fold, feature statistics are re-fitted on that fold's train slice
only (leak-free), and the model early-stops on the validation slice.

Usage:
    python validate.py                     # LightGBM, 4 folds x 13 weeks
    python validate.py --model xgboost
    python validate.py --folds 3 --test-weeks 8

Output:
    - results/cv_results.csv   (per-fold metrics, used by the model report)
    - console summary: mean +/- std per metric
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src import DataProcessor, StoreClustering, MetricsCalculator, set_seed, logger
from src.config import CONFIG
from src.models import XGBModel, LightGBMModel

DATA_PATH = Path(CONFIG['data']['processed_path'])
VAL_WEEKS = 13  # weeks reserved before each test window for early stopping


def _build_model(name):
    if name == 'lightgbm':
        if LightGBMModel is None:
            raise SystemExit('LightGBM not installed: pip install lightgbm')
        return LightGBMModel()
    if name == 'xgboost':
        return XGBModel()
    raise SystemExit(f'Unknown model: {name}')


def run_cv(model_name='lightgbm', n_folds=4, test_weeks=13):
    set_seed(42)

    if not DATA_PATH.exists():
        raise SystemExit(f'Data file not found: {DATA_PATH} - run scripts/merge_walmart.py first')

    print('\n' + '=' * 64)
    print(f'   ROLLING-ORIGIN CROSS-VALIDATION - {model_name} x {n_folds} folds')
    print('=' * 64)

    raw = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    dates = np.array(sorted(raw['Date'].unique()))
    n = len(dates)

    needed = n_folds * test_weeks + VAL_WEEKS + 52  # 52 = minimum training year
    if n < needed:
        raise SystemExit(f'Not enough history: {n} weeks < {needed} required')

    holiday_w = CONFIG['metrics']['holiday_weight']
    results = []

    for fold in range(n_folds):
        # Test window: the k-th block counting back from the end
        test_end_i = n - fold * test_weeks            # exclusive
        test_start_i = test_end_i - test_weeks
        val_start_i = test_start_i - VAL_WEEKS

        test_start, test_end = dates[test_start_i], dates[test_end_i - 1]
        val_start = dates[val_start_i]

        fold_id = n_folds - fold  # chronological numbering (1 = oldest window)
        logger.info(f'\n───── Fold {fold_id}: test {pd.Timestamp(test_start):%Y-%m-%d} '
                    f'→ {pd.Timestamp(test_end):%Y-%m-%d} ─────')

        # Leak-free processing: statistics fitted strictly before validation
        fold_raw = raw[raw['Date'] <= test_end].copy()
        processor = DataProcessor()
        df = processor.process_full(fold_raw, train_cutoff_date=val_start)

        train_df = df[df['Date'] < val_start].copy()
        val_df = df[(df['Date'] >= val_start) & (df['Date'] < test_start)].copy()
        test_df = df[df['Date'] >= test_start].copy()

        # Cluster feature (fitted on train)
        clustering = StoreClustering(n_clusters=CONFIG['clustering']['n_clusters'])
        clustering.fit(train_df)
        for part in (train_df, val_df, test_df):
            part['Cluster'] = part['Store'].map(clustering.get_cluster)

        model = _build_model(model_name)
        t0 = time.time()
        model.train(train_df, validation_data=val_df)
        train_time = time.time() - t0

        preds = model.predict(test_df)
        weights = np.where(test_df['IsHoliday'].values, holiday_w, 1)
        m = MetricsCalculator.calculate_metrics(
            test_df['Weekly_Sales'].values, preds, sample_weight=weights)

        n_holiday_weeks = int(test_df.loc[test_df['IsHoliday'] == 1, 'Date'].nunique())
        results.append({
            'Fold': fold_id,
            'Test_Start': pd.Timestamp(test_start).date(),
            'Test_End': pd.Timestamp(test_end).date(),
            'Holiday_Weeks': n_holiday_weeks,
            'WMAE': m['WMAE'], 'MAE': m['MAE'],
            'RMSE': m['RMSE'], 'MAPE': m['MAPE'],
            'Train_Time_s': round(train_time, 1),
            'Model': model_name,
        })
        logger.info(f'  Fold {fold_id}: WMAE={m["WMAE"]:.0f} | MAE={m["MAE"]:.0f} | '
                    f'MAPE={m["MAPE"]:.1f}% | holiday weeks in test: {n_holiday_weeks}')

    df_res = pd.DataFrame(results).sort_values('Fold').reset_index(drop=True)

    out = Path('results/cv_results.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out, index=False)

    print('\n' + '=' * 64)
    print('CV RESULTS PER FOLD')
    print('=' * 64)
    print(df_res.to_string(index=False))
    print('-' * 64)
    for metric in ['WMAE', 'MAE', 'MAPE']:
        print(f'  {metric:>5}: {df_res[metric].mean():,.1f} ± {df_res[metric].std():,.1f}')
    print(f'\nSaved to {out}')
    return df_res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Rolling-origin cross-validation')
    ap.add_argument('--model', default='lightgbm', choices=['lightgbm', 'xgboost'])
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--test-weeks', type=int, default=13)
    args = ap.parse_args()
    run_cv(args.model, args.folds, args.test_weeks)
