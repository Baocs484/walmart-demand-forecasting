# main.py
"""
Walmart Demand Forecasting System - CLI entry point.

Commands:
    python main.py                       # = train (LightGBM, fast mode)
    python main.py train                 # train + evaluate + reports + save artifacts
    python main.py train --model xgboost
    python main.py compare               # train & compare all 7 models
    python main.py forecast --weeks 4    # recursive future forecast (needs a trained model)

Legacy aliases still work: `python main.py xgboost`, `python main.py compare`.
"""

import argparse
import sys
from pathlib import Path

from src import DemandForecastingSystem, set_seed, logger
from src.config import CONFIG

DATA_PATH = Path(CONFIG['data']['processed_path'])


def _check_data():
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.info("Run data preparation first: python scripts/merge_walmart.py")
        sys.exit(1)


def cmd_train(model='lightgbm'):
    _check_data()
    run_mode = 'lightgbm' if model == 'lightgbm' else 'xgboost'
    logger.info(f"Running TRAIN mode ({run_mode})")
    system = DemandForecastingSystem(run_mode=run_mode)
    system.run(str(DATA_PATH))
    _print_outputs(compare=False)


def cmd_compare():
    _check_data()
    logger.info("Running COMPARE mode (all models)")
    system = DemandForecastingSystem(run_mode='compare')
    system.run(str(DATA_PATH))
    _print_outputs(compare=True)


def cmd_forecast(weeks):
    import pandas as pd
    from src.persistence import load_artifacts
    from src.forecaster import RecursiveForecaster

    _check_data()
    bundle, meta = load_artifacts()

    raw = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    forecaster = RecursiveForecaster(
        model=bundle['model'],
        processor=bundle['processor'],
        clustering=bundle['clustering'],
    )
    result = forecaster.forecast(raw, horizon_weeks=weeks)

    out = Path('results/predictions')
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f'forecast_{weeks}w.csv'
    result.to_csv(out_file, index=False)

    weekly = result.groupby(['Horizon', 'Date', 'IsHoliday'])['Forecast_Weekly_Sales'] \
                   .sum().reset_index()
    logger.info("\nForecast summary (network total per week):")
    for _, r in weekly.iterrows():
        flag = ' [HOLIDAY]' if r['IsHoliday'] else ''
        logger.info(f"  +{int(r['Horizon'])}w  {pd.Timestamp(r['Date']):%Y-%m-%d}{flag}: "
                    f"${r['Forecast_Weekly_Sales']:,.0f}")
    logger.info(f"\n  ✓ Saved: {out_file} ({len(result):,} rows)")


def _print_outputs(compare):
    logger.info("\n" + "=" * 60)
    logger.info("✓ SYSTEM COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("\n📁 Main outputs:")
    logger.info("  ⭐ results/dashboard.html                  — Business analytics report")
    logger.info("  ⭐ results/model_report.html               — Model diagnostics (data science)")
    logger.info("  ⭐ results/reports/forecast_report.xlsx    — Formatted Excel report")
    logger.info("  💾 models/pipeline.joblib                  — Saved pipeline (for `forecast`)")
    if compare:
        logger.info("  📊 results/model_comparison.csv           — All-model comparison")
    logger.info("\n📁 Detail outputs:")
    logger.info("  - results/inventory/*.csv                 — Raw inventory data (Power BI/pandas)")


def main():
    print("\n" + "=" * 60)
    print("   WALMART DEMAND FORECASTING SYSTEM")
    print("=" * 60 + "\n")
    set_seed(42)

    # Legacy positional aliases -> subcommands
    argv = sys.argv[1:]
    if argv and argv[0].lower() in ('xgboost', 'xgb'):
        argv = ['train', '--model', 'xgboost']
    elif argv and argv[0].lower() in ('all', 'benchmark'):
        argv = ['compare']
    elif not argv:
        argv = ['train']

    parser = argparse.ArgumentParser(description='Walmart Demand Forecasting System')
    sub = parser.add_subparsers(dest='command')

    p_train = sub.add_parser('train', help='train, evaluate and build reports')
    p_train.add_argument('--model', default='lightgbm', choices=['lightgbm', 'xgboost'])

    sub.add_parser('compare', help='train & compare all models')

    p_fc = sub.add_parser('forecast', help='recursive future forecast from saved model')
    p_fc.add_argument('--weeks', type=int, default=CONFIG['forecast']['horizon_weeks'])

    args = parser.parse_args(argv)

    try:
        if args.command == 'train':
            cmd_train(args.model)
        elif args.command == 'compare':
            cmd_compare()
        elif args.command == 'forecast':
            cmd_forecast(args.weeks)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
