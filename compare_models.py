# compare_models.py
"""
So sánh chi tiết tất cả models trên dữ liệu Walmart.

Usage:
    python compare_models.py

Output:
    - results/model_comparison.csv: Bảng so sánh metrics
    - results/plots/model_comparison.html: Biểu đồ so sánh (Plotly)
    - Console: Bảng xếp hạng models
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src import DataProcessor, StoreClustering, MetricsCalculator, set_seed, logger
from src.config import CONFIG
from src.models import (
    BaselineModel, RFModel, XGBModel, GBModel, EnsembleModel,
    LightGBMModel, CatBoostModel
)

# Plotly (optional)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

DATA_PATH = Path("data/processed/walmart_clean.csv")


def compare_all_models() -> pd.DataFrame:
    """
    Chạy tất cả models, đánh giá, và xuất kết quả so sánh.

    Returns:
        pd.DataFrame: Bảng so sánh metrics của tất cả models.
    """
    set_seed(42)

    print("\n" + "=" * 60)
    print("   MODEL COMPARISON - WALMART DEMAND FORECASTING")
    print("=" * 60 + "\n")

    # --- 1. Load & Process Data ---
    if not DATA_PATH.exists():
        logger.error(f"❌ Data file not found: {DATA_PATH}")
        logger.info("Please run: python scripts/merge_walmart.py")
        return pd.DataFrame()

    processor = DataProcessor()
    df_raw = processor.load_data(str(DATA_PATH))

    # --- 2. Time-based split (cutoffs chosen BEFORE fitting statistics) ---
    unique_dates = sorted(df_raw['Date'].unique())
    n_dates = len(unique_dates)

    train_ratio = CONFIG['split']['train_ratio']
    val_ratio = CONFIG['split']['val_ratio']
    train_cutoff = unique_dates[int(train_ratio * n_dates)]
    val_cutoff = unique_dates[int((train_ratio + val_ratio) * n_dates)]

    df = processor.process_full(df_raw, train_cutoff_date=train_cutoff)

    train_df = df[df['Date'] < train_cutoff].copy()
    val_df = df[(df['Date'] >= train_cutoff) & (df['Date'] < val_cutoff)].copy()
    test_df = df[df['Date'] >= val_cutoff].copy()

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # --- 2b. Store Clustering (fit trên train, dùng làm feature) ---
    clustering = StoreClustering(n_clusters=CONFIG['clustering']['n_clusters'])
    clustering.fit(train_df)
    for split_df in (train_df, val_df, test_df):
        split_df['Cluster'] = split_df['Store'].map(clustering.get_cluster)

    # --- 3. Prepare Models ---
    models: dict = {
        'Baseline': BaselineModel(),
        'RandomForest': RFModel(),
        'XGBoost': XGBModel(),
        'GradientBoost': GBModel(),
        'Ensemble': EnsembleModel(),
    }

    if LightGBMModel is not None:
        models['LightGBM'] = LightGBMModel()
    else:
        logger.warning("⚠ LightGBM not installed, skipping.")

    if CatBoostModel is not None:
        models['CatBoost'] = CatBoostModel()
    else:
        logger.warning("⚠ CatBoost not installed, skipping.")

    # --- 4. Train & Evaluate ---
    results = []
    predictions = {}

    weights = np.where(test_df['IsHoliday'].values, CONFIG['metrics']['holiday_weight'], 1)
    y_true = test_df['Weekly_Sales'].values

    for name, model in models.items():
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Training: {name}")
        logger.info(f"{'─' * 40}")

        try:
            start = time.time()
            model.train(train_df, validation_data=val_df)
            train_time = time.time() - start

            start = time.time()
            preds = model.predict(test_df)
            pred_time = time.time() - start

            # Accuracy metrics
            acc_metrics = MetricsCalculator.calculate_metrics(
                y_true, preds, sample_weight=weights
            )

            # Business metrics
            biz_metrics = MetricsCalculator.calculate_business_metrics(
                y_true, preds
            )

            result = {
                'Model': name,
                'WMAE': acc_metrics['WMAE'],
                'MAE': acc_metrics['MAE'],
                'RMSE': acc_metrics['RMSE'],
                'MAPE': acc_metrics['MAPE'],
                'sMAPE': acc_metrics['sMAPE'],
                'Service_Level': biz_metrics['Service_Level'],
                'Stockout_Rate': biz_metrics['Stockout_Rate'],
                'Total_Cost': biz_metrics['Total_Cost'],
                'Train_Time_s': round(train_time, 1),
                'Predict_Time_s': round(pred_time, 3),
            }

            results.append(result)
            predictions[name] = preds

            logger.info(
                f"  ✓ {name}: WMAE={acc_metrics['WMAE']:.2f} | "
                f"MAE={acc_metrics['MAE']:.2f} | "
                f"Time={train_time:.1f}s"
            )

        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")

    # --- 5. Results Table ---
    if not results:
        logger.error("No models completed successfully!")
        return pd.DataFrame()

    df_results = pd.DataFrame(results).sort_values('WMAE')

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS (Sorted by WMAE)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # --- 6. Save Results ---
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(output_dir / "model_comparison.csv", index=False)
    logger.info(f"\n✓ Results saved to {output_dir / 'model_comparison.csv'}")

    # --- 7. Visualization ---
    if go is not None:
        _create_comparison_chart(df_results)

    # --- 8. Winner ---
    winner = df_results.iloc[0]
    print(f"\n🏆 WINNER: {winner['Model']} (WMAE: {winner['WMAE']:.2f})")

    return df_results


def _create_comparison_chart(df_results: pd.DataFrame) -> None:
    """Tạo biểu đồ so sánh models bằng Plotly."""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'WMAE (Lower is Better)',
            'RMSE (Lower is Better)',
            'Service Level % (Higher is Better)',
            'Training Time (seconds)'
        )
    )

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
              '#FFA15A', '#19D3F3', '#FF6692']

    models = df_results['Model'].tolist()
    bar_colors = colors[:len(models)]

    # WMAE
    fig.add_trace(go.Bar(
        x=models, y=df_results['WMAE'],
        marker_color=bar_colors, name='WMAE',
        showlegend=False
    ), row=1, col=1)

    # RMSE
    fig.add_trace(go.Bar(
        x=models, y=df_results['RMSE'],
        marker_color=bar_colors, name='RMSE',
        showlegend=False
    ), row=1, col=2)

    # Service Level
    fig.add_trace(go.Bar(
        x=models, y=df_results['Service_Level'],
        marker_color=bar_colors, name='Service Level',
        showlegend=False
    ), row=2, col=1)

    # Training Time
    fig.add_trace(go.Bar(
        x=models, y=df_results['Train_Time_s'],
        marker_color=bar_colors, name='Train Time',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        title_text='Model Comparison Dashboard',
        template='plotly_white',
        height=700
    )

    output_path = "results/plots/model_comparison.html"
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"✓ Comparison chart saved to {output_path}")


if __name__ == "__main__":
    compare_all_models()
