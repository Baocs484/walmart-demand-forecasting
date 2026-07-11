# tests/test_system.py
import pytest
import os
import pandas as pd
from src.system import DemandForecastingSystem

def test_system_run_and_visualizations(tmp_path, sample_df, monkeypatch):
    """Test full DemandForecastingSystem run and verify all visualizations/reports are generated."""
    # Save synthetic sample data to temp path
    data_file = tmp_path / "synthetic_walmart.csv"

    df_temp = sample_df.copy()
    df_temp.to_csv(data_file, index=False)

    # Chạy trong thư mục tạm để KHÔNG ghi đè results/ thật của dự án
    # (system ghi output bằng đường dẫn tương đối "results/...")
    monkeypatch.chdir(tmp_path)
    
    # Run with xgboost mode
    system = DemandForecastingSystem(run_mode='xgboost')
    
    # Run the system
    system.run(str(data_file))
    
    # Verify training completed
    assert system.is_trained
    assert system.best_model_name == 'XGBoost'
    
    # Verify outputs are generated
    assert os.path.exists("results/dashboard.html")
    assert os.path.exists("results/model_report.html")
    assert os.path.exists("results/reports/forecast_report.xlsx")
    assert os.path.exists("results/inventory/store_inventory_summary.csv")
