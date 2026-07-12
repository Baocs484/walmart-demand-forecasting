# src/system.py - FIXED VERSION - DEFAULT XGBOOST

import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
from .utils import logger
from .config import CONFIG

# Import Components
from .data_processor import DataProcessor
from .store_clustering import StoreClustering
from .metrics import MetricsCalculator
from .inventory_report import generate_inventory_report
from .report_builder import build_dashboard, build_excel_report, build_model_report
from .persistence import save_artifacts
from .run_history import log_run
from .powerbi_export import export_powerbi_data

# Import Models
from .models.baseline_model import BaselineModel
from .models.ensemble_model import EnsembleModel
from .models.rf_model import RFModel
from .models.xgb_model import XGBModel
from .models.gb_model import GBModel

# Import Optional Models
try:
    from .models.lightgbm_model import LightGBMModel
except ImportError:
    LightGBMModel = None
try:
    from .models.catboost_model import CatBoostModel
except ImportError:
    CatBoostModel = None

class DemandForecastingSystem:
    def __init__(self, run_mode='lightgbm'):
        """
        Args:
            run_mode (str): Chế độ chạy
                - 'lightgbm': Chỉ chạy LightGBM (MẶC ĐỊNH - tốt nhất và nhanh nhất)
                - 'xgboost': Chỉ chạy XGBoost (chậm hơn chút)
                - 'compare': Chạy tất cả models để so sánh
        """
        logger.info("Initializing Full Forecasting System...")
        
        # 1. Core Components
        self.data_processor = DataProcessor()
        self.store_clustering = StoreClustering(n_clusters=CONFIG['clustering']['n_clusters'])
        
        # 2. CHẾ ĐỘ CHẠY
        self.run_mode = run_mode
        
        # 3. Model Zoo
        self.models = {
            'Baseline': BaselineModel(),
            'RandomForest': RFModel(),
            'XGBoost': XGBModel(),
            'GradientBoost': GBModel(),
            'Ensemble': EnsembleModel()
        }
        
        # Thêm model nâng cao nếu đã cài thư viện
        if LightGBMModel: self.models['LightGBM'] = LightGBMModel()
        if CatBoostModel: self.models['CatBoost'] = CatBoostModel()
            
        # 4. State variables
        self.best_model_name = None
        self.best_model_instance = None
        self.best_metrics = None
        self.comparison_results = []
        self.last_eval_data = None
        self.is_trained = False
        self.processed_data = None
        
        # Tạo thư mục lưu kết quả
        os.makedirs("results", exist_ok=True)

    def run(self, data_path):
        """Quy trình chạy chính: Load -> Train -> Predict -> Visualize -> Inventory"""
        start_global = time.time()
        
        # --- BƯỚC 1: LOAD & PROCESS DATA (leak-free) ---
        df_raw = self.data_processor.load_data(data_path)

        # Xác định cutoff TRƯỚC khi xử lý để mọi thống kê chỉ fit trên train
        unique_dates = sorted(df_raw['Date'].unique())
        n_dates = len(unique_dates)

        train_ratio = CONFIG['split']['train_ratio']
        val_ratio = CONFIG['split']['val_ratio']
        train_cutoff_date = unique_dates[int(train_ratio * n_dates)]
        val_cutoff_date = unique_dates[int((train_ratio + val_ratio) * n_dates)]

        df = self.data_processor.process_full(df_raw, train_cutoff_date=train_cutoff_date)
        self.processed_data = df

        train_df = df[df['Date'] < train_cutoff_date].copy()
        val_df = df[(df['Date'] >= train_cutoff_date) & (df['Date'] < val_cutoff_date)].copy()
        test_df = df[df['Date'] >= val_cutoff_date].copy()
        
        logger.info(f"\nData Split BY DATE:")
        logger.info(f"  Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} records)")
        logger.info(f"  Val:   {val_df['Date'].min()} to {val_df['Date'].max()} ({len(val_df)} records)")
        logger.info(f"  Test:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} records)")
        
        # Verification: Không có overlap
        assert train_df['Date'].max() < val_df['Date'].min(), "Train-Val overlap!"
        assert val_df['Date'].max() < test_df['Date'].min(), "Val-Test overlap!"
        logger.info("  No date overlap - Data split is valid")

        # Lưu thông tin split cho trang chẩn đoán model
        self.split_info = {
            'Train': (train_df['Date'].min(), train_df['Date'].max(), len(train_df)),
            'Validation': (val_df['Date'].min(), val_df['Date'].max(), len(val_df)),
            'Test': (test_df['Date'].min(), test_df['Date'].max(), len(test_df)),
        }
        
        real_train_df = train_df

        # --- BƯỚC 2: PHÂN CỤM CỬA HÀNG ---
        # Fit trên train để tránh leakage, sau đó gán cluster làm FEATURE cho models
        self.store_clustering.fit(real_train_df)
        df['Cluster'] = df['Store'].map(self.store_clustering.get_cluster)
        for split_df in (train_df, val_df, test_df):
            split_df['Cluster'] = split_df['Store'].map(self.store_clustering.get_cluster)

        # --- BƯỚC 3: TRAIN MODELS (Tùy theo mode) ---
        if self.run_mode == 'xgboost':
            # CHẾ ĐỘ TỐI ƯU: CHỈ CHẠY XGBOOST
            self._run_xgboost_only(real_train_df, val_df, test_df)
        elif self.run_mode == 'lightgbm':
            # CHẾ ĐỘ NHANH: CHỈ CHẠY LIGHTGBM
            self._run_lightgbm_only(real_train_df, val_df, test_df)
        else:
            # CHẾ ĐỘ SO SÁNH: CHẠY TẤT CẢ
            self._run_all_models(real_train_df, val_df, test_df)
        
        self.is_trained = True

        # --- BƯỚC 4: FEATURE IMPORTANCE (dùng cho dashboard + Excel) ---
        feature_importance_df = self.compute_feature_importance()
        if feature_importance_df is not None:
            logger.info("\nTop 5 Most Important Features:")
            for idx, row in feature_importance_df.head(5).iterrows():
                logger.info(f"   {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        # --- BƯỚC 5: BÁO CÁO TỒN KHO ---
        logger.info("\nGenerating inventory report...")
        
        class InventoryContext:
            def __init__(self, last_eval_data, store_clusters, processed_data):
                self.last_eval_data = {
                    'test_dates': last_eval_data['Date'],
                    'store': last_eval_data['Store'],
                    'dept': last_eval_data['Dept'],
                    'y_true': last_eval_data['y_true'],
                    'pred_ensemble': last_eval_data['y_pred'],
                    'is_holiday': last_eval_data['is_holiday']
                }
                self.store_clusters = store_clusters
                self.processed_data = processed_data
        
        inventory_context = InventoryContext(
            self.last_eval_data,
            self.store_clustering.store_clusters,
            self.processed_data
        )
        
        inventory_data = generate_inventory_report(inventory_context)

        # --- BƯỚC 6: DASHBOARD + EXCEL REPORT (output chính cho DA/DS) ---
        logger.info("\nBuilding dashboard & Excel report...")
        build_dashboard(
            eval_data=self.last_eval_data,
            metrics=self.best_metrics,
            best_model_name=self.best_model_name,
            feature_importance_df=feature_importance_df,
            inventory=inventory_data,
            comparison_results=self.comparison_results or None,
        )
        build_model_report(
            eval_data=self.last_eval_data,
            metrics=self.best_metrics,
            best_model_name=self.best_model_name,
            model_instance=self.best_model_instance,
            split_info=self.split_info,
            feature_importance_df=feature_importance_df,
            comparison_results=self.comparison_results or None,
        )
        build_excel_report(
            metrics=self.best_metrics,
            best_model_name=self.best_model_name,
            eval_data=self.last_eval_data,
            inventory=inventory_data,
            comparison_results=self.comparison_results or None,
            feature_importance_df=feature_importance_df,
        )

        # --- BƯỚC 6b: XUẤT DATA LAYER CHO POWER BI ---
        export_powerbi_data(
            eval_data=self.last_eval_data,
            inventory=inventory_data,
            processed_data=self.processed_data,
        )

        # --- BƯỚC 7: GHI LỊCH SỬ RUN + LƯU PIPELINE ---
        log_run(self.best_model_name, self.best_metrics, run_mode=self.run_mode)
        logger.info("\nSaving pipeline artifacts...")
        save_artifacts(
            model_instance=self.best_model_instance,
            model_name=self.best_model_name,
            processor=self.data_processor,
            clustering=self.store_clustering,
            metrics=self.best_metrics,
            split_info=self.split_info,
        )

        logger.info(f"\nSystem finished in {time.time() - start_global:.1f}s")

    def _run_xgboost_only(self, train_df, val_df, test_df):
        """CHẾ ĐỘ TỐI ƯU: Chỉ chạy XGBoost"""
        logger.info("\n" + "="*50)
        logger.info("RUNNING XGBOOST MODEL (Best Performance)")
        logger.info("="*50)
        
        if 'XGBoost' not in self.models:
            logger.error("XGBoost not available!")
            logger.info("Falling back to Ensemble model...")
            self._run_all_models(train_df, val_df, test_df)
            return
        
        model = self.models['XGBoost']
        
        try:
            # Train
            model.train(train_df, validation_data=val_df)
            
            # Predict
            preds = model.predict(test_df)
            
            # Evaluate - FIXED: Truyền holiday weights vào WMAE
            weights = np.where(test_df['IsHoliday'].values, CONFIG['metrics']['holiday_weight'], 1)
            metrics = MetricsCalculator.calculate_metrics(test_df['Weekly_Sales'].values, preds, sample_weight=weights)
            
            logger.info(f"\nXGBoost Performance:")
            logger.info(f"   WMAE:  {metrics['WMAE']:.2f}")
            logger.info(f"   MAE:   {metrics['MAE']:.2f}")
            logger.info(f"   RMSE:  {metrics['RMSE']:.2f}")
            logger.info(f"   MAPE:  {metrics['MAPE']:.2f}%")
            
            # Set as best model
            self.best_model_name = 'XGBoost'
            self.best_model_instance = model
            self.best_metrics = metrics
            
            # Lưu kết quả
            self.last_eval_data = {
                'Date': test_df['Date'].values,
                'Store': test_df['Store'].values,
                'Dept': test_df['Dept'].values,
                'Cluster': test_df['Store'].apply(self.store_clustering.get_cluster).values,
                'y_true': test_df['Weekly_Sales'].values,
                'y_pred': preds,
                'is_holiday': test_df['IsHoliday'].values
            }
            
            logger.info(f"\nUsing XGBoost (Best Model)")
            
        except Exception as e:
            logger.error(f"Error with XGBoost: {e}")
            raise

    def _run_lightgbm_only(self, train_df, val_df, test_df):
        """CHẾ ĐỘ NHANH: Chỉ chạy LightGBM"""
        logger.info("\n" + "="*50)
        logger.info("RUNNING LIGHTGBM MODEL (Fast Mode)")
        logger.info("="*50)
        
        if 'LightGBM' not in self.models:
            logger.error("LightGBM not available! Install: pip install lightgbm")
            logger.info("Falling back to Ensemble model...")
            self._run_all_models(train_df, val_df, test_df)
            return
        
        model = self.models['LightGBM']
        
        try:
            # Train
            model.train(train_df, validation_data=val_df)
            
            # Predict
            preds = model.predict(test_df)
            
            # Evaluate - FIXED: Truyền holiday weights vào WMAE
            weights = np.where(test_df['IsHoliday'].values, CONFIG['metrics']['holiday_weight'], 1)
            metrics = MetricsCalculator.calculate_metrics(test_df['Weekly_Sales'].values, preds, sample_weight=weights)
            
            logger.info(f"\nLightGBM Performance:")
            logger.info(f"   WMAE:  {metrics['WMAE']:.2f}")
            logger.info(f"   MAE:   {metrics['MAE']:.2f}")
            logger.info(f"   RMSE:  {metrics['RMSE']:.2f}")
            logger.info(f"   MAPE:  {metrics['MAPE']:.2f}%")
            
            # Set as best model
            self.best_model_name = 'LightGBM'
            self.best_model_instance = model
            self.best_metrics = metrics
            
            # Lưu kết quả
            self.last_eval_data = {
                'Date': test_df['Date'].values,
                'Store': test_df['Store'].values,
                'Dept': test_df['Dept'].values,
                'Cluster': test_df['Store'].apply(self.store_clustering.get_cluster).values,
                'y_true': test_df['Weekly_Sales'].values,
                'y_pred': preds,
                'is_holiday': test_df['IsHoliday'].values
            }
            
            logger.info(f"\nUsing LightGBM (Fast Mode)")
            
        except Exception as e:
            logger.error(f"Error with LightGBM: {e}")
            raise

    def _run_all_models(self, train_df, val_df, test_df):
        """CHẾ ĐỘ SO SÁNH: Chạy tất cả models"""
        logger.info("\n" + "="*50)
        logger.info("STARTING MODEL BATTLE (Compare All)")
        logger.info("="*50)
        
        best_wmae = float('inf')
        
        for name, model in self.models.items():
            logger.info(f"\nProcessing: {name}...")
            try:
                # Train
                model.train(train_df, validation_data=val_df)
                
                # Predict
                preds = model.predict(test_df)
                
                # Evaluate - FIXED: Truyền holiday weights vào WMAE
                weights = np.where(test_df['IsHoliday'].values, CONFIG['metrics']['holiday_weight'], 1)
                metrics = MetricsCalculator.calculate_metrics(test_df['Weekly_Sales'].values, preds, sample_weight=weights)
                metrics['Model'] = name
                self.comparison_results.append(metrics)
                
                logger.info(f"   {name}: WMAE={metrics['WMAE']:.2f} | MAE={metrics['MAE']:.2f}")
                
                # Tìm model tốt nhất
                if metrics['WMAE'] < best_wmae:
                    best_wmae = metrics['WMAE']
                    self.best_model_name = name
                    self.best_model_instance = model
                    self.best_metrics = metrics
                    
                    self.last_eval_data = {
                        'Date': test_df['Date'].values,
                        'Store': test_df['Store'].values,
                        'Dept': test_df['Dept'].values,
                        'Cluster': test_df['Store'].apply(self.store_clustering.get_cluster).values,
                        'y_true': test_df['Weekly_Sales'].values,
                        'y_pred': preds,
                        'is_holiday': test_df['IsHoliday'].values
                    }
            except Exception as e:
                logger.error(f"Error with {name}: {e}")

        self._save_comparison_results()
        logger.info(f"\nWINNER: {self.best_model_name} (WMAE: {best_wmae:.2f})")

    def _save_comparison_results(self):
        """Lưu bảng so sánh ra CSV"""
        if not self.comparison_results: return
        df_res = pd.DataFrame(self.comparison_results)
        cols = ['Model', 'WMAE', 'MAE', 'RMSE', 'MAPE']
        df_res = df_res[cols].sort_values('WMAE')
        
        print("\n" + df_res.to_string(index=False))
        df_res.to_csv("results/model_comparison.csv", index=False)

    def compute_feature_importance(self):
        """Tính Feature Importance của model tốt nhất, trả về DataFrame (hoặc None)."""
        if not self.best_model_instance:
            logger.warning("No trained model found for feature importance.")
            return None

        features = getattr(self.best_model_instance, 'feature_names_', [])
        if not features:
            logger.warning("No features listed in the best model.")
            return None

        # 1. Trích xuất importances
        importances = None
        
        # Trường hợp EnsembleModel
        if self.best_model_name == 'Ensemble':
            # Weighted average của sub-models
            w_gb = self.best_model_instance.global_weights.get('gb', 0.5)
            w_lgbm = self.best_model_instance.global_weights.get('lgbm', 0.5)
            
            gb_imp = None
            if hasattr(self.best_model_instance.gb, 'model') and hasattr(self.best_model_instance.gb.model, 'feature_importances_'):
                gb_imp = self.best_model_instance.gb.model.feature_importances_
            
            lgbm_imp = None
            if self.best_model_instance.lgbm and hasattr(self.best_model_instance.lgbm, 'model') and hasattr(self.best_model_instance.lgbm.model, 'feature_importances_'):
                lgbm_imp = self.best_model_instance.lgbm.model.feature_importances_
                
            if gb_imp is not None and lgbm_imp is not None:
                if len(gb_imp) == len(features) and len(lgbm_imp) == len(features):
                    importances = w_gb * gb_imp + w_lgbm * lgbm_imp
            elif gb_imp is not None:
                if len(gb_imp) == len(features):
                    importances = gb_imp
            elif lgbm_imp is not None:
                if len(lgbm_imp) == len(features):
                    importances = lgbm_imp
        else:
            # Các model đơn lẻ
            underlying_model = getattr(self.best_model_instance, 'model', None)
            if underlying_model is not None:
                if hasattr(underlying_model, 'feature_importances_'):
                    importances = underlying_model.feature_importances_
                elif hasattr(underlying_model, 'get_feature_importance'):
                    importances = underlying_model.get_feature_importance()

        if importances is None:
            logger.warning(f"Model {self.best_model_name} does not support feature importance.")
            return None

        if len(importances) != len(features):
            logger.warning(f"Length mismatch: {len(importances)} importances vs {len(features)} features.")
            return None

        # 2. Tạo DataFrame và sắp xếp
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)

        return df_imp