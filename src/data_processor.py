# src/data_processor.py
"""
Feature engineering with a leak-free fit/transform design.

Two usage modes:

1. LEAK-FREE (recommended - used by the training pipeline):
       df_raw = processor.load_data(path)
       df     = processor.process_full(df_raw, train_cutoff_date)
   Base features (lags, rolling, calendar, market shares) are computed on
   the full timeline - they only look BACKWARD, so this is safe. All fitted
   statistics (department aggregates, NaN-fill values, clip bounds) are
   computed ONLY on rows before `train_cutoff_date` and then applied to the
   whole frame.

2. LEGACY (kept for backward compatibility - fits statistics on the whole
   frame, mild leakage):
       df = processor.load_and_process_data(path)

After fitting, `transform(df_raw)` applies the stored statistics to new
data - this is what recursive forecasting and the predict CLI use.
"""

import numpy as np
import pandas as pd
from .utils import logger

EPSILON = 1e-6


class DataProcessor:
    def __init__(self):
        self.dept_stats_ = None    # department-level aggregates (from train)
        self.fill_stats_ = None    # NaN-fill values for lag/roll columns (from train)
        self.clip_bounds_ = None   # per-column q01/q99 (from train)

    @property
    def is_fitted(self):
        return self.dept_stats_ is not None

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def load_data(self, file_path):
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, parse_dates=['Date'])
        logger.info(f"Loaded: {df.shape}")
        return df

    def process_full(self, df_raw, train_cutoff_date=None):
        """
        Leak-free pipeline. Statistics are fitted on rows with
        Date < train_cutoff_date (or on everything if None - legacy mode).
        """
        df = self._create_base_features(df_raw)

        fit_slice = df if train_cutoff_date is None else df[df['Date'] < train_cutoff_date]
        if len(fit_slice) == 0:
            raise ValueError("train_cutoff_date leaves no rows to fit statistics on")

        self.dept_stats_ = self._compute_dept_stats(fit_slice)
        df = self._apply_dept_stats(df, self.dept_stats_)

        self.fill_stats_ = self._compute_fill_stats(fit_slice)
        df = self._handle_missing_values(df, self.fill_stats_)

        clip_slice = df if train_cutoff_date is None else df[df['Date'] < train_cutoff_date]
        self.clip_bounds_ = self._compute_clip_bounds(clip_slice)
        df = self._validate_and_clean(df, self.clip_bounds_)

        logger.info(f"Processed: {df.shape[0]} records, {df.shape[1]} features"
                    + ("" if train_cutoff_date is None else " (statistics fitted on train only)"))
        return df

    def transform(self, df_raw):
        """Apply the FITTED statistics to new raw data (inference path)."""
        if not self.is_fitted:
            raise RuntimeError("DataProcessor is not fitted - run process_full() first")
        df = self._create_base_features(df_raw)
        df = self._apply_dept_stats(df, self.dept_stats_)
        df = self._handle_missing_values(df, self.fill_stats_)
        df = self._validate_and_clean(df, self.clip_bounds_)
        return df

    def load_and_process_data(self, file_path):
        """LEGACY: load + process with statistics fitted on the whole frame."""
        df = self.load_data(file_path)
        return self.process_full(df, train_cutoff_date=None)

    def create_features(self, df):
        """LEGACY: base features + dept features with stats from this frame."""
        df = self._create_base_features(df)
        stats = self._compute_dept_stats(df)
        df = self._apply_dept_stats(df, stats)
        return df

    # ==================================================================
    # BASE FEATURES (backward-looking only - safe on the full timeline)
    # ==================================================================

    def _create_base_features(self, df):
        df = df.reset_index(drop=True).sort_values(['Store', 'Dept', 'Date'])

        logger.info("Creating features...")

        # 1. Time features
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        if 'IsHoliday' in df.columns:
            df['IsHoliday'] = df['IsHoliday'].astype(int)

        # 2. Cyclical encoding
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # 3. Lag features (backward-looking)
        gb = df.groupby(['Store', 'Dept'])['Weekly_Sales']
        df['Sales_lag_1'] = gb.shift(1)
        df['Sales_lag_4'] = gb.shift(4)
        df['Sales_lag_52'] = gb.shift(52)

        # 4. Rolling features (over lagged values - backward-looking)
        gb_lag = df.groupby(['Store', 'Dept'])['Sales_lag_1']
        df['Sales_roll_mean_4'] = gb_lag.rolling(4, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        df['Sales_roll_mean_13'] = gb_lag.rolling(13, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        df['Sales_roll_std_4'] = gb_lag.rolling(4, min_periods=1).std().reset_index(level=[0, 1], drop=True)

        # 5. Cross-sectional shares & ranks (aggregate LAGGED values within
        #    the same date - known at prediction time)
        if 'Region' not in df.columns:
            df['Region'] = 'R' + ((df['Store'] - 1) // 5 + 1).astype(str)

        regional_sales_lag = df.groupby(['Date', 'Region'])['Sales_lag_1'].transform('sum')
        df['Regional_Sales_Lag'] = regional_sales_lag
        df['Regional_Market_Share_Lag'] = df['Sales_lag_1'] / (regional_sales_lag + EPSILON)

        national_total_lag = df.groupby('Date')['Sales_lag_1'].transform('sum')
        df['National_Market_Share_Lag'] = df['Sales_lag_1'] / (national_total_lag + EPSILON)
        df['Store_Rank_Lag'] = df.groupby('Date')['Sales_lag_1'].rank(ascending=False, pct=True)

        # Growth rates (clipped)
        df['Sales_Growth_4w'] = ((df['Sales_lag_1'] - df['Sales_lag_4'])
                                 / (np.abs(df['Sales_lag_4']) + EPSILON)).clip(-10, 10)
        df['Sales_Growth_YoY'] = ((df['Sales_lag_1'] - df['Sales_lag_52'])
                                  / (np.abs(df['Sales_lag_52']) + EPSILON)).clip(-10, 10)

        # Within-store dept share/rank (lag-based, per date)
        store_total_lag = df.groupby(['Store', 'Date'])['Sales_lag_1'].transform('sum')
        df['Dept_Share_Lag'] = df['Sales_lag_1'] / (store_total_lag + EPSILON)
        df['Dept_Rank_Lag'] = df.groupby(['Store', 'Date'])['Sales_lag_1'].rank(ascending=False)

        # Within-dept store rank (lag-based, per date)
        df['Store_Rank_in_Dept'] = df.groupby(['Dept', 'Date'])['Sales_lag_1'] \
                                     .rank(ascending=False, pct=True).fillna(0.5)

        # 6. Store type encoding (A > B > C by size)
        if 'Type' in df.columns:
            df['Type_encoded'] = df['Type'].map({'A': 2, 'B': 1, 'C': 0}).fillna(1).astype(int)

        # 7. Markdown features
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        present = [c for c in markdown_cols if c in df.columns]
        if present:
            df[present] = df[present].fillna(0)
            df['Markdown_Total'] = df[present].sum(axis=1)
            df['Has_Markdown'] = (df['Markdown_Total'] > 0).astype(int)

        logger.info(f"Created {len(df.columns)} features")
        return df

    # ==================================================================
    # FITTED STATISTICS (computed on train only in the leak-free path)
    # ==================================================================

    def _compute_dept_stats(self, df):
        """Department-level aggregates over LAGGED sales."""
        g = df.groupby('Dept')['Sales_lag_1']
        stats = pd.DataFrame({
            'Dept_Avg_Sales': g.mean(),
            'Dept_Std': g.std(),
            'Dept_Zero_Freq': g.apply(lambda x: (x == 0).mean()),
            'Dept_Total': g.sum(),
        })
        stats['Dept_CV'] = (stats['Dept_Std'] / (stats['Dept_Avg_Sales'] + EPSILON)).clip(0, 3)
        stats['grand_total'] = stats['Dept_Total'].sum()
        return stats

    def _apply_dept_stats(self, df, stats):
        logger.info("  Applying department statistics (leak-free)...")
        df = df.merge(
            stats[['Dept_Avg_Sales', 'Dept_CV', 'Dept_Zero_Freq', 'Dept_Total']],
            left_on='Dept', right_index=True, how='left')

        df['Sales_vs_Dept_Avg'] = (df['Sales_lag_1']
                                   / (df['Dept_Avg_Sales'] + EPSILON)).clip(0, 5)
        grand_total = stats['grand_total'].iloc[0] if len(stats) else 0
        df['Dept_Size_Indicator'] = df['Dept_Total'] / (grand_total + EPSILON)
        df = df.drop(columns=['Dept_Total'])

        # Depts unseen at fit time -> neutral values
        df['Dept_Avg_Sales'] = df['Dept_Avg_Sales'].fillna(0)
        df['Dept_CV'] = df['Dept_CV'].fillna(0)
        df['Dept_Zero_Freq'] = df['Dept_Zero_Freq'].fillna(0)
        df['Dept_Size_Indicator'] = df['Dept_Size_Indicator'].fillna(0)
        return df

    def _compute_fill_stats(self, df):
        lag_cols = [c for c in df.columns if 'lag' in c.lower() or 'roll' in c.lower()]
        group_means = df.groupby(['Store', 'Dept'])[lag_cols].mean()
        global_means = df[lag_cols].mean()
        return {'lag_cols': lag_cols, 'group_means': group_means, 'global_means': global_means}

    def _handle_missing_values(self, df, fill_stats=None):
        """Fill NaN: ffill within series (past-only), then fitted group/global means."""
        if fill_stats is None:
            fill_stats = self._compute_fill_stats(df)

        lag_cols = [c for c in fill_stats['lag_cols'] if c in df.columns]

        # Forward fill within each series - uses only the past
        df[lag_cols] = df.groupby(['Store', 'Dept'])[lag_cols].ffill()

        # Remaining NaN (series starts) -> fitted per-series means
        gm = fill_stats['group_means']
        idx = pd.MultiIndex.from_arrays([df['Store'], df['Dept']])
        for col in lag_cols:
            if df[col].isna().any():
                mapped = gm[col].reindex(idx).to_numpy()
                df[col] = df[col].fillna(pd.Series(mapped, index=df.index))
                df[col] = df[col].fillna(fill_stats['global_means'].get(col, 0.0))
                df[col] = df[col].fillna(0)

        # Economic features -> forward/backward fill within series
        econ_cols = ['CPI', 'Unemployment', 'Temperature', 'Fuel_Price']
        for col in econ_cols:
            if col in df.columns:
                df[col] = df.groupby(['Store', 'Dept'])[col].ffill().bfill()

        df = df.fillna(0)
        return df

    def _compute_clip_bounds(self, df):
        skip = {'Store', 'Dept', 'Weekly_Sales', 'Size'}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
        return {c: (df[c].quantile(0.01), df[c].quantile(0.99)) for c in numeric_cols}

    def _validate_and_clean(self, df, clip_bounds=None):
        """Remove Inf rows, clip extremes to fitted 1%-99% bounds, final NaN sweep."""
        logger.info("Validating data...")

        float_cols = df.select_dtypes(include=[np.floating]).columns
        if len(float_cols) > 0:
            inf_mask = np.isinf(df[float_cols].values).any(axis=1)
            if inf_mask.sum() > 0:
                logger.warning(f"  Found {inf_mask.sum()} rows with Infinity -> Removing")
                df = df[~inf_mask].reset_index(drop=True)

        if df.isnull().any().any():
            df = df.fillna(0)

        for col in float_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0)

        if clip_bounds is None:
            clip_bounds = self._compute_clip_bounds(df)
        for col, (lo, hi) in clip_bounds.items():
            if col in df.columns and pd.notna(lo) and pd.notna(hi):
                df[col] = df[col].clip(lo, hi)

        if len(float_cols) > 0 and np.isinf(df[float_cols].values).any():
            logger.error("Still has Infinity after cleaning!")
        else:
            logger.info("  No Infinity values")
        if df.isnull().any().any():
            logger.error("Still has NaN after cleaning!")
        else:
            logger.info("  No NaN values")

        return df
