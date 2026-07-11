# src/forecaster.py
"""
Recursive multi-week forecasting.

The models are 1-step-ahead by construction (they rely on Sales_lag_1 = last
week's actual sales). To forecast H weeks into the future we iterate:

    predict week t+1  ->  append the prediction as if it were actual sales
    ->  rebuild lag features  ->  predict week t+2  ->  ...

Predictions therefore compound their own uncertainty - expect accuracy to
degrade with horizon. That is the honest trade-off of recursive forecasting.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from .utils import logger

# Walmart holiday weeks (week-ending Fridays, per the Kaggle competition)
WALMART_HOLIDAY_WEEKS = {
    # Super Bowl
    '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',
    # Labor Day
    '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06',
    # Thanksgiving
    '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29',
    # Christmas
    '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27',
}
WALMART_HOLIDAY_WEEKS = {pd.Timestamp(d) for d in WALMART_HOLIDAY_WEEKS}

# Columns carried forward from each series' last observed row
CARRY_FORWARD_COLS = ['Type', 'Size', 'Temperature', 'Fuel_Price',
                      'CPI', 'Unemployment', 'Region']
MARKDOWN_COLS = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

# Keep just enough history for lag_52 + rolling windows
HISTORY_WEEKS = 56


class RecursiveForecaster:
    def __init__(self, model, processor, clustering):
        if not processor.is_fitted:
            raise ValueError('DataProcessor must be fitted (train first)')
        self.model = model
        self.processor = processor
        self.clustering = clustering

    def forecast(self, df_raw_history, horizon_weeks=4):
        """
        Args:
            df_raw_history: raw dataframe (same schema as walmart_clean.csv)
            horizon_weeks: how many weeks beyond the last date to forecast

        Returns:
            DataFrame [Date, Store, Dept, Forecast_Weekly_Sales, IsHoliday, Horizon]
        """
        last_date = df_raw_history['Date'].max()
        logger.info(f'Recursive forecast: {horizon_weeks} weeks beyond {last_date:%Y-%m-%d}')

        # Trim history: enough for lag_52, keeps transform fast
        min_date = last_date - timedelta(weeks=HISTORY_WEEKS)
        history = df_raw_history[df_raw_history['Date'] >= min_date].copy()

        # Template: one future row per active series, seeded from its last row
        last_rows = (history.sort_values('Date')
                     .groupby(['Store', 'Dept'], as_index=False).tail(1))
        keep = ['Store', 'Dept'] + [c for c in CARRY_FORWARD_COLS if c in last_rows.columns]
        template = last_rows[keep].reset_index(drop=True)

        forecasts = []
        for h in range(1, horizon_weeks + 1):
            next_date = last_date + timedelta(weeks=h)
            is_holiday = next_date.normalize() in WALMART_HOLIDAY_WEEKS

            new_rows = template.copy()
            new_rows['Date'] = next_date
            new_rows['IsHoliday'] = is_holiday
            new_rows['Weekly_Sales'] = np.nan
            for c in MARKDOWN_COLS:
                if c in history.columns:
                    new_rows[c] = 0.0

            combined = pd.concat([history, new_rows], ignore_index=True, sort=False)
            processed = self.processor.transform(combined)

            target = processed[processed['Date'] == next_date].copy()
            target['Cluster'] = target['Store'].map(self.clustering.get_cluster)

            preds = self.model.predict(target)
            target = target[['Store', 'Dept']].assign(
                Date=next_date,
                Forecast_Weekly_Sales=preds,
                IsHoliday=is_holiday,
                Horizon=h,
            )
            forecasts.append(target)

            # Feed predictions back as "actuals" for the next iteration
            fed = new_rows.merge(target[['Store', 'Dept', 'Forecast_Weekly_Sales']],
                                 on=['Store', 'Dept'], how='left')
            fed['Weekly_Sales'] = fed['Forecast_Weekly_Sales'].fillna(0)
            fed = fed.drop(columns=['Forecast_Weekly_Sales'])
            history = pd.concat([history, fed], ignore_index=True, sort=False)

            total = target['Forecast_Weekly_Sales'].sum()
            logger.info(f'  Week +{h} ({next_date:%Y-%m-%d})'
                        f'{" [HOLIDAY]" if is_holiday else ""}: '
                        f'total forecast ${total:,.0f} across {len(target):,} series')

        result = pd.concat(forecasts, ignore_index=True)
        result = result[['Date', 'Store', 'Dept', 'Horizon', 'IsHoliday',
                         'Forecast_Weekly_Sales']]
        return result
