# src/powerbi_export.py
"""
Power BI data layer - exports a small star schema into results/powerbi/:

    fact_forecast.csv    one row per (Store, Dept, Date) in the test window
    fact_inventory.csv   one row per (Store, Dept) with ABC-XYZ policy fields
    dim_store.csv        store attributes (type, size, cluster, region)
    dim_date.csv         calendar attributes for the test window
    measures.dax         ready-to-paste DAX measures

Import the CSVs in Power BI Desktop, relate dim_store[Store] and
dim_date[Date] to the fact tables (one-to-many), paste the measures, and
build visuals. The finished report lives at powerbi/walmart_forecasting.pbix.

Two gotchas hit on real Power BI Desktop builds:
- Non-English Windows locales read "16065.49" as 1,606,549 - set
  "Locale for import" to English (US) before importing.
- "Mark as date table" requires a gap-free calendar, which is why dim_date
  contains every calendar day rather than only the weekly Fridays.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .utils import logger

OUT_DIR = Path('results/powerbi')

DAX_MEASURES = """\
// ============ Paste into Power BI: Modeling > New measure ============
// Core totals
Total Actual = SUM ( fact_forecast[Actual_Sales] )
Total Forecast = SUM ( fact_forecast[Forecast_Sales] )
Total Abs Error = SUM ( fact_forecast[Abs_Error] )

// Accuracy
Forecast Accuracy % = 1 - DIVIDE ( [Total Abs Error], [Total Actual] )
Forecast Bias % = DIVIDE ( [Total Forecast], [Total Actual] ) - 1
MAE = AVERAGE ( fact_forecast[Abs_Error] )
WMAE =
DIVIDE (
    SUMX ( fact_forecast, fact_forecast[Abs_Error] * IF ( fact_forecast[IsHoliday], 5, 1 ) ),
    SUMX ( fact_forecast, IF ( fact_forecast[IsHoliday], 5, 1 ) )
)

// Inventory
Avg Service Level % = AVERAGE ( fact_inventory[Service_Level] ) / 100
Series Needing Restock = CALCULATE ( COUNTROWS ( fact_inventory ), fact_inventory[Restock_Recommended] = 1 )
Total Restock Value = CALCULATE ( SUM ( fact_inventory[Restock_Value] ), fact_inventory[Restock_Recommended] = 1 )
Avg Stockout Rate % = AVERAGE ( fact_inventory[Stockout_Rate] ) / 100
"""


def export_powerbi_data(eval_data, inventory=None, processed_data=None):
    """Write the star-schema CSVs + DAX file. Never raises - export must not break runs."""
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # ---------- fact_forecast ----------
        fact = pd.DataFrame({
            'Date': pd.to_datetime(eval_data['Date']),
            'Store': eval_data['Store'],
            'Dept': eval_data['Dept'],
            'Actual_Sales': np.round(np.asarray(eval_data['y_true'], dtype=float), 2),
            'Forecast_Sales': np.round(np.asarray(eval_data['y_pred'], dtype=float), 2),
            'IsHoliday': np.asarray(eval_data['is_holiday']).astype(bool),
        })
        fact['Abs_Error'] = (fact['Actual_Sales'] - fact['Forecast_Sales']).abs().round(2)
        fact['Error'] = (fact['Actual_Sales'] - fact['Forecast_Sales']).round(2)
        fact.to_csv(OUT_DIR / 'fact_forecast.csv', index=False)

        # ---------- fact_inventory ----------
        if inventory and inventory.get('store_dept_full') is not None:
            inv = inventory['store_dept_full'][[
                'Store', 'Dept', 'ABC', 'XYZ',
                'Avg_Weekly_Sales', 'Total_Actual_Sales',
                'Stockout_Rate', 'Service_Level',
                'Recommended_Safety_Stock', 'Restock_Quantity', 'Restock_Recommended',
            ]].rename(columns={
                'Recommended_Safety_Stock': 'Safety_Stock',
                'Restock_Quantity': 'Restock_Value',
            }).round(2)
            inv['ABC_XYZ'] = inv['ABC'] + inv['XYZ']
            inv.to_csv(OUT_DIR / 'fact_inventory.csv', index=False)

        # ---------- dim_store ----------
        if processed_data is not None:
            cols = [c for c in ['Type', 'Size', 'Region', 'Cluster'] if c in processed_data.columns]
            dim_store = (processed_data.groupby('Store')[cols].first().reset_index())
            dim_store.to_csv(OUT_DIR / 'dim_store.csv', index=False)

        # ---------- dim_date ----------
        # Power BI's "Mark as date table" requires EVERY calendar day present
        # (no gaps) between min and max - fact_forecast only has one row per
        # week (Fridays), so a full daily range is generated here and the
        # one-to-many relationship still works: each weekly fact row matches
        # its single corresponding day in this daily calendar.
        full_range = pd.date_range(fact['Date'].min(), fact['Date'].max(), freq='D')
        dates = pd.DataFrame({'Date': full_range})
        dates['Year'] = dates['Date'].dt.year
        dates['Month'] = dates['Date'].dt.month
        dates['Month_Name'] = dates['Date'].dt.strftime('%b')
        dates['Quarter'] = 'Q' + dates['Date'].dt.quarter.astype(str)
        dates['Week_Of_Year'] = dates['Date'].dt.isocalendar().week.astype(int)
        holiday_dates = set(fact.loc[fact['IsHoliday'], 'Date'].unique())
        dates['IsHoliday'] = dates['Date'].isin(holiday_dates)
        dates.to_csv(OUT_DIR / 'dim_date.csv', index=False)

        # ---------- DAX ----------
        (OUT_DIR / 'measures.dax').write_text(DAX_MEASURES, encoding='utf-8')

        logger.info(f'  ✓ Power BI data layer exported: {OUT_DIR}/ '
                    f'({len(fact):,} forecast rows)')
        return str(OUT_DIR)
    except Exception as e:
        logger.warning(f'Power BI export failed (non-fatal): {e}')
        return None
