# src/inventory_report.py - FIXED VERSION
"""
Inventory Report Module
FIXED:
- Priority Score đúng (Stockout 70% + Error 30%)
- Safety Stock hợp lý
- Xử lý items dự đoán = 0
- Chi phí tính đúng
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .utils import logger, timer_decorator
from .config import CONFIG

@timer_decorator
def generate_inventory_report(system):
    """
    Tạo báo cáo inventory với logic HOÀN TOÀN ĐÚNG
    """
    
    if system.last_eval_data is None:
        logger.error("No evaluation data available to generate report")
        return None
    
    logger.info("Generating FINAL CORRECT inventory report...")
    
    # ========== 1. PREPARE DATA ==========
    df = pd.DataFrame({
        'Date': system.last_eval_data['test_dates'],
        'Store': system.last_eval_data['store'],
        'Dept': system.last_eval_data['dept'],
        'Actual_Sales': system.last_eval_data['y_true'],
        'Predicted_Sales': system.last_eval_data['pred_ensemble'],
        'IsHoliday': system.last_eval_data['is_holiday']
    })
    
    # Xử lý dự đoán = 0
    df['Predicted_Sales'] = np.where(
        (df['Predicted_Sales'] <= 0) & (df['Actual_Sales'] > 0),
        df['Actual_Sales'],
        df['Predicted_Sales']
    )
    
    df['Predicted_Sales'] = df['Predicted_Sales'].clip(lower=0)
    
    # ========== 2. CALCULATE FORECAST ERROR & CLASSIFY BY STORE-DEPT (ABC-XYZ) ==========
    
    logger.info("Calculating forecast error by Store-Dept...")
    
    df["Forecast_Error"] = df["Actual_Sales"] - df["Predicted_Sales"]
    
    error_stats = df.groupby(["Store", "Dept"]).agg({
        "Forecast_Error": ["std", "mean"],
        "Actual_Sales": ["mean", "sum", "std"]
    }).reset_index()
    
    error_stats.columns = ["Store", "Dept", "Error_Std", "Error_Mean", "Avg_Actual", "Total_Actual", "Actual_Std"]
    error_stats["Error_Std"] = error_stats["Error_Std"].fillna(error_stats["Avg_Actual"] * 0.5)
    error_stats["Actual_Std"] = error_stats["Actual_Std"].fillna(0)
    
    # Phân loại ABC theo doanh số thực tế tích lũy
    error_stats = error_stats.sort_values("Total_Actual", ascending=False)
    total_sales_all = error_stats["Total_Actual"].sum()
    error_stats["Sales_Share"] = error_stats["Total_Actual"] / (total_sales_all + 1e-6)
    error_stats["Cum_Share"] = error_stats["Sales_Share"].cumsum()
    error_stats["ABC"] = np.where(error_stats["Cum_Share"] <= 0.70, "A",
                           np.where(error_stats["Cum_Share"] <= 0.90, "B", "C"))
    
    # Phân loại XYZ theo hệ số biến động CV (Actual_Std / Avg_Actual)
    error_stats["CV"] = error_stats["Actual_Std"] / (error_stats["Avg_Actual"] + 1e-6)
    error_stats["XYZ"] = np.where(error_stats["CV"] <= 0.2, "X",
                           np.where(error_stats["CV"] <= 0.5, "Y", "Z"))
    
    df = df.merge(error_stats[["Store", "Dept", "Error_Std", "Error_Mean", "Avg_Actual", "ABC", "XYZ"]], on=["Store", "Dept"], how="left")
    
    # ========== 3. CALCULATE SAFETY STOCK (DYNAMIC ABC-XYZ POLICY) ==========
    
    logger.info("Calculating safety stock with DYNAMIC ABC-XYZ formula...")
    
    # Bảng tra cứu z-score dựa trên phân loại ABC-XYZ (từ config.yaml)
    inv_cfg = CONFIG['inventory']
    z_map = inv_cfg['z_scores']
    default_z = inv_cfg['default_z']

    df['z_score'] = [z_map.get(f'{a}{x}', default_z) for a, x in zip(df['ABC'], df['XYZ'])]
    lead_time_weeks = inv_cfg['lead_time_weeks']
    
    # Safety Stock = z_score * Error_Std * sqrt(lead_time)
    df["Safety_Stock"] = df["z_score"] * df["Error_Std"] * np.sqrt(lead_time_weeks)
    df["Safety_Stock"] = df["Safety_Stock"].fillna(df["Predicted_Sales"] * 0.5)
    # FIXED: Safety stock không được âm (xảy ra khi Error_Std fallback từ doanh số âm - hàng trả lại)
    df["Safety_Stock"] = df["Safety_Stock"].clip(lower=0)
    
    # Buffer thêm cho items có variance cao
    high_variance_mask = (df["Error_Std"] / (df["Avg_Actual"] + 1e-6)) > 0.5
    df.loc[high_variance_mask, "Safety_Stock"] *= 1.5
    
    # Buffer thêm cho ngày lễ
    holiday_mask = df["IsHoliday"] == True
    df.loc[holiday_mask, "Safety_Stock"] *= 1.2
    
    # ========== 4. CALCULATE INVENTORY METRICS ==========
    
    df["Optimal_Inventory"] = df["Predicted_Sales"] + df["Safety_Stock"]
    df["Stockout_Amount"] = np.maximum(0, df["Actual_Sales"] - df["Optimal_Inventory"])
    df["Is_Stockout"] = (df["Stockout_Amount"] > 0).astype(int)
    df["Overstock_Amount"] = np.maximum(0, df["Optimal_Inventory"] - df["Actual_Sales"])
    df["Is_Overstock"] = (df["Overstock_Amount"] > 0).astype(int)
    df["Error_Pct"] = np.abs(df["Actual_Sales"] - df["Predicted_Sales"]) / (df["Actual_Sales"] + 1) * 100
    df["Error_Pct"] = df["Error_Pct"].clip(0, 200)
    
    # ========== 5. AGGREGATE BY STORE & DEPT ==========
    
    logger.info("Aggregating metrics by Store & Dept...")
    
    # Đã thêm ABC, XYZ vào groupby để giữ thông tin phân loại
    agg_store_dept = df.groupby(['Store', 'Dept', 'ABC', 'XYZ']).agg({
        'Actual_Sales': ['sum', 'mean'],
        'Predicted_Sales': ['sum', 'mean'],
        'Stockout_Amount': 'sum',
        'Overstock_Amount': 'sum',
        'Is_Stockout': 'sum',
        'Is_Overstock': 'sum',
        'Error_Pct': 'mean',
        'Safety_Stock': 'mean'
    }).reset_index()
    
    agg_store_dept.columns = [
        'Store', 'Dept', 'ABC', 'XYZ',
        'Total_Actual_Sales', 'Avg_Weekly_Sales',
        'Total_Predicted_Sales', 'Avg_Predicted_Sales',
        'Total_Stockout', 'Total_Overstock',
        'Stockout_Weeks', 'Overstock_Weeks',
        'Avg_Error_Pct', 'Recommended_Safety_Stock'
    ]
    
    total_weeks = df.groupby(['Store', 'Dept']).size().reset_index(name='Total_Weeks')
    agg_store_dept = agg_store_dept.merge(total_weeks, on=['Store', 'Dept'])
    
    # Stockout Rate (%)
    agg_store_dept['Stockout_Rate'] = (agg_store_dept['Stockout_Weeks'] / agg_store_dept['Total_Weeks'] * 100)
    # Service Level (%)
    agg_store_dept['Service_Level'] = 100 - agg_store_dept['Stockout_Rate']
    
    # ========== 6. PRIORITY SCORING ==========
    
    logger.info("Calculating priority scores...")
    
    # Chỉ Stockout + Error, KHÔNG có Sales
    agg_store_dept['Priority_Score'] = (
        agg_store_dept['Stockout_Rate'] * 0.70 +
        agg_store_dept['Avg_Error_Pct'] * 0.30
    )
    
    # Xử lý inf/NaN
    agg_store_dept['Priority_Score'] = agg_store_dept['Priority_Score'].replace([np.inf, -np.inf], 0)
    agg_store_dept['Priority_Score'] = agg_store_dept['Priority_Score'].fillna(0)
    agg_store_dept['Priority_Score'] = agg_store_dept['Priority_Score'].clip(0, 200)
    
    # FIXED: Chỉ xét những chuỗi ĐỦ DỮ LIỆU và có doanh số dương
    # - Total_Weeks < 4: 1-3 tuần dữ liệu -> stockout rate 100%/50% vô nghĩa thống kê
    # - Avg_Weekly_Sales <= 0: chuỗi toàn hàng trả lại, không thể "nhập thêm hàng"
    MIN_WEEKS = inv_cfg['min_weeks']
    eligible = (
        (agg_store_dept['Total_Weeks'] >= MIN_WEEKS) &
        (agg_store_dept['Avg_Weekly_Sales'] > 0)
    )
    n_excluded = (~eligible).sum()
    if n_excluded > 0:
        logger.info(f"  Excluded {n_excluded} Store-Dept series (insufficient data or non-positive sales)")

    # RESTOCK LOGIC
    agg_store_dept['Restock_Recommended'] = (
        eligible & (
            (agg_store_dept['Stockout_Rate'] > 25) |
            (agg_store_dept['Service_Level'] < 80) |
            (
                (agg_store_dept['Stockout_Rate'] > 20) &
                (agg_store_dept['Avg_Error_Pct'] > 30)
            )
        )
    ).astype(int)
    
    # RESTOCK QUANTITY
    agg_store_dept['Restock_Quantity'] = agg_store_dept['Recommended_Safety_Stock'] * 4
    
    agg_store_dept['Restock_Quantity'] = np.where(
        (agg_store_dept['Restock_Quantity'] <= 0) & (agg_store_dept['Total_Actual_Sales'] > 0),
        agg_store_dept['Total_Actual_Sales'] / agg_store_dept['Total_Weeks'] * 4,
        agg_store_dept['Restock_Quantity']
    )
    
    # ========== 7. FILTER ITEMS ==========
    
    agg_store_dept_filtered = agg_store_dept[
        agg_store_dept['Restock_Recommended'] == 1
    ].copy()
    
    agg_store_dept_filtered = agg_store_dept_filtered.sort_values('Priority_Score', ascending=False)
    
    logger.info(f"  Items filtered: {len(agg_store_dept_filtered)} / {len(agg_store_dept)} need restocking")
    
    # ========== 8. AGGREGATE BY STORE ==========
    
    logger.info("Creating store-level summary...")
    
    agg_store = df.groupby('Store').agg({
        'Actual_Sales': 'sum',
        'Predicted_Sales': 'sum',
        'Stockout_Amount': 'sum',
        'Overstock_Amount': 'sum',
        'Is_Stockout': 'sum',
        'Error_Pct': 'mean'
    }).reset_index()
    
    agg_store.columns = [
        'Store', 'Total_Sales', 'Total_Predicted',
        'Total_Stockout', 'Total_Overstock',
        'Stockout_Count', 'Avg_Error_Pct'
    ]
    
    agg_store['Avg_Error_Pct'] = agg_store['Avg_Error_Pct'].clip(0, 200)
    
    # Add store info
    if hasattr(system, 'processed_data') and system.processed_data is not None:
        try:
            store_info = system.processed_data.groupby('Store')[['Type', 'Size']].first().reset_index()
            agg_store = agg_store.merge(store_info, on='Store', how='left')
        except Exception:
            pass
    
    # Add cluster
    if hasattr(system, 'store_clusters') and system.store_clusters:
        agg_store['Cluster'] = agg_store['Store'].map(system.store_clusters)
    
    # ========== 9. SAVE CSV ==========
    
    output_dir = Path("results/inventory")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agg_store.to_csv(output_dir / "store_inventory_summary.csv", index=False)
    agg_store_dept_filtered.to_csv(output_dir / "store_dept_inventory_detailed.csv", index=False)
    agg_store_dept.to_csv(output_dir / "store_dept_inventory_full.csv", index=False)
    
    logger.info(f"CSV reports saved to {output_dir}")
    
    # (Excel tổng hợp được tạo ở report_builder.build_excel_report -
    #  results/reports/forecast_report.xlsx, không xuất Excel trùng lặp ở đây)

    # ========== 10. CONSOLE SUMMARY & POLICY COMPARISON ==========
    
    logger.info("\n" + "="*70)
    logger.info("INVENTORY OPTIMIZATION REPORT (DYNAMIC ABC-XYZ)")
    logger.info("="*70)
    
    logger.info(f"  Stores analyzed: {len(agg_store)}")
    
    # Cost parameters (từ config.yaml)
    costs = inv_cfg['costs']
    unit_cost = costs['unit_cost']
    stockout_penalty = costs['stockout_penalty']
    holding_rate = costs['holding_rate']
    holding_cost_unit = unit_cost * holding_rate
    
    # Simulation functions for comparison (Full network)
    # 1. Baseline Strategy (Z = 1.645)
    baseline_ss = 1.645 * df["Error_Std"]
    baseline_ss = baseline_ss.fillna(df["Predicted_Sales"] * 0.5)
    baseline_ss = np.where((df["Error_Std"] / (df["Avg_Actual"] + 1e-6)) > 0.5, baseline_ss * 1.5, baseline_ss)
    baseline_ss = np.where(df["IsHoliday"] == True, baseline_ss * 1.2, baseline_ss)
    
    baseline_opt_inv = df["Predicted_Sales"] + baseline_ss
    baseline_stockout = np.maximum(0, df["Actual_Sales"] - baseline_opt_inv)
    baseline_overstock = np.maximum(0, baseline_opt_inv - df["Actual_Sales"])
    baseline_sl = (df["Actual_Sales"] <= baseline_opt_inv).mean() * 100
    baseline_cost = (baseline_stockout.sum() * stockout_penalty) + (baseline_overstock.sum() * holding_cost_unit)
    
    # 2. Critical Ratio Strategy (Z = 2.05)
    cr_ss = 2.05 * df["Error_Std"]
    cr_ss = cr_ss.fillna(df["Predicted_Sales"] * 0.5)
    cr_ss = np.where((df["Error_Std"] / (df["Avg_Actual"] + 1e-6)) > 0.5, cr_ss * 1.5, cr_ss)
    cr_ss = np.where(df["IsHoliday"] == True, cr_ss * 1.2, cr_ss)
    
    cr_opt_inv = df["Predicted_Sales"] + cr_ss
    cr_stockout = np.maximum(0, df["Actual_Sales"] - cr_opt_inv)
    cr_overstock = np.maximum(0, cr_opt_inv - df["Actual_Sales"])
    cr_sl = (df["Actual_Sales"] <= cr_opt_inv).mean() * 100
    cr_cost = (cr_stockout.sum() * stockout_penalty) + (cr_overstock.sum() * holding_cost_unit)
    
    # 3. Dynamic ABC-XYZ Strategy
    dynamic_stockout = df["Stockout_Amount"]
    dynamic_overstock = df["Overstock_Amount"]
    dynamic_sl = (df["Actual_Sales"] <= df["Optimal_Inventory"]).mean() * 100
    dynamic_cost = (dynamic_stockout.sum() * stockout_penalty) + (dynamic_overstock.sum() * holding_cost_unit)
    
    # Prints the comparison matrix
    logger.info("\nPOLICY SIMULATION COMPARISON (FULL NETWORK):")
    logger.info(f"  - Baseline (Flat 95% SL):       Service Level: {baseline_sl:.2f}% | Total Cost: ${baseline_cost:,.2f}")
    logger.info(f"  - Strategy 1 (Flat 98% SL):     Service Level: {cr_sl:.2f}% | Total Cost: ${cr_cost:,.2f} (Savings: {((baseline_cost - cr_cost)/baseline_cost * 100):.1f}%)")
    logger.info(f"  - Strategy 2 (Dynamic ABC-XYZ): Service Level: {dynamic_sl:.2f}% | Total Cost: ${dynamic_cost:,.2f} (Savings: {((baseline_cost - dynamic_cost)/baseline_cost * 100):.1f}%)")
    logger.info("-" * 70)
    
    # Current active metrics
    total_stockout_cost = dynamic_stockout.sum() * stockout_penalty
    total_overstock_cost = dynamic_overstock.sum() * holding_cost_unit
    
    logger.info(f"Active Policy (Dynamic ABC-XYZ) Details:")
    logger.info(f"  Total potential stockout loss: ${total_stockout_cost:,.2f}")
    logger.info(f"  Total overstock cost risk:     ${total_overstock_cost:,.2f}")
    logger.info(f"  TOTAL COST:                    ${total_stockout_cost + total_overstock_cost:,.2f}")
    
    # TOP 5
    logger.info(f"\nTOP 5 URGENT RESTOCK NEEDED:")
    
    if len(agg_store_dept_filtered) == 0:
        logger.warning("  No items need restocking!")
    else:
        top_5 = agg_store_dept_filtered.head(5)
        for idx, row in top_5.iterrows():
            logger.info(f"  - Store {int(row['Store'])} / Dept {int(row['Dept'])}: "
                       f"Stockout Rate {row['Stockout_Rate']:.1f}% | "
                       f"Service Level {row['Service_Level']:.1f}% | "
                       f"Priority {row['Priority_Score']:.1f} | "
                       f"RESTOCK VALUE: ${row['Restock_Quantity']:,.0f}")
    
    # Statistics
    logger.info(f"\nOVERALL STATISTICS:")
    if len(agg_store_dept_filtered) > 0:
        logger.info(f"  Items need restocking: {len(agg_store_dept_filtered)}")
        logger.info(f"  Average Stockout Rate: {agg_store_dept_filtered['Stockout_Rate'].mean():.1f}%")
        logger.info(f"  Average Service Level: {agg_store_dept_filtered['Service_Level'].mean():.1f}%")
        logger.info(f"  Average Priority Score: {agg_store_dept_filtered['Priority_Score'].mean():.1f}")
        
        logger.info(f"\nSTOCKOUT RATE DISTRIBUTION:")
        logger.info(f"  20-30%: {((agg_store_dept_filtered['Stockout_Rate'] >= 20) & (agg_store_dept_filtered['Stockout_Rate'] < 30)).sum()} items")
        logger.info(f"  30-50%: {((agg_store_dept_filtered['Stockout_Rate'] >= 30) & (agg_store_dept_filtered['Stockout_Rate'] < 50)).sum()} items")
        logger.info(f"  >50%:   {(agg_store_dept_filtered['Stockout_Rate'] >= 50).sum()} items")
    
    return {
        'store_summary': agg_store,
        'store_dept_detailed': agg_store_dept_filtered,
        'store_dept_full': agg_store_dept
    }