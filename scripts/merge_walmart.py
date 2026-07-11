import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def merge_walmart_data():
    """
    ✨ CẢI TIẾN: Merge dữ liệu Walmart phù hợp với hệ thống đa cửa hàng
    
    Xử lý:
    1. Merge 3 files: train + stores + features
    2. Xử lý conflict trong IsHoliday
    3. Fill NaN thông minh cho MarkDown
    4. Tạo features bổ sung cho đa cửa hàng
    5. Validation đầy đủ
    """
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("WALMART DATA MERGING - MULTI-STORE SYSTEM")
    print("="*60)
    
    try:
        # ========== 1. LOAD DATA ==========
        print("\n[1/6] Loading raw data files...")
        
        train = pd.read_csv(raw_dir / 'train.csv', parse_dates=['Date'])
        stores = pd.read_csv(raw_dir / 'stores.csv')
        features = pd.read_csv(raw_dir / 'features.csv', parse_dates=['Date'])
        
        print(f"  ✓ train.csv: {train.shape}")
        print(f"  ✓ stores.csv: {stores.shape}")
        print(f"  ✓ features.csv: {features.shape}")
        
        # ========== 2. DATA QUALITY CHECKS ==========
        print("\n[2/6] Data quality checks...")
        
        # Check train
        assert 'Weekly_Sales' in train.columns, "Missing Weekly_Sales in train.csv"
        assert train['Weekly_Sales'].notna().all(), "NaN found in Weekly_Sales"
        print(f"  ✓ Train data: {train['Store'].nunique()} stores, {train['Dept'].nunique()} depts")
        
        # Check stores
        assert stores['Store'].nunique() == len(stores), "Duplicate stores in stores.csv"
        print(f"  ✓ Stores data: {len(stores)} unique stores")
        
        # Check features
        print(f"  ✓ Features data: {features['Store'].nunique()} stores covered")
        
        # ========== 3. MERGE STEP 1: Train + Stores ==========
        print("\n[3/6] Merging train.csv + stores.csv...")
        
        data = train.merge(stores, on='Store', how='left', validate='m:1')
        
        # Validation
        assert len(data) == len(train), "Row count mismatch after merge!"
        missing_stores = data['Type'].isna().sum()
        if missing_stores > 0:
            print(f"  ⚠ Warning: {missing_stores} rows missing store info")
        else:
            print(f"  ✓ Merged successfully: {data.shape}")
        
        # ========== 4. MERGE STEP 2: Data + Features (TRICKY!) ==========
        print("\n[4/6] Merging with features.csv...")
        
        # ✨ KEY FIX: Không merge theo IsHoliday (vì có conflict)
        # Chỉ merge theo Store + Date
        
        # Rename IsHoliday trong features để tránh conflict
        features_renamed = features.rename(columns={'IsHoliday': 'IsHoliday_features'})
        
        data = data.merge(
            features_renamed, 
            on=['Store', 'Date'], 
            how='left',
            validate='m:1'
        )
        
        print(f"  ✓ Merged with features: {data.shape}")
        
        # ========== 5. XỬ LÝ ISHOLIDAY CONFLICT ==========
        print("\n[5/6] Resolving IsHoliday conflicts...")
        
        # So sánh 2 cột IsHoliday
        if 'IsHoliday_features' in data.columns:
            conflicts = (data['IsHoliday'] != data['IsHoliday_features']).sum()
            print(f"  ℹ Found {conflicts} conflicts between train and features IsHoliday")
            
            # ✨ STRATEGY: Ưu tiên IsHoliday từ train.csv (chính xác hơn)
            # Nhưng nếu train=False và features=True → giữ train (conservative)
            data['IsHoliday_final'] = data['IsHoliday']
            
            # Drop duplicate column
            data = data.drop(columns=['IsHoliday_features'])
            data = data.rename(columns={'IsHoliday_final': 'IsHoliday'})
            
            print(f"  ✓ IsHoliday resolved: Using train.csv values")
        
        # ========== 6. XỬ LÝ NaN TRONG MARKDOWN (CRITICAL!) ==========
        print("\n[6/6] Handling MarkDown NaN values...")
        
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        
        for col in markdown_cols:
            if col in data.columns:
                nan_count = data[col].isna().sum()
                nan_pct = nan_count / len(data) * 100
                print(f"  • {col}: {nan_pct:.1f}% NaN")
                
                # ✨ STRATEGY: NaN = Không có promotion = 0
                data[col] = data[col].fillna(0)
        
        print(f"  ✓ MarkDown columns filled with 0")
        
        # ========== 7. THÊM FEATURES BỔ SUNG CHO ĐA CỬA HÀNG ==========
        print("\n[Bonus] Adding multi-store features...")
        
        # 7.1 Region (group stores)
        data['Region'] = 'R' + ((data['Store'] - 1) // 5 + 1).astype(str)
        print(f"  ✓ Created {data['Region'].nunique()} regions")
        
        # 7.2 Store-Dept ID (unique identifier)
        data['Store_Dept_ID'] = data['Store'].astype(str) + '_' + data['Dept'].astype(str)
        print(f"  ✓ Created Store_Dept_ID: {data['Store_Dept_ID'].nunique()} unique combinations")
        
        # 7.3 Date features (để dễ phân tích)
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Week'] = data['Date'].dt.isocalendar().week
        print(f"  ✓ Added temporal features: Year, Month, Week")
        
        # ========== 8. FINAL VALIDATION ==========
        print("\n" + "="*60)
        print("FINAL VALIDATION")
        print("="*60)
        
        # Check completeness
        print(f"Final shape: {data.shape}")
        print(f"Stores: {data['Store'].nunique()}")
        print(f"Departments: {data['Dept'].nunique()}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"Total weeks: {data['Date'].nunique()}")
        
        # Check NaN
        nan_summary = data.isnull().sum()
        critical_cols = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Type', 'Size']
        
        critical_nans = nan_summary[critical_cols]
        if critical_nans.sum() > 0:
            print("\n⚠ WARNING: NaN in critical columns!")
            print(critical_nans[critical_nans > 0])
        else:
            print("\n✓ No NaN in critical columns")
        
        # Check for other NaN
        other_nans = nan_summary[nan_summary > 0]
        if len(other_nans) > 0:
            print("\nNaN in other columns:")
            for col, count in other_nans.items():
                print(f"  • {col}: {count} ({count/len(data)*100:.1f}%)")
        
        # ========== 9. SAVE ==========
        output_path = processed_dir / 'walmart_clean.csv'
        data.to_csv(output_path, index=False)
        print(f"\n✓ Merged data saved to {output_path}")
        
        # ========== 10. SUMMARY STATISTICS ==========
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\n📊 Sales Statistics:")
        print(f"  Mean weekly sales: ${data['Weekly_Sales'].mean():,.2f}")
        print(f"  Median weekly sales: ${data['Weekly_Sales'].median():,.2f}")
        print(f"  Max weekly sales: ${data['Weekly_Sales'].max():,.2f}")
        print(f"  Min weekly sales: ${data['Weekly_Sales'].min():,.2f}")
        
        print(f"\n🏪 Store Statistics:")
        store_stats = data.groupby('Store')['Weekly_Sales'].agg(['count', 'mean', 'sum'])
        print(f"  Avg records per store: {store_stats['count'].mean():.0f}")
        print(f"  Avg weekly sales per store: ${store_stats['mean'].mean():,.2f}")
        
        print(f"\n🏷️ Department Statistics:")
        dept_stats = data.groupby('Dept')['Weekly_Sales'].agg(['count', 'mean'])
        print(f"  Total departments: {len(dept_stats)}")
        print(f"  Avg weekly sales per dept: ${dept_stats['mean'].mean():,.2f}")
        
        print(f"\n📅 Temporal Coverage:")
        print(f"  Years: {sorted(data['Year'].unique())}")
        print(f"  Total weeks: {data['Date'].nunique()}")
        print(f"  Holiday weeks: {data['IsHoliday'].sum()}")
        
        print("\n" + "="*60)
        print("✓ MERGE COMPLETE!")
        print("="*60)
        
        return data
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure these files exist in data/raw/:")
        print("  • train.csv")
        print("  • stores.csv")
        print("  • features.csv")
        print("\nDownload from: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data")
        return None
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_walmart_data(data_path):
    """
    ✨ MỚI: Validate dữ liệu đã merge
    """
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    issues = []
    
    # 1. Check required columns
    required_cols = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Type', 'Size', 
                     'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    else:
        print("✓ All required columns present")
    
    # 2. Check for duplicates
    dup_count = df.duplicated(subset=['Store', 'Dept', 'Date']).sum()
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicate records")
    else:
        print("✓ No duplicate records")
    
    # 3. Check negative sales
    neg_sales = (df['Weekly_Sales'] < 0).sum()
    if neg_sales > 0:
        issues.append(f"Found {neg_sales} negative sales values")
        print(f"⚠ {neg_sales} negative sales (may be returns)")
    else:
        print("✓ No negative sales")
    
    # 4. Check date continuity per store-dept
    print("\n📅 Checking date continuity...")
    gaps_found = 0
    
    for (store, dept), group in df.groupby(['Store', 'Dept']):
        dates = group['Date'].sort_values()
        date_diffs = dates.diff().dt.days.dropna()
        
        # Expected: 7 days between records (weekly data)
        if (date_diffs != 7).any():
            gaps_found += 1
    
    if gaps_found > 0:
        print(f"⚠ Found date gaps in {gaps_found} store-dept combinations")
    else:
        print("✓ No date gaps")
    
    # 5. Summary
    if issues:
        print("\n❌ VALIDATION FAILED:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("\n✓ VALIDATION PASSED!")
        return True


if __name__ == "__main__":
    # Merge data
    data = merge_walmart_data()
    
    if data is not None:
        # Validate
        validate_walmart_data('data/processed/walmart_clean.csv')
        
        print("\n🎉 Ready for modeling!")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Check: results/metrics/")
        print("  3. View: results/visualizations/")