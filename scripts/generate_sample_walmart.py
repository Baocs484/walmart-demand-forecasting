import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_walmart(n_stores=3, n_depts=5, n_days=300):  # Tăng n_days lên 200
    np.random.seed(42)
    start_date = datetime(2010, 2, 5)
    dates = [start_date + timedelta(days=7*i) for i in range(n_days)]
    rows = []
    
    for s in range(1, n_stores + 1):
        typ = np.random.choice(['A', 'B', 'C'])
        size = np.random.randint(100_000, 200_000)
        for p in range(1, n_depts + 1):
            base = np.random.randint(500, 5000)
            for d in dates:
                dow = d.weekday()
                m = d.month
                is_h = ((m == 12 and d.day in [24, 25, 31]) or 
                        (m == 11 and d.day in [25, 26]) or 
                        (m == 2 and d.day == 12))
                seasonal = 1.5 if m in [11, 12] else 1.0
                trend = 1 + (d - start_date).days / 365 * 0.1
                noise = np.random.normal(1, 0.1)
                sale = max(0, base * seasonal * trend * noise * (1.5 if is_h else 1.0))
                rows.append({
                    'Store': s,
                    'Dept': p,
                    'Date': d,
                    'Weekly_Sales': sale,
                    'IsHoliday': is_h,
                    'Type': typ,
                    'Size': size,
                    'Temperature': np.random.uniform(30, 90),
                    'Fuel_Price': np.random.uniform(2.5, 4.5),
                    'MarkDown1': np.random.uniform(0, 1000) if np.random.rand() > 0.5 else np.nan,
                    'MarkDown2': np.random.uniform(0, 1000) if np.random.rand() > 0.5 else np.nan,
                    'MarkDown3': np.random.uniform(0, 1000) if np.random.rand() > 0.5 else np.nan,
                    'MarkDown4': np.random.uniform(0, 1000) if np.random.rand() > 0.5 else np.nan,
                    'MarkDown5': np.random.uniform(0, 1000) if np.random.rand() > 0.5 else np.nan,
                    'CPI': np.random.uniform(100, 200),
                    'Unemployment': np.random.uniform(5, 10)
                })
    
    df = pd.DataFrame(rows)
    output_path = Path('data/sample/sample_walmart.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of stores: {df['Store'].nunique()}")
    print(f"Number of records: {len(df)}")
    return df

if __name__ == "__main__":
    generate_sample_walmart()