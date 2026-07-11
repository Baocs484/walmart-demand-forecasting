import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Cấu hình sys.path để import được src khi chạy từ trong thư mục scripts
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import DataProcessor từ thư mục src
from src.data_processor import DataProcessor
from src.utils import logger

logger.info("--- Starting Correlation Analysis ---")

# Đảm bảo thư mục results/visualizations tồn tại
os.makedirs("results/visualizations", exist_ok=True)

# 1. Tải và xử lý dữ liệu (giống hệt main.py)
data_processor = DataProcessor()
data_path = BASE_DIR / "data" / "processed" / "walmart_clean.csv" 

if not data_path.exists():
    logger.error(f"Error: File not found at {data_path}")
else:
    # Tải và xử lý features (bao gồm cả các feature gây nhiễu)
    # Chúng ta cần TẤT CẢ các feature để so sánh
    df = data_processor.load_and_process_data(str(data_path))
    logger.info(f"Data loaded. Shape: {df.shape}")

    # 2. Chỉ chọn các cột số học để tính tương quan
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 3. Tính toán ma trận tương quan
    corr_matrix = numeric_df.corr()
    
    # 4. Lọc ra các tương quan CHỈ VỚI 'Weekly_Sales'
    # Đây là bước quan trọng nhất
    corr_target = corr_matrix[['Weekly_Sales']].sort_values(by='Weekly_Sales', 
                                                           ascending=False)
    
    # 5. Vẽ Heatmap
    logger.info("Generating heatmap...")
    plt.figure(figsize=(10, 15)) # Tăng kích thước để đọc được hết
    
    sns.heatmap(
        corr_target, 
        annot=True,     # Hiển thị số (giá trị tương quan)
        fmt=".2f",      # Định dạng 2 chữ số
        cmap='vlag',    # Dùng màu 'vlag' (Đỏ = dương, Xanh = âm, Trắng = 0)
        vmin=-1,        # Đảm bảo 0 là màu trắng
        vmax=1
    )
    
    plt.title('Heatmap Tương Quan với Doanh số (Weekly_Sales)', fontsize=16)
    plt.tight_layout()
    
    # 6. Lưu và hiển thị
    output_path = "results/visualizations/correlation_heatmap.png"
    plt.savefig(output_path)
    logger.info(f"Heatmap saved to {output_path}")
    
    plt.show() # Hiển thị biểu đồ
    
    logger.info("--- Analysis Complete ---")