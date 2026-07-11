import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ngưỡng chấp nhận (Acceptance Thresholds)
THRESHOLDS = {
    'MAPE': {'excellent': 5, 'good': 10, 'acceptable': 20},
    'Service_Level': {'minimum': 95, 'target': 98},
    'Stockout_Rate': {'maximum': 5, 'target': 2}
}

class MetricsCalculator:
    """
    Lớp (class) tĩnh chứa các phương thức để tính toán
    chỉ số lỗi và chỉ số nghiệp vụ.
    """

    @staticmethod
    def calculate_metrics(actual, predicted, sample_weight=None):
        """
        1. Accuracy Metrics
        Tính toán các chỉ số lỗi dự đoán (so sánh thực tế và dự đoán).

        Bao gồm: MAE, RMSE, MAPE, sMAPE, và WMAE (nếu có trọng số).
        """
        a = np.array(actual)
        p = np.array(predicted)

        # MAE (Mean Absolute Error): Lỗi tuyệt đối trung bình
        mae = mean_absolute_error(a, p)
        
        # RMSE (Root Mean Square Error): Căn bậc hai lỗi bình phương trung bình
        rmse = np.sqrt(mean_squared_error(a, p))
        
        # MAPE (Mean Absolute Percentage Error): Phần trăm lỗi tuyệt đối trung bình
        # FIXED: Lọc giá trị a > 100 để tránh MAPE bị thổi phồng bởi sales gần 0
        mask = a > 100
        mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.sum() > 0 else np.nan
        
        # sMAPE (Symmetric MAPE): MAPE đối xứng, thêm 0.01 để ổn định
        smape = np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p) + 0.01)) * 100

        # WMAE (Weighted MAE): MAE có trọng số (dùng cho các ngày lễ)
        if sample_weight is not None:
            wmae = (np.sum(sample_weight * np.abs(a - p)) / np.sum(sample_weight))
        else:
            wmae = mae  # Nếu không có trọng số, WMAE = MAE

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape, 'WMAE': wmae}

    @staticmethod
    def calculate_business_metrics(actual, predicted, unit_cost=10, holding_rate=0.01, safety_factor=1.2):
        """
        2. Business Metrics
        Tính toán các chỉ số nghiệp vụ (chi phí, dịch vụ) dựa trên dự đoán.
        
        Args:
            unit_cost (float): Giá vốn/chi phí trên mỗi đơn vị sản phẩm.
            holding_rate (float): Tỷ lệ chi phí lưu kho (ví dụ: 1% giá vốn).
            safety_factor (float): Hệ số an toàn để điều chỉnh dự báo, giảm stockout.
        """
        a = np.array(actual)
        p = np.array(predicted) * safety_factor  # Tăng dự báo để giảm stockout

        # Stock-out Rate: Tỷ lệ thiếu hàng (dự đoán < thực tế)
        stockout_rate = float(np.mean(a > p) * 100)

        # Overstock Cost: Chi phí tồn kho dư thừa (dự đoán > thực tế)
        over = np.maximum(0, p - a)
        over_cost = float(np.sum(over * unit_cost * holding_rate))

        # Service Level: Mức độ đáp ứng nhu cầu (dự đoán >= thực tế)
        service = float(np.mean(p >= a) * 100)

        # Stockout Cost: Chi phí thiếu hàng (mất doanh thu)
        stockout_cost = float(np.sum(np.maximum(0, a - p) * unit_cost * 5))

        # Total Cost: Tổng chi phí (tồn kho + thiếu hàng)
        total = float(over_cost + stockout_cost)

        return {'Stockout_Rate': stockout_rate, 'Overstock_Cost': over_cost,
                'Service_Level': service, 'Total_Cost': total, 'Stockout_Cost': stockout_cost}