# Hướng Dẫn Dựng Dashboard Power BI

Dựng một dashboard Power BI 3 trang từ tầng dữ liệu do dự án xuất ra, mất khoảng
30–60 phút. Kết quả tương đương `results/dashboard.html` nhưng là một report
Power BI thật — một sản phẩm portfolio độc lập.

**Điều kiện:** chạy `python main.py train` trước. Lệnh này xuất tầng dữ liệu vào
`results/powerbi/`:

| File | Cấp độ chi tiết | Vai trò |
|------|-------|------|
| `fact_forecast.csv` | Store × Dept × Tuần | thực tế vs dự báo + sai số |
| `fact_inventory.csv` | Store × Dept | chính sách ABC-XYZ, thiếu hàng, giá trị nhập |
| `dim_store.csv` | Store | loại, quy mô, khu vực, cụm |
| `dim_date.csv` | Tuần | thuộc tính lịch, cờ ngày lễ |
| `measures.dax` | — | các công thức để dán (mỗi công thức 1 lần `New measure`) |

## 1. Nhập dữ liệu & dựng model (10 phút)

> **⚠️ Windows tiếng Việt: sửa định dạng số TRƯỚC TIÊN.**
> File CSV dùng dấu **chấm** làm dấu thập phân. Với bản địa tiếng Việt/châu Âu,
> Power BI sẽ đọc `16065.49` thành **1.606.549** — mọi con số bị phóng đại
> gấp 100 lần. Trước khi làm gì khác:
> **Tệp → Tùy chọn và cài đặt → Tùy chọn → Tệp hiện tại → Cài đặt khu vực →
> Bản địa để nhập: English (United States)**.
> Nếu đã lỡ nhập rồi: vào Power Query, chọn các cột số
> (`Actual_Sales`, `Forecast_Sales`, `Abs_Error`, `Error`, ...) → chuột phải →
> **Thay đổi loại → Sử dụng bản địa… → Số thập phân → English (United States)**.
> Kiểm tra: dòng đầu tiên của `fact_forecast` phải hiện Actual_Sales ≈ 16,065.49.

1. **Power BI Desktop → Get Data → Text/CSV** — nhập 4 file CSV
   (hoặc *Get Data → Folder* trỏ vào `results/powerbi/` rồi mở rộng).
2. Vào **Model view**, tạo các quan hệ (đều là một-nhiều, một chiều).
   Trong hộp thoại, **tự tay bấm chọn cột khóa ở CẢ HAI bảng preview**
   — Power BI đôi khi tự chọn nhầm cặp cột như `IsHoliday`↔`IsHoliday`,
   dẫn đến lỗi "giá trị trùng lặp" vì cột đó không phải khóa duy nhất:
   - `dim_store[Store]` 1—* `fact_forecast[Store]`
   - `dim_store[Store]` 1—* `fact_inventory[Store]`
   - `dim_date[Date]` 1—* `fact_forecast[Date]`

   **Nếu không thấy nút Lưu ở cuối hộp thoại**: phóng to cửa sổ Power BI Desktop
   hết cỡ, hoặc kéo giãn viền dưới/góc dưới-phải của hộp thoại; nếu vẫn không
   thấy, nhấn phím **Enter** (thường kích hoạt nút mặc định); nếu vẫn không được,
   thử giảm tỷ lệ hiển thị Windows (Settings → System → Display → Scale) xuống
   100% rồi mở lại hộp thoại.
3. Đánh dấu `dim_date` là **bảng ngày** (Table tools → Mark as date table).
4. Mở `measures.dax` bằng trình soạn thảo văn bản; với mỗi công thức:
   **Modeling → New measure**, dán từng khối, nhấn Enter. Định dạng các
   measure `%` thành phần trăm, các measure `$` thành tiền tệ (0 chữ số
   thập phân).

## 2. Trang 1 — Tổng Quan Điều Hành (15 phút)

| Thành phần | Visual | Trường / ghi chú |
|---------|--------|----------------|
| Hàng KPI | 5 Card visuals | `Total Actual`, `Forecast Accuracy %`, `Forecast Bias %`, `Avg Service Level %`, `Series Needing Restock` |
| Xu hướng | Line chart | X = `dim_date[Date]`, Y = `Total Actual` và `Total Forecast` — 2 đường. Tùy chọn: thêm `IsHoliday` vào tooltip |
| Xếp hạng cửa hàng | Bar chart | X = `Total Actual`, Y = `dim_store[Store]`, sắp giảm dần, lọc top 10 |
| Bộ lọc (1 hàng, trên cùng) | Slicer ×3 | `dim_store[Type]`, `dim_store[Cluster]`, `dim_date[Month_Name]` |

Mẹo thiết kế cho trông chuyên nghiệp: dùng 1 màu nhấn (`#2A78D6`) cho số liệu
thực tế, 1 màu phụ nhạt hơn cho dự báo; tiêu đề viết hoa chữ đầu câu; tắt
gridline gây rối (Format → Gridlines off); căn trái ô văn bản tiêu đề trang.

## 3. Trang 2 — Độ Chính Xác Dự Báo (10 phút)

| Thành phần | Visual | Trường |
|---------|--------|--------|
| Sai số theo tuần | Column chart | X = `Date`, Y = `MAE`. Tuần lễ sẽ nổi bật nếu thêm `dim_date[IsHoliday]` vào Legend |
| Sai số theo phòng ban | Bar chart (top N) | Y = `fact_forecast[Dept]`, X = `Total Abs Error`, Top N = 10 |
| Tuần lễ vs tuần thường | Clustered column | X = `fact_forecast[IsHoliday]`, Y = `MAE` |
| Scatter | Scatter chart | X = `Total Actual`, Y = `Total Forecast`, Details = `Store` — điểm dưới đường chéo là cửa hàng bị dự báo thiếu |

## 4. Trang 3 — Hành Động Tồn Kho (10 phút)

| Thành phần | Visual | Trường |
|---------|--------|--------|
| Ma trận ABC-XYZ | Matrix visual | Rows = `ABC`, Columns = `XYZ`, Values = đếm Store + `Total Restock Value` |
| Bảng ưu tiên | Table | Store, Dept, `ABC_XYZ`, `Stockout_Rate`, `Service_Level`, `Restock_Value`; lọc `Restock_Recommended = 1`; định dạng có điều kiện (data bars) trên `Stockout_Rate` |
| Hàng KPI | Cards | `Series Needing Restock`, `Total Restock Value`, `Avg Stockout Rate %` |

## 5. Hoàn thiện

- **File → Save as** `powerbi/walmart_forecasting.pbix` (thư mục này đã bị
  gitignore, trừ khi bạn chủ động commit file .pbix — thường nặng vài MB).
- Chụp màn hình Trang 1 để đưa vào README / CV.
- Tùy chọn: publish lên Power BI Service (gói miễn phí) và gắn link công khai
  vào README.

## Ghi chú

- Toàn bộ giá trị là **đô la ($)** — dataset gốc không có giá đơn vị sản phẩm.
- `WMAE` ở đây khớp với metric chính của dự án (tuần lễ trọng số ×5).
- Chạy lại `python main.py train` sẽ làm mới các file CSV; bấm **Refresh**
  trong Power BI để lấy dữ liệu mới.
