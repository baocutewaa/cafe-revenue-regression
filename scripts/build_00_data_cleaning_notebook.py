"""Build and execute notebooks/00_data_cleaning_and_verification.ipynb.

Covers Giai đoạn 0 — Làm sạch có kiểm chứng:
1. Executive summary & 3-layer data architecture (raw -> clean -> features).
2. Data profiling & Data Quality Register (DQ-01 to DQ-15).
3. Automated cleaning pipeline execution with strict data contracts:
   - Exact deduplication (132 rows removed -> 26,500 clean rows)
   - S12 unit error correction (/1000 for March 2024, 30 rows)
   - Outlier detection (50 high spikes ~10x, 25 low spikes ~0.1x, 5 review rows)
   - Weather Fahrenheit correction (22 rows in HCM 10/06-01/07/2025) & full grid (1,642 rows)
   - Stores metadata standardization & area_m2 imputation (S19, S24)
   - Holidays daily aggregation & Tet festival flags
4. Comprehensive visualization of data before and after cleaning:
   - S12 timeline before vs after unit fix
   - Target revenue distribution before vs after cleaning
   - Temperature timeline showing Fahrenheit correction
5. Verification of baseline reproducibility:
   - Raw data baseline (0.2387)
   - Dedup data baseline A (0.2391)
   - Clean fit baseline B (0.2393)
   - Clean validation baseline C (0.2286 / 0.2275) showing ~0.012 noise floor
6. Ablation study C0 to C5 across Ridge & LightGBM across all temporal folds:
   - Primary Val Q4/2025, Cleaned Val, Cold-Start
   - Fold A (Q1/2025 - Tet) & Tet window (-7..+7 days)
   - Fold B (Q2/2025) & Fold C (Q3/2025)
7. Final conclusion & adoption of Configuration C4.
"""

from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

project_root = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Title & Executive Summary
cells.append(nbf.v4.new_markdown_cell("""# Giai đoạn 0: Làm sạch có kiểm chứng (Verified Data Cleaning & Contract Testing)

> **Dự án:** Dự báo doanh thu cửa hàng cà phê (Cafe Revenue Regression)  
> **Tài liệu tham chiếu:** [`Regression_Plan_v2.md`](../Regression_Plan_v2.md) (Mục 5 & Mục 8 — Giai đoạn 0)  
> **Target:** `revenue` (Triệu VND)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Triết lý kiến trúc:** Kiến trúc ba lớp dữ liệu độc lập:
> ```text
> data/raw/ (Bất biến)  ──>  data/clean/ (Sửa lỗi + Cờ + Log)  ──>  features / models
> (train, stores,             train_clean.parquet (26.500 dòng)       imputer, encoder, scaler
>  weather, holidays)         stores_clean, weather_clean, holidays   fit strictly per fold
> ```

---

### Mục tiêu và Nội dung thực hiện trong Notebook:
1. **Kiểm toán chất lượng dữ liệu (Data Quality Audit):** Rà soát toàn bộ 4 tập dữ liệu thô đối chiếu với Sổ đăng ký vấn đề chất lượng (**DQ-01 đến DQ-15**).
2. **Thực thi Pipeline làm sạch chuẩn mực & Kiểm thử hợp đồng dữ liệu (Data Contracts):**
   - Loại bỏ chính xác **132 bản ghi trùng lặp hoàn toàn** (exact duplicates), đưa `train.csv` về **26.500 dòng sạch**. Khóa chính `(date, store_id)` duy nhất 100%.
   - Sửa lỗi đơn vị ghi chép cửa hàng **S12 trong tháng 03/2024** (30 dòng bị nhân 1.000, chia 1.000 đưa median về 29.17 triệu VND, không xóa dòng).
   - Phát hiện và gắn cờ **75 đột biến doanh thu bất thường** (**50 đột biến cao $\\approx \\times 10$** và **25 đột biến thấp $\\approx \\times 0.1$**).
   - Hiệu chỉnh **22 ngày nhiệt độ ghi bằng độ F** tại TP.HCM (84.0 – 89.2 °F) về độ C nhiệt đới (27.9 – 31.8 °C).
   - Dựng lưới thời tiết toàn diện ($821 \\text{ ngày} \\times 2 \\text{ thành phố} = 1.642 \\text{ dòng}$), điền thiếu an toàn có cờ theo dõi.
   - Chuẩn hóa loại cửa hàng, quận, parse ngày mở và impute diện tích `area_m2` cho **S19 và S24**.
   - Tổng hợp ngày lễ thành **53 ngày duy nhất** và tạo các cờ đặc trưng mùa **Tết Nguyên Đán**.
3. **Trực quan hóa Đối chiếu Trước và Sau làm sạch:** So sánh phân phối target, biểu đồ chuỗi thời gian của S12 và nhiệt độ TP.HCM.
4. **Xác nhận Baseline và Sàn nhiễu Validation:** Tái lập baseline Starter Notebook (**RMSLE = 0.2387** thô, **0.2391** dedup) và chứng minh **sàn nhiễu $\\approx 0.012$** do 6 điểm ngoại lệ trong tập validation Q4/2025.
5. **Nghiệm thu Thực nghiệm Ablation C0–C5:** Đo lường tác động từng can thiệp làm sạch trên mô hình Ridge và LightGBM qua 4 fold thời gian (đặc biệt Fold A mùa Tết).
6. **Chốt cấu hình tiêu chuẩn (C4):** Phê duyệt bộ dữ liệu sạch chính thức lưu tại `data/clean/`.
"""))

# Cell 2: Imports & Setup
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    TRAIN_RAW_FILE, STORES_RAW_FILE, WEATHER_RAW_FILE, HOLIDAYS_RAW_FILE,
    CLEAN_DATA_DIR, REPORTS_DIR, DATE_COL, STORE_COL, TARGET_COL, PROMO_COL
)
from src.data.clean import (
    clean_train, clean_stores, clean_weather, clean_holidays,
    join_cleaned_data, export_clean_datasets
)
from src.models.baseline import WeekdayMedianBaseline
from src.evaluation.metrics import rmsle, mae

# Plot formatting
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

print("Đã nạp toàn bộ cấu hình và thư viện thành công!")
"""))

# Cell 3: Markdown Section 1 - Profiling Raw Datasets
cells.append(nbf.v4.new_markdown_cell("""## 1. Kiểm toán Chất lượng Dữ liệu Thô (Data Profiling & Audit)

Khảo sát độc lập trên 4 tệp dữ liệu thô ban đầu:
- `train.csv`: 26.632 dòng lịch sử doanh thu từ 01/01/2024 đến 31/12/2025 (45 cửa hàng).
- `stores.csv`: 45 cửa hàng với các trường phân loại, diện tích, số ghế, bãi đỗ xe và ngày khai trương.
- `weather.csv`: 1.593 bản ghi thời tiết (nhiệt độ, lượng mưa) tại Hà Nội và TP.HCM.
- `holidays.csv`: 55 sự kiện ngày lễ và lễ hội văn hóa tại Việt Nam.
"""))

# Cell 4: Code Section 1
cells.append(nbf.v4.new_code_cell("""raw_train = pd.read_csv(TRAIN_RAW_FILE, parse_dates=[DATE_COL])
raw_stores = pd.read_csv(STORES_RAW_FILE)
raw_weather = pd.read_csv(WEATHER_RAW_FILE, parse_dates=[DATE_COL])
raw_holidays = pd.read_csv(HOLIDAYS_RAW_FILE, parse_dates=[DATE_COL])

print(f"1. raw_train:    {raw_train.shape[0]:,} dòng, {raw_train[STORE_COL].nunique()} cửa hàng, thời gian {raw_train[DATE_COL].min().date()} đến {raw_train[DATE_COL].max().date()}")
print(f"   Exact duplicates: {raw_train.duplicated().sum()} dòng")
print(f"2. raw_stores:   {raw_stores.shape[0]} cửa hàng, diện tích thiếu: {raw_stores['area_m2'].isna().sum()} cửa hàng")
print(f"3. raw_weather:  {raw_weather.shape[0]:,} dòng, nhiệt độ > 45°C: {(raw_weather['temp_c'] > 45).sum()} dòng (lỗi đơn vị Fahrenheit)")
print(f"4. raw_holidays: {raw_holidays.shape[0]} sự kiện, số ngày duy nhất: {raw_holidays[DATE_COL].nunique()} ngày")
"""))

# Cell 5: Markdown Section 2 - Detailed Outlier Audit S12 & Spikes
cells.append(nbf.v4.new_markdown_cell("""## 2. Kiểm toán Chi tiết Lỗi Đơn vị S12 và 75 Đột biến Doanh thu

### 2.1. Lỗi đơn vị S12 (Tháng 03/2024)
- Toàn bộ 30 dòng ghi nhận trong tháng 03/2024 của cửa hàng **S12** bị ghi theo **Nghìn VND** (dao động từ 10.620 đến 53.600 triệu VND).
- Trung bình trước khi sửa là **28.087,67 triệu VND** (khoảng 28 tỷ VND/ngày cho một quán cà phê!).
- Phép chia 1.000 đưa trung bình về **28,09 triệu VND** (median = **29,17 triệu VND**), hoàn toàn tương thích với tháng 02/2024 (22,3 triệu VND có Tết) và tháng 04/2024 (27,0 triệu VND).
"""))

# Cell 6: Code Section 2
cells.append(nbf.v4.new_code_cell("""s12_raw_m = (raw_train[STORE_COL] == "S12") & raw_train[DATE_COL].between("2024-03-01", "2024-03-31")
s12_raw_rows = raw_train.loc[s12_raw_m]

print(f"Số dòng S12 trong tháng 3/2024: {len(s12_raw_rows)} dòng")
print(f"Doanh thu nhỏ nhất: {s12_raw_rows['revenue'].min():,.1f} Tr. VND")
print(f"Doanh thu lớn nhất: {s12_raw_rows['revenue'].max():,.1f} Tr. VND")
print(f"Doanh thu trung vị: {s12_raw_rows['revenue'].median():,.1f} Tr. VND")
print(f"Sau khi chia 1.000 -> Trung vị: {s12_raw_rows['revenue'].median() / 1000:.2f} Tr. VND")
"""))

# Cell 7: Markdown Section 3 - Executing Cleaning Pipeline with Zero Leakage
cells.append(nbf.v4.new_markdown_cell("""## 3. Thực thi Pipeline Làm sạch Dữ liệu chuẩn mực (Leakage-Free Cleaning)

Áp dụng phân loại 3 nhóm can thiệp:
- **Loại A (Xác định):** Khử trùng lặp, chia 1.000 cho S12, đổi độ F sang độ C, chuẩn hóa chuỗi và parse ngày.
- **Loại B (Quy tắc thống kê):** Phát hiện đột biến theo tỷ lệ so với median tham chiếu `ref_end = '2025-10-01'` (đảm bảo không rò rỉ dữ liệu khi đánh giá Holdout).
- **Loại C (Hiện tượng thật):** Giữ nguyên doanh thu thấp do Tết và mùa mưa bão; không điền 0 cho 633 ngày vắng.
"""))

# Cell 8: Code Section 3
cells.append(nbf.v4.new_code_cell("""# 1. Clean train with ref_end='2025-10-01' to prevent CV lookahead leakage
train_clean, train_log = clean_train(raw_train, ref_end="2025-10-01")
print(f"Train cleaning log:\\n{json.dumps(train_log, indent=2)}")

# 2. Clean auxiliary tables
stores_clean, stores_log = clean_stores(raw_stores)
weather_clean, weather_log = clean_weather(raw_weather, ref_end="2025-10-01")
holidays_clean, holidays_log = clean_holidays(raw_holidays)

# 3. Safe multi-table join
df_clean = join_cleaned_data(train_clean, stores_clean, weather_clean, holidays_clean)
print(f"\\nKết quả sau khi join an toàn: {df_clean.shape[0]:,} dòng, {df_clean.shape[1]} cột (Bảo toàn 100% số dòng)")
"""))

# Cell 9: Markdown Section 4 - Visualizing Before vs After
cells.append(nbf.v4.new_markdown_cell("""## 4. Trực quan hóa Đối chứng: Trước và Sau khi Làm sạch"""))

# Cell 10: Code Section 4 - Visualizations
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: S12 Timeline before and after unit fix
s12_clean = train_clean[train_clean[STORE_COL] == "S12"].sort_values(DATE_COL)
s12_raw = raw_train[raw_train[STORE_COL] == "S12"].sort_values(DATE_COL)

axes[0, 0].plot(s12_clean[DATE_COL], s12_clean["revenue_raw"], color="salmon", alpha=0.7, label="Dữ liệu thô (Lỗi x1000 Tháng 3/2024)")
axes[0, 0].set_title("Cửa hàng S12: Doanh thu thô ban đầu (Max 53.600 Tr. VND)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Doanh thu thô (Tr. VND)")
axes[0, 0].legend()

axes[0, 1].plot(s12_clean[DATE_COL], s12_clean["revenue"], color="teal", label="Sau khi sửa lỗi chia 1000")
axes[0, 1].set_title("Cửa hàng S12: Doanh thu sau khi hiệu chuẩn (Chuẩn mực 20-35 Tr. VND)", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Doanh thu sạch (Tr. VND)")
axes[0, 1].legend()

# Plot 2: Weather Fahrenheit fix
hcm_raw = raw_weather[raw_weather["city"] == "Ho Chi Minh"].sort_values(DATE_COL)
hcm_clean = weather_clean[weather_clean["city"] == "Ho Chi Minh"].sort_values(DATE_COL)

axes[1, 0].plot(hcm_raw[DATE_COL], hcm_raw["temp_c"], color="firebrick", label="Nhiệt độ thô (22 ngày độ F: 84-89°F)")
axes[1, 0].set_title("Thời tiết TP.HCM: Lỗi đơn vị Fahrenheit Tháng 06/2025", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Nhiệt độ (°C)")
axes[1, 0].legend()

axes[1, 1].plot(hcm_clean[DATE_COL], hcm_clean["temp_c"], color="steelblue", label="Sau khi đổi (°F - 32) * 5/9 sang °C")
axes[1, 1].set_title("Thời tiết TP.HCM: Sau khi chuẩn hóa (24 - 34 °C)", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Nhiệt độ (°C)")
axes[1, 1].legend()

plt.tight_layout()
plt.show()
"""))

# Cell 11: Markdown Section 5 - Outliers Flagged Inspection
cells.append(nbf.v4.new_markdown_cell("""## 5. Rà soát Sổ đăng ký Ngoại lệ Đã Gắn cờ (`outliers_flagged.csv`)

Tổng hợp 110 dòng can thiệp:
- **30 dòng `unit_x1000`:** S12 Tháng 3/2024.
- **50 dòng `spike_x10`:** Đột biến cao ngẫu nhiên (47 dòng thuộc train, 3 dòng thuộc Holdout Q4/2025).
- **25 dòng `spike_low_x01`:** Đột biến thấp ngẫu nhiên (22 dòng thuộc train, 3 dòng thuộc Holdout Q4/2025).
- **5 dòng `review_flag`:** Tỷ lệ 3.0 – 5.0 lần median.
"""))

# Cell 12: Code Section 5
cells.append(nbf.v4.new_code_cell("""outliers_df = pd.read_csv(REPORTS_DIR / "outliers_flagged.csv")
print(f"Tổng số bản ghi trong outliers_flagged.csv: {len(outliers_df)} dòng")
print("\\nPhân loại theo fix_flag:")
print(outliers_df["fix_flag"].value_counts(dropna=False))
print(f"\\nSố dòng cần xem thủ công (review_flag == 1): {(outliers_df['review_flag'] == 1).sum()} dòng")

print("\\nTrích xuất mẫu 5 dòng đột biến cao (spike_x10):")
print(outliers_df[outliers_df["fix_flag"] == "spike_x10"].head(5)[["date", "store_id", "revenue_raw", "revenue", "ratio_to_ref", "is_promo"]])
"""))

# Cell 13: Markdown Section 6 - Baseline Reproducibility and Noise Floor
cells.append(nbf.v4.new_markdown_cell("""## 6. Tái lập Baseline & Khám phá Sàn Nhiễu Validation Q4/2025

Theo `Regression_Plan_v2.md` Mục 3.3, đánh giá mô hình Baseline Median `(store_id, dayofweek)` trên các trạng thái dữ liệu:
- **Mốc Starter Notebook (Dữ liệu thô):** **RMSLE = 0.2387**
- **Trạng thái A (Dữ liệu đã khử trùng lặp):** **RMSLE = 0.2391** (Mốc so sánh chính thống)
- **Trạng thái B (Làm sạch Train):** **RMSLE = 0.2393** (Baseline median bền vững nên ít đổi)
- **Trạng thái C (Làm sạch cả Train và Validation):** **RMSLE = 0.2286** (~**0.2275**)

Sự chênh lệch **0.0105 – 0.0116 RMSLE** giữa Trạng thái A và Trạng thái C bắt nguồn từ đúng **6 dòng đột biến ngẫu nhiên trong quý 4/2025** (3 cao + 3 thấp), tạo ra một **sàn nhiễu không thể dự báo**.
"""))

# Cell 14: Code Section 6
cells.append(nbf.v4.new_code_cell("""# 1. Mốc Notebook thô
fit_raw = raw_train[raw_train[DATE_COL] < "2025-10-01"]
val_raw = raw_train[raw_train[DATE_COL] >= "2025-10-01"]
b_raw = WeekdayMedianBaseline()
b_raw.fit(fit_raw)
p_raw = b_raw.predict(val_raw)
score_raw = rmsle(val_raw[TARGET_COL], p_raw)

# 2. Mốc A (Dữ liệu sạch khử trùng lặp)
fit_a = train_clean[train_clean[DATE_COL] < "2025-10-01"]
val_a = train_clean[train_clean[DATE_COL] >= "2025-10-01"]
b_a = WeekdayMedianBaseline()
b_a.fit(fit_a)
p_a = b_a.predict(val_a)
score_a = rmsle(val_a[TARGET_COL], p_a)

# 3. Mốc B (Train sạch S12 & loại đột biến, Val gốc)
fit_b = train_clean[(train_clean[DATE_COL] < "2025-10-01") & (~train_clean["fix_flag"].str.startswith("spike"))]
b_b = WeekdayMedianBaseline()
b_b.fit(fit_b)
p_b = b_b.predict(val_a)
score_b = rmsle(val_a[TARGET_COL], p_b)

# 4. Mốc C (Làm sạch cả Train và Val)
val_c = val_a[~val_a["fix_flag"].str.startswith("spike")]
p_c = b_b.predict(val_c)
score_c = rmsle(val_c[TARGET_COL], p_c)

print(f"Mốc Starter Notebook (Thô):   RMSLE = {score_raw:.4f}")
print(f"Trạng thái A (Đã Dedup):      RMSLE = {score_a:.4f}  <-- MỐC CHUẨN SO SÁNH")
print(f"Trạng thái B (Train sạch):     RMSLE = {score_b:.4f}")
print(f"Trạng thái C (Val sạch):      RMSLE = {score_c:.4f}  (Chênh lệch sàn nhiễu: {score_a - score_c:+.4f})")
"""))

# Cell 15: Markdown Section 7 - Ablation C0-C5
cells.append(nbf.v4.new_markdown_cell("""## 7. Nghiệm thu Thực nghiệm Ablation C0–C5

Thực nghiệm đo lường trên cả 2 họ mô hình (Ridge và LightGBM) qua 4 fold thời gian:
- **C0:** Chỉ khử trùng lặp (chưa sửa S12, giữ đột biến trong fit)
- **C1:** C0 + Sửa lỗi đơn vị S12 (/1000)
- **C2:** C1 + Loại 75 đột biến khỏi fit
- **C3:** C1 + Sửa đột biến (cao /10, thấp *10) thay vì loại
- **C4:** C2 + Làm sạch toàn diện thời tiết và stores
- **C5:** C1 + Huber loss (giữ đột biến, dùng hàm tổn thất bền vững)
"""))

# Cell 16: Code Section 7
cells.append(nbf.v4.new_code_cell("""ablation_df = pd.read_csv(REPORTS_DIR / "data_cleaning_ablation.csv")
print("Bảng Kết quả Nghiệm thu Ablation C0–C5:")
print(ablation_df[["config_id", "model", "primary_val_rmsle", "clean_val_rmsle", "fold_a_rmsle", "tet_window_rmsle", "cold_start_rmsle"]].to_string(index=False))
"""))

# Cell 17: Markdown Section 8 - Plotting Ablation Comparisons
cells.append(nbf.v4.new_markdown_cell("""## 8. Trực quan hóa So sánh Hiệu năng Ablation

Phân tích biểu đồ minh chứng:
1. **Lỗi S12 làm nổ tung mô hình Boosting:** Tại Fold A (Q1/2025), LightGBM ở C0 đạt RMSLE **0.7164**. Khi sửa S12 ở C1, RMSLE lập tức hạ sâu xuống **0.3340** (giảm 0.3824 RMSLE!).
2. **Sàn nhiễu 0.012 do đột biến validation:** Tại Q4/2025, RMSLE trên tập sạch luôn thấp hơn tập gốc khoảng 0.010 – 0.012.
"""))

# Cell 18: Code Section 8
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ridge_sub = ablation_df[ablation_df["model"] == "Ridge"]
lgb_sub = ablation_df[ablation_df["model"] == "LightGBM"]

# Plot 1: Primary Val vs Clean Val
x_labels = [f"{row.config_id}" for _, row in ridge_sub.iterrows()]
x = np.arange(len(x_labels))
width = 0.35

axes[0].bar(x - width/2, ridge_sub["primary_val_rmsle"], width, label="Ridge (Val Gốc)", color="#4575b4", alpha=0.9)
axes[0].bar(x + width/2, ridge_sub["clean_val_rmsle"], width, label="Ridge (Val Lọc Nhiễu)", color="#74add1", alpha=0.9)
axes[0].axhline(0.2391, color="red", linestyle="--", label="Baseline A (0.2391)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(x_labels)
axes[0].set_title("Ridge: Primary Val Gốc vs Val Đã Lọc Nhiễu", fontsize=12, fontweight="bold")
axes[0].set_ylabel("RMSLE")
axes[0].set_ylim(0.18, 0.26)
axes[0].legend()

# Plot 2: Fold A (Q1/2025 - Tet) comparison
axes[1].bar(x - width/2, lgb_sub["fold_a_rmsle"], width, label="LightGBM (Fold A)", color="#d73027", alpha=0.9)
axes[1].bar(x + width/2, ridge_sub["fold_a_rmsle"], width, label="Ridge (Fold A)", color="#fdae61", alpha=0.9)
axes[1].set_xticks(x)
axes[1].set_xticklabels(x_labels)
axes[1].set_title("Fold A (Q1/2025 - Tết): Minh chứng lỗi S12 làm nổ tung C0 ở LightGBM", fontsize=12, fontweight="bold")
axes[1].set_ylabel("RMSLE")
axes[1].set_ylim(0.20, 0.75)
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

# Cell 19: Markdown Section 9 - Final Conclusion and Exported Clean Assets
cells.append(nbf.v4.new_markdown_cell("""## 9. Kết luận Nghiệm thu & Bộ Dữ liệu Sạch Xuất bản

### Quyết định kỹ thuật:
1. **Phê duyệt Cấu hình C4 làm tiêu chuẩn cho toàn dự án:**
   - Đã xử lý triệt để 132 trùng lặp.
   - Đã hiệu chỉnh 30 dòng đơn vị S12.
   - Đã gắn cờ và loại bỏ 75 đột biến khỏi tập fit.
   - Đã chuẩn hóa 100% thời tiết (°F -> °C, lưới 1.642 dòng) và thông tin cửa hàng.
2. **Bộ dữ liệu sạch đã xuất bản tại `data/clean/`:**
   - `train_clean.parquet` (26.500 dòng)
   - `stores_clean.parquet` (45 dòng)
   - `weather_clean.parquet` (1.642 dòng)
   - `holidays_daily.parquet` (53 dòng)
   - `reports/outliers_flagged.csv` (110 dòng ngoại lệ)
3. **Sẵn sàng bước sang các giai đoạn tiếp theo** (Mô hình tuyến tính, Tree-based models, Target transformation, và Ensembles).
"""))

# Cell 20: Code Section 9
cells.append(nbf.v4.new_code_cell("""clean_files = list(CLEAN_DATA_DIR.glob("*.*"))
print(f"Danh sách các file trong {CLEAN_DATA_DIR}:")
for f in clean_files:
    print(f" - {f.name:25s} ({f.stat().st_size / 1024:.1f} KB)")
"""))

# Attach cells to notebook
nb.cells = cells

# Save and execute
output_path = project_root / "notebooks" / "00_data_cleaning_and_verification.ipynb"

print(f"Executing and saving notebook to {output_path}...")
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(project_root / "notebooks")}})
client.execute()

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook created and executed successfully: {output_path}")
