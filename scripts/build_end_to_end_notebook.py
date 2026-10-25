"""Script to build and execute notebooks/end_to_end_training_pipeline.ipynb.

This notebook implements the complete end-to-end Machine Learning training
pipeline for the Cafe Revenue Regression project, progressing from raw data
and problem definition all the way to the champion ensemble model
(RMSLE = 0.1603 on Primary Validation Q4/2025, +32.98% improvement vs Baseline).
"""

from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

project_root = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# ==============================================================================
# Cell 1: Big Title & Executive Summary
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Quy Trình Huấn Luyện Machine Learning Toàn Diện (End-to-End ML Pipeline)

> **Dự án:** Dự Báo Doanh Thu Chuỗi Cửa Hàng Cà Phê (*Cafe Revenue Regression*)  
> **Biến mục tiêu:** `revenue` (Triệu VND/ngày, không âm)  
> **Thước đo đánh giá chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Kế hoạch triển khai:** Chuẩn hóa 100% theo [`Regression_Plan.md`](../Regression_Plan.md)  
> **Phân vai dữ liệu:**
> - **Tập Train:** `date < 2025-10-01` — **22.668 dòng sạch** (sau khi khử 132 bản ghi trùng lặp), 21 tháng lịch sử (01/01/2024 – 30/09/2025).
> - **Tập Primary Validation:** `2025-10-01 <= date < 2026-01-01` (Quý 4/2025) — **3.832 dòng sạch**, đồng bộ tuyệt đối với [`starter_cafe.ipynb`](../data/raw/starter_cafe.ipynb).
> - **Tập Test chính thức:** Quý 1/2026 (`01/01/2026 <= date <= 31/03/2026`) do ban tổ chức công bố trên Kaggle.
>
> **Tóm tắt thành tích lũy tiến qua 5 thế hệ mô hình trên Primary Validation (Q4/2025):**
> - Mốc khởi điểm Baseline: **RMSLE = 0.2391** *(0.2387 trên dữ liệu thô)* | MAE = **3.57 triệu VND**
> - Mô hình Tuyến tính Ridge (Giai đoạn 2): **RMSLE = 0.1978** | MAE = **2.94 triệu VND** (+17.27% so với Baseline)
> - Mô hình Cây LightGBM Tuned (Giai đoạn 3): **RMSLE = 0.1667** | MAE = **2.49 triệu VND** (+30.28% so với Baseline)
> - Tối ưu Target Transform Huber Loss (Giai đoạn 4): **RMSLE = 0.1664** | MAE = **2.46 triệu VND** (+30.43% so với Baseline)
> - **Champion Ensemble + Duan Smearing (Mô hình Tối ưu Nhất Toàn Dự án):** **RMSLE = 0.1603** | MAE = **2.33 triệu VND** (**+32.98% cải thiện toàn diện so với Baseline**).

---

## Mục Lục Quy Trình Huấn Luyện 10 Bước Chuẩn Mực

1. **[Bước 1 — Problem Definition](#1-bước-1--problem-definition-xác-định-bài-toán-kinh-doanh--mục-tiêu-kỹ-thuật):** Xác định bài toán hồi quy chuỗi thời gian, ràng buộc horizon 90 ngày, và bản chất hàm mất mát RMSLE.
2. **[Bước 2 — Data Understanding & EDA](#2-bước-2--data-understanding-và-eda-khám-phá-dữ-liệu):** Khám phá cấu trúc 4 nguồn dữ liệu thô, phân tích phân phối lệch phải (+11.71) và trực quan hóa các mối tương quan kinh doanh.
3. **[Bước 3 — Data Cleaning](#3-bước-3--data-cleaning-làm-sạch-dữ-liệu--kiểm-soát-rò-rỉ):** Khử trùng lặp 132 dòng trong `train.csv`, hiệu chỉnh ngoại lệ đơn vị cửa hàng S12, và thiết lập nguyên tắc chống Data Leakage.
4. **[Bước 4 — Chia Dataset](#4-bước-4--chia-dataset-train-và-validation-theo-thời-gian):** Thiết lập phân chia thời gian (Train < 2025-10-01, Val Q4/2025) và cấu trúc Walk-Forward Cross-Validation 4 Fold.
5. **[Bước 5 — Preprocessing & Feature Engineering](#5-bước-5--preprocessing-và-feature-engineering-đa-tầng):** Xây dựng các tầng đặc trưng E1 (Lịch + Store), E2 (Khuyến mãi + Tương tác), E3 (Thời tiết + Ngày lễ), và Out-of-fold Historical Target Statistics an toàn tuyệt đối.
6. **[Bước 6 — Model Selection & Progressive Training](#6-bước-6--model-selection-và-progressive-training-hành-trình-5-thế-hệ-mô-hình):** Huấn luyện lũy tiến qua 5 thế hệ mô hình từ Baseline Median -> Linear Ridge -> LightGBM Tuned -> LightGBM Huber Loss -> Champion Ensemble (91% LGBM + 9% Ridge + Duan Smearing).
7. **[Bước 7 — Evaluation & Hyperparameter Tuning](#7-bước-7--evaluation-và-hyperparameter-tuning-đánh-giá-toàn-diện):** So sánh bảng tổng hợp metrics, kiểm định Walk-Forward 4 chu kỳ, phân tích sai số theo phân khúc (Subgroups), và chẩn đoán Residuals.
8. **[Bước 8 — Final Retraining trên Toàn Bộ Dữ Liệu](#8-bước-8--final-retraining-tái-huấn-luyện-toàn-bộ-dữ-liệu-lịch-sử-chuẩn-bị-cho-kaggle-test):** Tái huấn luyện mô hình Champion trên toàn bộ 26.500 dòng sạch (2024–2025) để tối đa hóa hiệu năng cho tập Test Q1/2026.
9. **[Bước 9 — Lưu Model, Triển Khai & Xuất File Submission](#9-bước-9--lưu-model-triển-khai--xuất-file-submission-artifact--inference-pipeline):** Đóng gói artifact `.joblib`, xây dựng hàm suy luận sản xuất `predict_cafe_revenue()`, kiểm thử suy luận và xuất tệp nộp bài Kaggle (`submission.csv`).
10. **[Bước 10 — Monitoring & Retraining](#10-bước-10--monitoring-và-retraining-vận-hành-mlops-thực-tế):** Thiết lập chiến lược giám sát Data Drift / Concept Drift, ngưỡng cảnh báo suy giảm hiệu năng, và chu kỳ retrain định kỳ.
11. **[Checklist Nghiệm Thu & Nguyên Tắc Cốt Lõi](#11-checklist-nghiệm-thu-kỹ-thuật-và-các-nguyên-tắc-cốt-lõi):** Tổng kết 13 tiêu chí chất lượng và khẳng định độ tin cậy của sản phẩm.
"""))

# ==============================================================================
# Cell 2: Imports & Environment Setup
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 1. Cấu hình môi trường & Thư viện phụ thuộc
import os
import sys
import warnings
from pathlib import Path

# Tự động nhận diện Project Root cho dù chạy từ thư mục gốc hay thư mục notebooks/
CURRENT_DIR = Path.cwd()
if (CURRENT_DIR / "src").exists():
    PROJECT_ROOT = CURRENT_DIR
elif (CURRENT_DIR.parent / "src").exists():
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Thiết lập phong cách trực quan hóa
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

# Nạp các module kiến trúc từ src/
from src.config import (
    DATE_COL, STORE_COL, TARGET_COL, PROMO_COL,
    PRIMARY_TRAIN_END, PRIMARY_VAL_END,
    WALK_FORWARD_FOLDS, MODELS_DIR, REPORTS_DIR, SPLITS_DIR,
    TRAIN_RAW_FILE, STORES_RAW_FILE, WEATHER_RAW_FILE, HOLIDAYS_RAW_FILE
)
from src.data.load import (
    load_raw_train, load_raw_stores, load_raw_weather, load_raw_holidays,
    deduplicate_train, load_splits
)
from src.data.join import join_all_data
from src.features.builder import build_all_features
from src.features.tree_builder import (
    correct_revenue_outliers,
    prepare_tree_feature_matrices,
    compute_historical_target_stats,
    apply_historical_target_stats
)
from src.models.baseline import WeekdayMedianBaseline
from src.models.linear import LogTransformedLinearModel
from src.models.target_transform import TransformedTargetTreeModel
from src.models.ensemble import WeightedLogEnsemble
from src.evaluation.metrics import compute_metrics, rmsle, mae, rmse, log_residual
from src.evaluation.backtest import evaluate_subgroups

print(f"Môi trường sẵn sàng. Project Root: {PROJECT_ROOT}")
"""))

# ==============================================================================
# Cell 3: Bước 1 — Problem Definition
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Bước 1 — Problem Definition: Xác định bài toán kinh doanh & Mục tiêu kỹ thuật

Trước khi tiến hành tiền xử lý và xây dựng mô hình, bài toán cần được định nghĩa tường minh về mục tiêu nghiệp vụ, đặc điểm biến mục tiêu, hàm mất mát và phạm vi dự báo:

### 1.1. Bối cảnh nghiệp vụ & Mục tiêu
- **Bài toán:** Dự báo doanh thu hàng ngày theo từng cửa hàng trong chuỗi quán cà phê (*panel time-series regression*).
- **Biến mục tiêu ($y$):** `revenue` (Doanh thu thực tế trong ngày, tính theo triệu VND, điều kiện bắt buộc: $y \\ge 0$).
- **Loại hình bài toán ML:** *Supervised Learning — Panel Time-Series Tabular Regression* (Hồi quy dữ liệu bảng đa thực thể chuỗi thời gian).
- **Phạm vi & Horizon dự báo:** Dự báo định kỳ 1 quý tương lai (90 ngày liên tục, từ 01/01/2026 đến 31/03/2026) phục vụ lập kế hoạch mua sắm nguyên vật liệu (hạt cà phê, sữa, đường), điều phối nhân sự ca kíp và chỉ tiêu doanh số.
- **Ràng buộc thông tin (Zero Data Leakage Contract):** Tại thời điểm cắt dữ liệu $T_0$, mô hình chỉ được phép biết các dữ liệu lịch sử đã diễn ra trước $T_0$, cùng các thông tin đã biết trước của tương lai như lịch vạn niên, ngày lễ quốc gia, thời tiết dự báo và chiến dịch khuyến mãi được lên lịch trước.

### 1.2. Thước đo đánh giá kỹ thuật (Evaluation Metric): RMSLE
Bài toán sử dụng thước đo **RMSLE (Root Mean Squared Logarithmic Error)**:

$$\\text{RMSLE} = \\sqrt{ \\frac{1}{N} \\sum_{i=1}^N \\left( \\log(y_i + 1) - \\log(\\hat{y}_i + 1) \\right)^2 }$$

**Đặc điểm của RMSLE trong bài toán F&B:**
1. **Đo lường sai số tương đối (Relative Error):** Sai lệch 2 triệu VND tại quán nhỏ có doanh thu 10 triệu/ngày (lệch 20%) có mức độ nghiêm trọng tương đương với sai lệch 10 triệu VND tại quán lớn có doanh thu 50 triệu/ngày (lệch 20%).
2. **Phạt nặng dự báo thiếu (Penalizes Underprediction):** Trong F&B, dự báo thiếu dẫn đến thiếu hụt nguyên liệu, đứt gãy phục vụ và mất khách hàng trung thành, thiệt hại lớn hơn việc chuẩn bị dư một ít nguyên liệu dự trữ.
3. **Bất biến với tỷ lệ quy mô cửa hàng:** Giúp mô hình không bị chi phối thiên vị hoàn toàn bởi các cửa hàng Flagship doanh thu cực lớn mà bỏ quên các cửa hàng vệ tinh nhỏ.
"""))

# ==============================================================================
# Cell 4: Bước 2.1 — Đọc & Khám phá Dữ liệu Thô (Data Understanding)
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 2. Bước 2 — Data Understanding: Đọc và kiểm tra cấu trúc 4 nguồn dữ liệu thô
raw_train = load_raw_train()
stores_raw = load_raw_stores()
weather_raw = load_raw_weather()
holidays_raw = load_raw_holidays()

print("=" * 70)
print("THỐNG KÊ KÍCH THƯỚC CÁC BẢNG DỮ LIỆU:")
print(f" - train.csv:    {raw_train.shape[0]:,} dòng x {raw_train.shape[1]} cột (Khoảng ngày: {raw_train['date'].min()} -> {raw_train['date'].max()})")
print(f" - stores.csv:   {stores_raw.shape[0]:,} dòng x {stores_raw.shape[1]} cột (Gồm {len(stores_raw)} cửa hàng)")
print(f" - weather.csv:  {weather_raw.shape[0]:,} dòng x {weather_raw.shape[1]} cột (Thông tin thời tiết theo thành phố)")
print(f" - holidays.csv: {holidays_raw.shape[0]:,} dòng x {holidays_raw.shape[1]} cột (Lịch nghỉ lễ, Tết)")
print("=" * 70)

# Kiểm tra Missing values
print("\\nKIỂM TRA GIÁ TRỊ THIẾU (MISSING VALUES):")
for name, df in [("train", raw_train), ("stores", stores_raw), ("weather", weather_raw), ("holidays", holidays_raw)]:
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        print(f" - Bảng {name:10s}: Không có giá trị thiếu (0 nulls)")
    else:
        print(f" - Bảng {name:10s}: Có cột thiếu -> {null_cols.to_dict()}")

# Xem 5 dòng đầu train.csv
print("\\n5 DÒNG ĐẦU CỦA BẢNG train.csv:")
display(raw_train.head())
"""))

# ==============================================================================
# Cell 5: Bước 2.2 — Phân tích EDA & Độ Lệch Target
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Bước 2 (tiếp theo) — Phân tích Khám phá Dữ liệu (EDA) & Phân phối Biến mục tiêu

Phân tích biến mục tiêu `revenue` là bước quan trọng định hình thiết kế mô hình:
- **Độ lệch của doanh thu gốc (`revenue`):** Doanh thu thực tế có độ lệch phải rất nặng (**Skewness = +11.71**, **Kurtosis = 257.34**). Hiện tượng này xuất hiện do các ngày lễ Tết, các đợt khuyến mãi bùng nổ, hoặc các ngày cuối tuần khách tăng đột biến tại các cửa hàng trung tâm thương mại.
- **Biến đổi log-space $\\log(1 + y)$:** Khi áp dụng phép biến đổi $\\log(1 + \\text{revenue})$, độ lệch giảm ngoạn mục xuống **Skewness = -0.27** và **Kurtosis = 1.27**, đồ thị phân phối trở nên cân xứng gần như phân phối chuẩn (Gaussian-like).
- **Mối liên hệ giữa tối ưu hóa và metric:** Tối ưu hóa hàm mất mát bình phương sai số (MSE/Huber) trên không gian $\\log(1 + y)$ chính là tối ưu trực tiếp đại lượng bên trong căn bậc hai của metric **RMSLE**.
"""))

# ==============================================================================
# Cell 6: Bước 2.3 — Trực quan hóa EDA
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# Trực quan hóa Khám phá Dữ liệu (EDA)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Phân phối Target gốc vs Log1p
ax1 = axes[0, 0]
rev_clean = raw_train["revenue"].dropna()
skew_raw = rev_clean.skew()
skew_log = np.log1p(rev_clean).skew()

sns.kdeplot(rev_clean, ax=ax1, color="#e74c3c", fill=True, alpha=0.4, label=f"Raw Revenue (Skew: {skew_raw:.2f})")
ax1.set_title("Phân phối Doanh thu Gốc (Lệch phải nặng)", fontweight="bold")
ax1.set_xlabel("Doanh thu (Triệu VND)")
ax1.set_xlim(0, 80)
ax1.legend(loc="upper right")

# Biểu đồ lồng trục phụ cho Log1p
ax1_twin = ax1.twiny()
sns.kdeplot(np.log1p(rev_clean), ax=ax1_twin, color="#2980b9", fill=True, alpha=0.3, label=f"Log1p Revenue (Skew: {skew_log:.2f})")
ax1_twin.set_xlabel("log1p(Revenue) [Trục xanh]")
ax1_twin.legend(loc="upper left")

# 2. Doanh thu theo Ngày trong tuần & Khuyến mãi
ax2 = axes[0, 1]
raw_train_eda = raw_train.copy()
raw_train_eda["date"] = pd.to_datetime(raw_train_eda["date"])
raw_train_eda["dayofweek"] = raw_train_eda["date"].dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

sns.boxplot(
    data=raw_train_eda, x="dayofweek", y="revenue", hue="is_promo",
    order=dow_order, palette={0: "#95a5a6", 1: "#e67e22"},
    showfliers=False, ax=ax2
)
ax2.set_title("Doanh thu theo Thứ trong tuần & Tác động của Khuyến mãi", fontweight="bold")
ax2.set_xlabel("Thứ trong tuần")
ax2.set_ylabel("Doanh thu (Triệu VND)")
ax2.legend(title="Khuyến mãi (is_promo)", loc="upper left")
ax2.tick_params(axis="x", rotation=25)

# 3. Doanh thu theo Loại hình Cửa hàng (Store Type)
ax3 = axes[1, 0]
store_merged = raw_train_eda.merge(stores_raw, on="store_id", how="left")

sns.barplot(data=store_merged, x="store_type", y="revenue", estimator=np.median, errorbar=None, palette="Blues_d", ax=ax3)
ax3.set_title("Doanh thu Trung vị theo Loại Cửa Hàng (Store Type)", fontweight="bold")
ax3.set_xlabel("Loại hình Cửa hàng")
ax3.set_ylabel("Doanh thu Trung vị (Triệu VND)")
for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.1f}M", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

# 4. Chuỗi thời gian Doanh thu trung bình toàn hệ thống
ax4 = axes[1, 1]
daily_sys_rev = raw_train_eda.groupby("date")["revenue"].mean()
daily_sys_rev.rolling(7).mean().plot(ax=ax4, color="#27ae60", lw=2, label="Trung bình trượt 7 ngày")
ax4.set_title("Xu hướng Doanh thu Hàng ngày Toàn Chuỗi (2024 - 2025)", fontweight="bold")
ax4.set_xlabel("Thời gian")
ax4.set_ylabel("Doanh thu Trung bình (Triệu VND)")
ax4.legend(loc="upper left")

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 7: Bước 3 — Data Cleaning
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Bước 3 — Data Cleaning: Làm sạch dữ liệu & Kiểm soát Rò rỉ (Leakage)

Dựa trên các phát hiện từ bước EDA, quy trình làm sạch dữ liệu tập trung xử lý các sai sót cấu trúc và ngoại lệ nghiệp vụ mà không làm méo mó tính đại diện của tập dữ liệu:

1. **Khử trùng lặp bản ghi (Deduplication):**
   - Bộ dữ liệu `train.csv` ban đầu có 26.632 dòng.
   - Phát hiện **132 bản ghi trùng lặp hoàn toàn** trên tất cả các cột.
   - Sau khi khử trùng lặp, dữ liệu còn đúng **26.500 dòng**, đảm bảo hợp đồng dữ liệu: cặp khóa `(date, store_id)` là duy nhất 100%.
2. **Hiệu chỉnh Ngoại lệ Tỷ lệ Đơn vị Cửa hàng S12 (Unit Scale Correction):**
   - Cửa hàng S12 mới khai trương từ tháng 04/2025. Trong những ngày đầu vận hành, nhân viên thu ngân đã nhập nhầm đơn vị doanh thu (đồng thay vì nghìn đồng), khiến doanh thu ghi nhận tăng vọt gấp 10-100 lần so với diện tích thực tế.
   - Hàm `correct_revenue_outliers()` xác định và hiệu chỉnh đúng 31 điểm ngoại lai sai đơn vị về tỷ lệ chuẩn.
3. **Nguyên tắc chống Data Leakage (Không để rò rỉ dữ liệu tương lai):**
   - Không fit imputer, scaler, target encoder trên toàn bộ dataset trước khi chia tách.
   - Mọi giá trị thống kê trung bình, trung vị lịch sử (`log_hist_sw_med`, `log_hist_s_med`) chỉ được tính toán thuần túy trên tập dữ liệu quá khứ (Train Set), sau đó ánh xạ (map) sang tập Validation.
"""))

# ==============================================================================
# Cell 8: Bước 3.1 — Thực thi Làm sạch Dữ liệu
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 3. Thực thi Làm sạch Dữ liệu: Deduplication & Outlier Correction
clean_train, dup_count = deduplicate_train(raw_train)
clean_train, n_corrected = correct_revenue_outliers(clean_train)

# Kiểm tra lại tính duy nhất của khóa (date, store_id)
duplicate_keys = clean_train.duplicated(subset=[DATE_COL, STORE_COL]).sum()

print("=" * 70)
print("KẾT QUẢ QUY TRÌNH DATA CLEANING:")
print(f" - Số dòng ban đầu:                  {len(raw_train):,} dòng")
print(f" - Bản ghi trùng lặp đã loại bỏ:     {dup_count:,} dòng")
print(f" - Bản ghi ngoại lệ S12 đã sửa:      {n_corrected:,} dòng")
print(f" - Số dòng sạch còn lại:             {len(clean_train):,} dòng")
print(f" - Trùng lặp khóa (date, store_id):  {duplicate_keys} (Đảm bảo tính duy nhất tuyệt đối)")
print("=" * 70)
"""))

# ==============================================================================
# Cell 9: Bước 4 — Chia Dataset
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Bước 4 — Chia dataset: Train và Validation theo Thời Gian

Để đồng bộ 100% với Notebook cơ sở của ban tổ chức ([`starter_cafe.ipynb`](../data/raw/starter_cafe.ipynb)) và [Regression_Plan.md](../Regression_Plan.md) (Mục 2.3), tập dữ liệu `train.csv` (26.500 dòng sạch) được chia theo đúng thời gian:

### Chiến lược Phân chia Chuỗi Thời Gian Chuẩn Mực:
1. **Tập Train (`date < 2025-10-01`):**
   - Khoảng thời gian: 21 tháng lịch sử từ **01/01/2024 đến 30/09/2025**.
   - Kích thước: **22.668 dòng sạch** (85.54% tổng dữ liệu).
   - Nhiệm vụ: Huấn luyện mô hình, học hệ số hồi quy và tính toán các bảng lookup thống kê lịch sử.
2. **Tập Primary Validation (`2025-10-01 <= date < 2026-01-01` — Quý 4/2025):**
   - Khoảng thời gian: Đúng 92 ngày của **Quý 4/2025**.
   - Kích thước: **3.832 dòng sạch** (14.46% tổng dữ liệu).
   - Nhiệm vụ: Mốc đánh giá chuẩn mực (benchmark), so sánh giữa các thế hệ mô hình, lựa chọn siêu tham số và tối ưu trọng số kết hợp (Ensemble).
3. **Tập Test chính thức (`01/01/2026 <= date <= 31/03/2026` — Quý 1/2026):**
   - Do giảng viên công bố trên Kaggle (dự án không tự cắt tệp test nội bộ từ `train.csv`).

Ngoài ra, thiết kế kiểm định bổ sung **Walk-Forward Cross-Validation qua 4 Fold thời gian**:
- **Fold A (Q1/2025):** Train đến `2025-01-01`, Validate trên Q1/2025 (đo mùa vụ tương đồng Q1/2026).
- **Fold B (Q2/2025):** Train đến `2025-04-01`, Validate trên Q2/2025.
- **Fold C (Q3/2025):** Train đến `2025-07-01`, Validate trên Q3/2025.
- **Primary Val (Q4/2025):** Train đến `2025-10-01`, Validate trên Q4/2025.
"""))

# ==============================================================================
# Cell 10: Bước 4.1 — Tách Dữ liệu & Thiết lập Temporal Splits
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 4. Ghép nối dữ liệu đa nguồn an toàn và phân chia Temporal Splits
# Bước 4.1: Ghép nối an toàn đa nguồn (Stores, Weather, Holidays)
joined_df = join_all_data(clean_train, stores_raw, weather_raw, holidays_raw)

# Bước 4.2: Phân chia tập dữ liệu theo mốc thời gian chặt chẽ
raw_train_split = joined_df[joined_df[DATE_COL] < PRIMARY_TRAIN_END].copy()
raw_val_split   = joined_df[joined_df[DATE_COL] >= PRIMARY_TRAIN_END].copy()

total_rows = len(joined_df)
splits_summary = pd.DataFrame([
    {
        "Tập Dữ Liệu": "Train Set (Lịch sử 21 tháng)",
        "Từ Ngày": raw_train_split[DATE_COL].min().strftime("%Y-%m-%d"),
        "Đến Ngày": raw_train_split[DATE_COL].max().strftime("%Y-%m-%d"),
        "Số Dòng": f"{len(raw_train_split):,}",
        "Tỷ Lệ %": f"{len(raw_train_split)/total_rows*100:.2f}%",
        "Mục Đích": "Học tham số & tính toán bảng thống kê OOF"
    },
    {
        "Tập Dữ Liệu": "Primary Validation (Q4/2025)",
        "Từ Ngày": raw_val_split[DATE_COL].min().strftime("%Y-%m-%d"),
        "Đến Ngày": raw_val_split[DATE_COL].max().strftime("%Y-%m-%d"),
        "Số Dòng": f"{len(raw_val_split):,}",
        "Tỷ Lệ %": f"{len(raw_val_split)/total_rows*100:.2f}%",
        "Mục Đích": "So sánh mô hình & tối ưu trọng số Ensemble"
    }
])

display(splits_summary)
"""))

# ==============================================================================
# Cell 11: Bước 5 — Preprocessing & Feature Engineering
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Bước 5 — Preprocessing và Feature Engineering Đa Tầng

Hệ thống đặc trưng được xây dựng theo kiến trúc đa tầng lũy tiến, kết hợp giữa tri thức ngành F&B, chu kỳ thời gian và các biến tương tác môi trường:

### 5.1. Hệ thống đặc trưng đa tầng lũy tiến:
1. **Tầng E1 (Lịch & Thuộc tính Cửa hàng):**
   - *Calendar features:* `dayofweek`, `month`, `quarter`, `day`, `is_weekend`.
   - *Sóng điều hòa chu kỳ (Harmonic features):* $\\sin / \\cos$ của ngày trong tuần (chu kỳ 7 ngày) và ngày trong năm (chu kỳ 365.25 ngày) giúp mô hình nắm bắt tính liên tục mượt mà.
   - *Metadata cửa hàng:* `store_age_days` (tuổi quán tính từ ngày khai trương), `seating_density` (mật độ chỗ ngồi / m² diện tích).
2. **Tầng E2 (Tín hiệu Khuyến mãi & Tương tác Cửa hàng):**
   - Cờ khuyến mãi `is_promo`.
   - Tương tác khuyến mãi cuối tuần: `promo_weekend = is_promo * is_weekend`.
   - Tương tác cửa hàng x thứ trong tuần: `store_dow_interaction`.
3. **Tầng E3 (Tương tác Môi trường & Thời tiết):**
   - Biến thời tiết: `temp_max`, `temp_min`, `precipitation`, cờ trời mưa `is_rain`.
   - Cờ ngày lễ quốc gia `is_holiday`.
   - Tương tác thời tiết x cuối tuần: `rain_weekend = is_rain * is_weekend`.
4. **Horizon-safe Out-of-Fold Historical Target Statistics (Đặc trưng cốt lõi):**
   - `log_hist_sw_med`: Trung vị log doanh thu theo cặp `(store_id, dayofweek)` tính thuần túy trên tập train.
   - `log_hist_s_med`: Trung vị log doanh thu tổng quát theo từng `store_id`.
   - **Cơ chế Fallback an toàn:** Nếu cửa hàng mới chưa có lịch sử (như S12 trong những ngày đầu), tự động fallback về trung vị toàn hệ thống `global_median`. Đảm bảo **zero leakage** và an toàn cho bài toán dự báo 90 ngày.
"""))

# ==============================================================================
# Cell 12: Bước 5.1 — Xây dựng Đặc trưng & Ma trận Huấn luyện
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 5. Xây dựng Đặc trưng và Chuẩn bị Ma trận Huấn luyện
# Bước 5.1: Sinh toàn bộ các tầng đặc trưng E1, E2, E3
featured_df = build_all_features(joined_df)

# Bước 5.2: Tách lại theo thời gian
featured_tr = featured_df[featured_df[DATE_COL] < PRIMARY_TRAIN_END].copy()
featured_va = featured_df[featured_df[DATE_COL] >= PRIMARY_TRAIN_END].copy()

# Bước 5.3: Chuẩn bị ma trận đặc trưng cho mô hình cây (kèm Out-of-fold Historical Target Stats)
train_mat, val_mat, _, tree_feature_cols = prepare_tree_feature_matrices(
    featured_tr, featured_va
)

y_train = train_mat[TARGET_COL].values
y_val   = val_mat[TARGET_COL].values

print(f"Đã tạo thành công {len(tree_feature_cols)} đặc trưng cho mô hình cây.")
print("Danh sách các đặc trưng đầu vào:")
print(", ".join(tree_feature_cols[:15]) + f", ... (và {len(tree_feature_cols)-15} đặc trưng khác)")
"""))

# ==============================================================================
# Cell 13: Bước 6 — Model Selection & Progressive Training
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Bước 6 — Model Selection và Progressive Training: Hành trình 5 Thế hệ Mô hình

Quy trình phát triển mô hình tuân thủ phương pháp tiếp cận khoa học lũy tiến: bắt đầu từ một baseline tối giản để thiết lập mốc chuẩn, sau đó nâng cấp dần lên các thuật toán tuyến tính, cây quyết định phi tuyến, tối ưu hóa hàm mục tiêu và kết hợp mô hình (ensemble):

```text
[Thế hệ 0: Baseline]  --->  [Thế hệ 1: Ridge E3]  --->  [Thế hệ 2: LightGBM Tuned]
WeekdayMedianBaseline        Linear Model (log1p)        Tree-based Nonlinear (MSE)
RMSLE = 0.2391              RMSLE = 0.1978 (+17.27%)     RMSLE = 0.1667 (+30.28%)
                                                                  |
                                                                  v
[Thế hệ 4: Champion Ensemble]  <---  [Thế hệ 3: LightGBM Huber Loss]
0.91 LGBM + 0.09 Ridge + Duan Smearing    Log1p Target + Huber Robust Loss
RMSLE = 0.1603 (+32.98% ĐỈNH CAO)         RMSLE = 0.1664 (+30.43%)
```

### Chi tiết 5 Mô hình:
1. **Model 0 (Baseline):** `WeekdayMedianBaseline` — Bảng lookup trung vị doanh thu lịch sử theo `(store_id, dayofweek)` với 2 tầng fallback an toàn (RMSLE = 0.2391).
2. **Model 1 (Linear Ridge Regression):** Hồi quy tuyến tính chính quy $L_2$ trên không gian `log1p(revenue)` sử dụng tầng đặc trưng E3 (thời tiết, ngày lễ, khuyến mãi) (RMSLE = 0.1978).
3. **Model 2 (LightGBM Tuned):** Cây quyết định tăng cường gradient phi tuyến tính, xử lý tương tác phi tuyến (`num_leaves=20, n_estimators=300`) (RMSLE = 0.1667).
4. **Model 3 (LightGBM Huber Loss):** Tối ưu hóa hàm mục tiêu robust **Huber Loss** trong không gian log. Kết hợp ưu điểm của $L_2$ ở sai số nhỏ và $L_1$ ở sai số lớn, triệt tiêu ảnh hưởng của các điểm đột biến doanh thu (RMSLE = 0.1664).
5. **Model 4 (Champion Weighted Ensemble + Duan Smearing):** Kết hợp tối ưu **0.91 LightGBM Huber + 0.09 Linear Ridge** kết hợp hệ số **Duan's Smearing Factor** ($s = 1.0222$) để triệt tiêu độ lệch Jensen khi nghịch đảo `expm1` (**RMSLE = 0.1603** | MAE = **2.33 triệu VND**).
"""))

# ==============================================================================
# Cell 14: Bước 6.1 — Huấn luyện & Đánh giá 5 Thế hệ Mô hình
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 6. Huấn luyện và Đánh giá 5 Thế hệ Mô hình trên Primary Validation (Q4/2025)
model_results = []

# --- Model 0: Baseline Median ---
print("[1/5] Huấn luyện Model 0: WeekdayMedianBaseline...")
m0_baseline = WeekdayMedianBaseline().fit(train_mat)
p_val_m0 = m0_baseline.predict(val_mat)
sc_m0 = compute_metrics(y_val, p_val_m0)
model_results.append({
    "Thế hệ": "GĐ 1: Baseline",
    "Tên Mô hình": "WeekdayMedianBaseline",
    "Không gian": "Doanh thu gốc",
    "RMSLE": sc_m0["rmsle"],
    "MAE (Tr. VND)": sc_m0["mae"],
    "RMSE": sc_m0["rmse"],
    "Cải thiện vs Baseline (%)": "0.0% (Mốc chuẩn)"
})

# --- Model 1: Linear Ridge (E3 features) ---
print("[2/5] Huấn luyện Model 1: Linear Ridge Regression (log1p)...")
m1_ridge = LogTransformedLinearModel(
    model_type="ridge", alpha=1.0, feature_tier="E3_weather_holiday"
)
m1_ridge.fit(train_mat, y_train)
p_val_m1 = m1_ridge.predict(val_mat)
sc_m1 = compute_metrics(y_val, p_val_m1)
impr_m1 = ((sc_m0["rmsle"] - sc_m1["rmsle"]) / sc_m0["rmsle"]) * 100
model_results.append({
    "Thế hệ": "GĐ 2: Linear",
    "Tên Mô hình": "Linear Ridge Regression (alpha=1.0)",
    "Không gian": "log1p",
    "RMSLE": sc_m1["rmsle"],
    "MAE (Tr. VND)": sc_m1["mae"],
    "RMSE": sc_m1["rmse"],
    "Cải thiện vs Baseline (%)": f"+{impr_m1:.2f}%"
})

# --- Model 2: LightGBM Tuned (MSE Loss) ---
print("[3/5] Huấn luyện Model 2: LightGBM Tuned (MSE Loss)...")
m2_lgbm_tuned = TransformedTargetTreeModel(
    transform_type="log1p", objective="regression",
    n_estimators=300, learning_rate=0.05, num_leaves=20,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
m2_lgbm_tuned.fit(train_mat[tree_feature_cols], y_train)
p_val_m2 = m2_lgbm_tuned.predict(val_mat[tree_feature_cols])
sc_m2 = compute_metrics(y_val, p_val_m2)
impr_m2 = ((sc_m0["rmsle"] - sc_m2["rmsle"]) / sc_m0["rmsle"]) * 100
model_results.append({
    "Thế hệ": "GĐ 3: Tree Non-linear",
    "Tên Mô hình": "LightGBM Tuned (MSE Loss)",
    "Không gian": "log1p",
    "RMSLE": sc_m2["rmsle"],
    "MAE (Tr. VND)": sc_m2["mae"],
    "RMSE": sc_m2["rmse"],
    "Cải thiện vs Baseline (%)": f"+{impr_m2:.2f}%"
})

# --- Model 3: LightGBM Huber Loss ---
print("[4/5] Huấn luyện Model 3: LightGBM Robust Huber Loss...")
m3_lgbm_huber = TransformedTargetTreeModel(
    transform_type="log1p", objective="huber",
    n_estimators=300, learning_rate=0.05, num_leaves=20,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
m3_lgbm_huber.fit(train_mat[tree_feature_cols], y_train)
p_val_m3 = m3_lgbm_huber.predict(val_mat[tree_feature_cols])
sc_m3 = compute_metrics(y_val, p_val_m3)
impr_m3 = ((sc_m0["rmsle"] - sc_m3["rmsle"]) / sc_m0["rmsle"]) * 100
model_results.append({
    "Thế hệ": "GĐ 4: Loss Optimization",
    "Tên Mô hình": "LightGBM Huber Loss (Robust)",
    "Không gian": "log1p",
    "RMSLE": sc_m3["rmsle"],
    "MAE (Tr. VND)": sc_m3["mae"],
    "RMSE": sc_m3["rmse"],
    "Cải thiện vs Baseline (%)": f"+{impr_m3:.2f}%"
})

# --- Model 4: Champion Ensemble + Duan Smearing ---
print("[5/5] Xây dựng Model 4: Champion Ensemble (0.91 LGBM + 0.09 Ridge + Duan Smearing)...")
# Dự báo log của từng mô hình trên validation
z_val_m3 = m3_lgbm_huber.predict_transformed(val_mat[tree_feature_cols])
z_val_m1 = np.log1p(p_val_m1)

# Kết hợp trọng số tối ưu trong log-space: 0.91 LightGBM + 0.09 Ridge
z_val_blend = 0.91 * z_val_m3 + 0.09 * z_val_m1

# Ước lượng hệ số Duan Smearing trên tập train
z_tr_m3 = m3_lgbm_huber.predict_transformed(train_mat[tree_feature_cols])
z_tr_m1 = np.log1p(m1_ridge.predict(train_mat))
z_tr_blend = 0.91 * z_tr_m3 + 0.09 * z_tr_m1
train_resids = np.log1p(y_train) - z_tr_blend
duan_factor = float(np.mean(np.exp(train_resids)))

# Tái biến đổi với Duan Smearing: y_hat = exp(z_hat) * s - 1
p_val_m4 = np.clip(np.exp(z_val_blend) * duan_factor - 1.0, 0, None)
sc_m4 = compute_metrics(y_val, p_val_m4)
impr_m4 = ((sc_m0["rmsle"] - sc_m4["rmsle"]) / sc_m0["rmsle"]) * 100
model_results.append({
    "Thế hệ": "GĐ 5: Champion Ensemble",
    "Tên Mô hình": f"Weighted Ensemble + Duan Smearing (s={duan_factor:.4f})",
    "Không gian": "log1p blend",
    "RMSLE": sc_m4["rmsle"],
    "MAE (Tr. VND)": sc_m4["mae"],
    "RMSE": sc_m4["rmse"],
    "Cải thiện vs Baseline (%)": f"+{impr_m4:.2f}% (Tối ưu nhất)"
})

df_progression = pd.DataFrame(model_results)
display(df_progression)
"""))

# ==============================================================================
# Cell 15: Bước 7 — Evaluation & Hyperparameter Tuning
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Bước 7 — Evaluation và Hyperparameter Tuning: Đánh giá Toàn diện & Kiểm chứng

Để đảm bảo mô hình hoạt động bền vững và không bị chi phối bởi các yếu tố mùa vụ cá biệt, quy trình đánh giá kết hợp kiểm định chuỗi thời gian 4 chu kỳ (Walk-Forward Cross-Validation), phân tích sai số chi tiết theo phân khúc (Subgroup Error Analysis) và chẩn đoán độ lệch kỳ vọng qua residuals:

### Các nội dung đánh giá chuyên sâu:
1. **Walk-Forward Cross-Validation qua 4 chu kỳ:** Kiểm tra tính ổn định qua các mùa vụ khác nhau (Fold A: Q1/2025, Fold B: Q2/2025, Fold C: Q3/2025 và Primary Val: Q4/2025).
2. **Phân tích sai số theo Phân khúc (Subgroup Error Analysis):**
   - *Phân khúc Loại cửa hàng (`store_type`):* Đánh giá khả năng dự báo tại các quán cà phê trường đại học, khu văn phòng, mặt phố.
   - *Phân khúc Khuyến mãi (`is_promo`):* Khắc phục hoàn toàn điểm mù dự báo thiếu khi có khuyến mãi của baseline.
   - *Phân khúc Thành phố.*
3. **Chẩn đoán Sai số Log Residuals & Trực quan hóa Actual vs Predicted.**
"""))

# ==============================================================================
# Cell 16: Bước 7.1 — Walk-Forward Backtesting & Subgroup Error Analysis
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 7.1. Walk-Forward Backtesting & Phân tích Chi tiết Lỗi theo Nhóm (Subgroup Analysis)

# --- 1. Walk-Forward Backtesting qua 4 Fold ---
wf_rows = []
for fold_key in ["fold_a", "fold_b", "fold_c", "primary_val"]:
    cfg = WALK_FORWARD_FOLDS[fold_key]
    tr_end = cfg["train_end"]
    val_st = cfg["val_start"]
    val_en = cfg["val_end"]
    fold_name = cfg["name"]
    
    # Split fold
    sub_tr = featured_df[featured_df[DATE_COL] < tr_end].copy()
    sub_va = featured_df[(featured_df[DATE_COL] >= val_st) & (featured_df[DATE_COL] < val_en)].copy()
    
    sub_tr_m, sub_va_m, _, _ = prepare_tree_feature_matrices(sub_tr, sub_va)
    y_sub_tr = sub_tr_m[TARGET_COL].values
    y_sub_va = sub_va_m[TARGET_COL].values
    
    # Baseline
    b_mod = WeekdayMedianBaseline().fit(sub_tr_m)
    p_b = b_mod.predict(sub_va_m)
    m_b = compute_metrics(y_sub_va, p_b)
    
    # Ridge
    r_mod = LogTransformedLinearModel(model_type="ridge", alpha=1.0, feature_tier="E3_weather_holiday")
    r_mod.fit(sub_tr_m, y_sub_tr)
    p_r = r_mod.predict(sub_va_m)
    m_r = compute_metrics(y_sub_va, p_r)
    
    # LGBM Huber
    l_mod = TransformedTargetTreeModel(
        transform_type="log1p", objective="huber",
        n_estimators=300, learning_rate=0.05, num_leaves=20,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    l_mod.fit(sub_tr_m[tree_feature_cols], y_sub_tr)
    p_l = l_mod.predict(sub_va_m[tree_feature_cols])
    m_l = compute_metrics(y_sub_va, p_l)
    
    # Champion Ensemble
    z_sub_l = l_mod.predict_transformed(sub_va_m[tree_feature_cols])
    z_sub_r = np.log1p(p_r)
    z_sub_ens = 0.91 * z_sub_l + 0.09 * z_sub_r
    
    # Smearing factor trên sub_tr
    z_tr_l = l_mod.predict_transformed(sub_tr_m[tree_feature_cols])
    z_tr_r = np.log1p(r_mod.predict(sub_tr_m))
    s_factor = float(np.mean(np.exp(np.log1p(y_sub_tr) - (0.91 * z_tr_l + 0.09 * z_tr_r))))
    
    p_ens = np.clip(np.exp(z_sub_ens) * s_factor - 1.0, 0, None)
    m_ens = compute_metrics(y_sub_va, p_ens)
    
    wf_rows.append({
        "Chu kỳ Walk-Forward": fold_name,
        "Baseline RMSLE": m_b["rmsle"],
        "Ridge RMSLE": m_r["rmsle"],
        "LGBM Huber RMSLE": m_l["rmsle"],
        "Champion Ensemble RMSLE": m_ens["rmsle"],
        "Cải thiện vs Baseline (%)": f"+{((m_b['rmsle']-m_ens['rmsle'])/m_b['rmsle']*100):.2f}%",
        "Ensemble MAE (Tr. VND)": m_ens["mae"]
    })

print("BẢNG KẾT QUẢ WALK-FORWARD BACKTESTING QUA 4 CHU KỲ KIỂM ĐỊNH:")
display(pd.DataFrame(wf_rows))

# --- 2. Phân tích Subgroup theo Loại Cửa Hàng & Khuyến Mãi ---
subgroups_report = evaluate_subgroups(val_mat, p_val_m4, stores_df=stores_raw)
print("\\nPHÂN TÍCH HIỆU NĂNG THEO LOẠI HÌNH CỬA HÀNG (STORE TYPE):")
display(subgroups_report["by_store_type"])

print("\\nPHÂN TÍCH HIỆU NĂNG THEO TRẠNG THÁI KHUYẾN MÃI (IS_PROMO):")
display(subgroups_report["by_promo"])
"""))

# ==============================================================================
# Cell 17: Bước 7.2 — Trực quan hóa Đánh giá Hiệu năng & Chẩn đoán Sai số
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 7.2. Trực quan hóa Đánh giá Hiệu năng & Chẩn đoán Sai số Residual
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. So sánh RMSLE của 5 Thế hệ Mô hình
ax1 = axes[0, 0]
model_names = [r["Tên Mô hình"].split("(")[0].strip() for r in model_results]
rmsle_vals = [r["RMSLE"] for r in model_results]
colors = ["#7f8c8d", "#3498db", "#2ecc71", "#9b59b6", "#e74c3c"]

bars = ax1.bar(model_names, rmsle_vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
ax1.set_title("Hành trình Cải thiện RMSLE qua 5 Thế hệ Mô hình", fontweight="bold")
ax1.set_ylabel("RMSLE (Thấp hơn là tốt hơn)")
ax1.set_ylim(0.14, 0.26)
ax1.tick_params(axis="x", rotation=20)
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f"{height:.4f}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontweight="bold")

# 2. Scatter Actual vs Predicted của Champion Ensemble trên Primary Validation
ax2 = axes[0, 1]
ax2.scatter(y_val, p_val_m4, alpha=0.35, color="#2980b9", edgecolors="none", s=25, label="Mẫu Q4/2025")
max_val = max(y_val.max(), p_val_m4.max())
ax2.plot([0, max_val], [0, max_val], color="#e74c3c", linestyle="--", lw=2, label="Đường lý tưởng (y = y_hat)")
ax2.set_title(f"Doanh thu Thực tế vs Dự báo (Champion Ensemble: RMSLE = {sc_m4['rmsle']:.4f})", fontweight="bold")
ax2.set_xlabel("Doanh thu Thực tế (Triệu VND)")
ax2.set_ylabel("Doanh thu Dự báo (Triệu VND)")
ax2.set_xlim(0, 75)
ax2.set_ylim(0, 75)
ax2.legend(loc="upper left")

# 3. Phân phối Sai số Log Residuals: log(1+y) - log(1+y_hat)
ax3 = axes[1, 0]
resids_ens = np.log1p(y_val) - np.log1p(p_val_m4)
resids_base = np.log1p(y_val) - np.log1p(p_val_m0)

sns.kdeplot(resids_base, ax=ax3, color="#95a5a6", fill=True, alpha=0.3, label=f"Baseline (Mean: {resids_base.mean():+.3f}, Std: {resids_base.std():.3f})")
sns.kdeplot(resids_ens, ax=ax3, color="#27ae60", fill=True, alpha=0.5, label=f"Champion Ensemble (Mean: {resids_ens.mean():+.3f}, Std: {resids_ens.std():.3f})")
ax3.axvline(0, color="black", linestyle=":", lw=1.5)
ax3.set_title("Phân phối Sai số Log Residuals (Độ lệch kỳ vọng)", fontweight="bold")
ax3.set_xlabel("Log Residual = log(1 + y) - log(1 + y_hat)")
ax3.set_xlim(-1.0, 1.0)
ax3.legend(loc="upper right")

# 4. So sánh Dãy thời gian Thực tế vs Dự báo 35 ngày của một cửa hàng mẫu (S01)
ax4 = axes[1, 1]
val_mat_sample = val_mat[val_mat[STORE_COL] == "S01"].sort_values(DATE_COL).head(35)
sample_dates = val_mat_sample[DATE_COL]
sample_actual = val_mat_sample[TARGET_COL].values
sample_idx = val_mat_sample.index
sample_pred_ens = p_val_m4[val_mat.index.get_indexer(sample_idx)]
sample_pred_base = p_val_m0[val_mat.index.get_indexer(sample_idx)]

ax4.plot(sample_dates, sample_actual, marker="o", color="black", lw=2, label="Thực tế (Ground-Truth)")
ax4.plot(sample_dates, sample_pred_ens, marker="s", color="#27ae60", lw=2, linestyle="-", label="Champion Ensemble")
ax4.plot(sample_dates, sample_pred_base, marker="x", color="#e74c3c", lw=1.5, linestyle="--", label="Baseline Median")
ax4.set_title("Theo dõi Dự báo 35 ngày tại Cửa hàng S01 (Q4/2025)", fontweight="bold")
ax4.set_xlabel("Ngày")
ax4.set_ylabel("Doanh thu (Triệu VND)")
ax4.tick_params(axis="x", rotation=30)
ax4.legend(loc="upper left")

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 18: Bước 8 — Final Retraining trên Toàn Bộ Dữ Liệu Lịch Sử
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Bước 8 — Final Retraining: Tái huấn luyện Toàn bộ Dữ liệu Lịch sử chuẩn bị cho Kaggle Test

Theo nguyên tắc chuẩn mực tại Mục 2.3 của [Regression_Plan.md](../Regression_Plan.md):
> *"Sau khi chốt mô hình tốt nhất, khi giảng viên công bố file `test.csv` trên Kaggle, toàn bộ pipeline sẽ được retrain trên toàn bộ 26.500 dòng sạch (toàn bộ năm 2024–2025) để suy luận và xuất file `submission.csv`."*

### Ý nghĩa của việc Retrain trên toàn bộ 26.500 dòng sạch:
1. **Tận dụng tối đa thông tin gần nhất:** Việc bổ sung thêm 3.832 dòng của Quý 4/2025 giúp mô hình cập nhật xu hướng kinh doanh và hành vi tiêu dùng sát nhất với thời điểm kiểm tra của Quý 1/2026.
2. **Cập nhật Bảng Thống kê Lịch sử (Target Statistics):** Tính toán lại trung vị lịch sử trên toàn bộ 2 năm giúp giảm phương sai ước lượng, đặc biệt với các cửa hàng mới mở trong năm 2025 như S12.
3. **Mô hình sẵn sàng xuất trận:** Sau khi retrain, pipeline mô hình Champion bao trùm toàn bộ tri thức từ quá khứ, sẵn sàng phục vụ suy luận cho file `test.csv` chính thức.
"""))

# ==============================================================================
# Cell 19: Bước 8.1 — Thực thi Retrain trên Toàn Bộ 26.500 Dòng
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 8. Thực thi Retrain Toàn Bộ Pipeline trên 26.500 dòng sạch (2024 - 2025)
print("Đang tiến hành Retrain Champion Ensemble trên toàn bộ 26.500 dòng sạch...")

# 1. Tính toán ma trận đặc trưng trên toàn bộ dataset
full_train_mat, _, _, _ = prepare_tree_feature_matrices(featured_df, featured_df)
y_full = full_train_mat[TARGET_COL].values

# 2. Fit Linear Ridge trên toàn bộ dữ liệu
full_ridge = LogTransformedLinearModel(model_type="ridge", alpha=1.0, feature_tier="E3_weather_holiday")
full_ridge.fit(full_train_mat, y_full)

# 3. Fit LightGBM Huber trên toàn bộ dữ liệu
full_lgbm_huber = TransformedTargetTreeModel(
    transform_type="log1p", objective="huber",
    n_estimators=300, learning_rate=0.05, num_leaves=20,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
full_lgbm_huber.fit(full_train_mat[tree_feature_cols], y_full)

# 4. Tính toán hệ số Duan Smearing trên toàn bộ dữ liệu
z_full_lgbm = full_lgbm_huber.predict_transformed(full_train_mat[tree_feature_cols])
z_full_ridge = np.log1p(full_ridge.predict(full_train_mat))
z_full_blend = 0.91 * z_full_lgbm + 0.09 * z_full_ridge
full_resids = np.log1p(y_full) - z_full_blend
full_duan_factor = float(np.mean(np.exp(full_resids)))

print(f"Retrain hoàn tất!")
print(f" - Tổng số dòng huấn luyện:    {len(full_train_mat):,} dòng (01/01/2024 -> 31/12/2025)")
print(f" - Hệ số Duan Smearing toàn cục: {full_duan_factor:.4f}")
"""))

# ==============================================================================
# Cell 20: Bước 9 — Lưu Model và Triển Khai
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Bước 9 — Lưu model, Triển khai & Xuất File Submission: Artifact & Inference Pipeline

Để phục vụ triển khai trong môi trường vận hành thực tế (Production) và nộp bài thi trên nền tảng Kaggle, toàn bộ pipeline hoàn chỉnh bao gồm các mô hình thành phần, trọng số kết hợp và hệ số hiệu chỉnh Duan Smearing được đóng gói và serialize thành một tệp artifact duy nhất:

1. Đóng gói mô hình **Champion Ensemble** hoàn chỉnh cùng siêu tham số và hệ số Duan Smearing vào tệp artifact: `models/best_production_model.joblib`.
2. Xây dựng hàm suy luận sản xuất `predict_cafe_revenue(raw_new_df)`. Hàm này nhận trực tiếp dữ liệu thô mới, tự động join thông tin cửa hàng, thời tiết, ngày lễ, sinh các đặc trưng cần thiết và trả về doanh thu dự báo bằng triệu VND.
3. Tải lại artifact bằng `joblib.load()` và thực hiện kiểm thử suy luận (Inference Verification Test).
4. Xuất file kết quả dự báo mẫu kiểm thử: `reports/production_predictions_sample.csv`.
5. Tạo và xuất file nộp bài chính thức **`submission.csv`** đạt chuẩn 100% định dạng Kaggle (gồm 2 cột `id` và `revenue`, lưu tại thư mục gốc của dự án).
"""))

# ==============================================================================
# Cell 21: Bước 9.1 — Đóng gói Artifact, Tải lại & Chạy Kiểm thử Suy luận
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 9. Đóng gói Mô hình Thành Sản Phẩm & Kiểm Thử Suy Luận (Inference Verification)

# Bước 9.1: Khởi tạo Pipeline đóng gói mô hình Ensemble
champion_pipeline = WeightedLogEnsemble(
    base_models={"lgbm_huber": full_lgbm_huber, "ridge": full_ridge},
    weights={"lgbm_huber": 0.91, "ridge": 0.09},
    use_smearing=True,
    random_state=42
)
champion_pipeline.is_fitted = True
champion_pipeline.model_names_ = ["lgbm_huber", "ridge"]
champion_pipeline.weights_ = {"lgbm_huber": 0.91, "ridge": 0.09}
champion_pipeline.duan_factor_ = full_duan_factor

# Lưu artifact vào thư mục models/
prod_model_path = MODELS_DIR / "best_production_model.joblib"
champion_pipeline.save(prod_model_path)
print(f"Đã lưu thành công Champion Model Pipeline vào: {prod_model_path}")

# Bước 9.2: Tải lại mô hình bằng joblib để mô phỏng Server triển khai
loaded_prod_model = joblib.load(prod_model_path)
print("Đã tải lại thành công mô hình từ đĩa cứng.")

# Bước 9.3: Định nghĩa hàm suy luận hoàn chỉnh phục vụ Production / API
def predict_cafe_revenue(raw_input_df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Hàm suy luận đầu cuối nhận dữ liệu thô mới và trả về dự báo doanh thu thực tế.
    
    Quy trình thực thi:
    1. Ghép nối metadata (stores, weather, holidays)
    2. Sinh các đặc trưng đa tầng E1, E2, E3
    3. Ánh xạ đặc trưng cây sử dụng bảng thống kê lịch sử OOF từ toàn bộ tập dữ liệu đã học
    4. Dự báo bằng Weighted Ensemble kết hợp Duan Smearing
    \"\"\"
    # Join dữ liệu tham chiếu
    joined = join_all_data(raw_input_df, stores_raw, weather_raw, holidays_raw)
    feat = build_all_features(joined)
    
    # Ánh xạ đặc trưng cây sử dụng bảng thống kê lịch sử OOF từ toàn bộ dataset
    stats = compute_historical_target_stats(featured_df)
    input_mat = apply_historical_target_stats(feat, stats)
    
    # Suy luận log-space từ 2 mô hình thành phần
    z_lgbm = loaded_prod_model.base_models["lgbm_huber"].predict_transformed(input_mat[tree_feature_cols])
    p_ridge = loaded_prod_model.base_models["ridge"].predict(input_mat)
    z_ridge = np.log1p(np.clip(p_ridge, 0, None))
    
    # Kết hợp trọng số 91/9 & Duan Smearing
    z_blend = 0.91 * z_lgbm + 0.09 * z_ridge
    preds_revenue = np.clip(np.exp(z_blend) * loaded_prod_model.duan_factor_ - 1.0, 0, None)
    
    result_df = raw_input_df[[DATE_COL, STORE_COL, PROMO_COL]].copy()
    result_df["predicted_revenue_million_vnd"] = np.round(preds_revenue, 3)
    return result_df

# Bước 9.4: Chạy kiểm thử suy luận trên 10 dòng đầu của Q4/2025 (dữ liệu thô thực tế: date, store_id, is_promo)
raw_test_input = clean_train[clean_train[DATE_COL] >= PRIMARY_TRAIN_END].head(10).copy()
sample_input = raw_test_input[[DATE_COL, STORE_COL, PROMO_COL]].copy()
sample_predictions = predict_cafe_revenue(sample_input)
sample_predictions["actual_revenue"] = raw_test_input[TARGET_COL].values
sample_predictions["absolute_error"] = np.abs(sample_predictions["predicted_revenue_million_vnd"] - sample_predictions["actual_revenue"])

print("\\nKẾT QUẢ KIỂM THỬ SUY LUẬN TRÊN DỮ LIỆU THỰC TẾ:")
display(sample_predictions)

# Xuất file dự báo mẫu ra reports/
out_pred_path = REPORTS_DIR / "production_predictions_sample.csv"
sample_predictions.to_csv(out_pred_path, index=False)
print(f"Đã xuất kết quả kiểm thử ra: {out_pred_path}")
"""))

# ==============================================================================
# Cell 22: Bước 9.2 — Tạo và Xuất Tệp Nộp Bài Kaggle (submission.csv)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 9.2. Tạo và Xuất Tệp Nộp Bài Kaggle (submission.csv)

Tệp nộp bài chính thức của cuộc thi Kaggle yêu cầu cấu trúc chuẩn:
- **Tên tệp:** `submission.csv` (lưu tại thư mục gốc của dự án, phù hợp với quy chuẩn output của Kaggle).
- **Cấu trúc:** Đúng 2 cột `id` và `revenue`.
- **Ràng buộc:** Giữ nguyên thứ tự `id` từ tệp test, không chứa giá trị thiếu (NaN/null), và giá trị doanh thu dự báo không âm ($y \\ge 0$).

Đoạn mã dưới đây hỗ trợ tự động nhận diện cả môi trường thi đấu Kaggle (nơi tệp `test.csv` được nạp từ `/kaggle/input/...`) và môi trường kiểm thử máy cục bộ, thực thi dự báo từ pipeline mô hình Champion Ensemble và xuất tệp `submission.csv` đạt chuẩn:
"""))

# ==============================================================================
# Cell 23: Bước 9.2 Code — Thực thi Xuất file submission.csv
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 9.2. Quy trình sinh và xuất tệp submission.csv nộp lên Kaggle
import glob

# 1. Tìm kiếm file test.csv (tự động nhận diện trên Kaggle và máy cục bộ)
test_candidates = [
    Path("test.csv"),
    Path("data/raw/test.csv"),
    Path("../data/raw/test.csv"),
    Path("../data/cafe/test.csv"),
    Path("/kaggle/input/cafe-revenue-regression/test.csv"),
    Path("/kaggle/input/cafe-revenue/test.csv"),
]

# Tìm kiếm mở rộng bằng glob nếu đang chạy trên Kaggle
kaggle_hits = glob.glob("/kaggle/input/**/test.csv", recursive=True)
if kaggle_hits:
    test_candidates.insert(0, Path(kaggle_hits[0]))

official_test_path = None
for p in test_candidates:
    if p.exists():
        official_test_path = p
        break

# 2. Xử lý tập dữ liệu Test để sinh submission
if official_test_path is not None:
    print(f"Đã tìm thấy tệp test.csv chính thức tại: {official_test_path}")
    test_raw_df = pd.read_csv(official_test_path, parse_dates=[DATE_COL])
else:
    # Môi trường máy cục bộ: sử dụng tập dữ liệu validation Q4/2025 để minh chứng quy trình xuất submission
    print("Môi trường cục bộ: Sử dụng tập dữ liệu validation Q4/2025 để minh chứng quy trình xuất submission.")
    clean_test_source = clean_train[clean_train[DATE_COL] >= PRIMARY_TRAIN_END].reset_index(drop=True)
    test_raw_df = clean_test_source[[DATE_COL, STORE_COL, PROMO_COL]].copy()
    test_raw_df["id"] = range(100000, 100000 + len(test_raw_df))

# 3. Chạy suy luận từ Champion Ensemble Pipeline
test_preds_df = predict_cafe_revenue(test_raw_df)

# 4. Định dạng DataFrame submission theo đúng quy chuẩn Kaggle (id, revenue)
submission_df = pd.DataFrame({
    "id": test_raw_df["id"],
    "revenue": np.round(np.clip(test_preds_df["predicted_revenue_million_vnd"], 0, None), 2)
})

# 5. Xuất file submission.csv ra thư mục gốc (chuẩn Kaggle)
submission_output_path = Path("submission.csv")
submission_df.to_csv(submission_output_path, index=False)

print("Đã xuất tệp nộp bài thành công:")
print(f" - Đường dẫn nộp bài (Kaggle submission): {submission_output_path.resolve()}")
print(f" - Kích thước tệp:                        {len(submission_df):,} dòng x {submission_df.shape[1]} cột")
print(f" - Kiểm tra giá trị Null:                 {submission_df.isnull().sum().to_dict()}")

# 6. Kiểm tra các dòng đầu và thống kê mô tả doanh thu dự báo
print(\"\\n10 dòng đầu tiên của tệp submission.csv:\")
display(submission_df.head(10))

print(\"\\nThống kê mô tả doanh thu dự báo trong submission.csv:\")
display(submission_df[\"revenue\"].describe().round(2).to_frame().T)
"""))

# ==============================================================================
# Cell 24: Bước 10 — Monitoring & Retraining
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 10. Bước 10 — Monitoring và Retraining: Vận hành MLOps Thực tế

Sau khi mô hình được đưa vào vận hành thực tế, chất lượng dự báo có thể suy giảm theo thời gian do sự thay đổi của môi trường kinh doanh và hành vi khách hàng. Hệ thống MLOps thiết lập 4 trụ cột giám sát và vận hành:

### 10.1. Giám sát Trôi dạt Dữ liệu (Data Drift Monitoring)
- **Đặc trưng khí tượng:** Phân phối lượng mưa và nhiệt độ có thể thay đổi đột ngột giữa các năm. Sử dụng kiểm định **Population Stability Index (PSI)** hoặc **Kolmogorov-Smirnov (KS-test)** để phát hiện drift nếu PSI > 0.2.
- **Biến động cửa hàng:** Thay đổi diện tích quán, sắp xếp lại bàn ghế làm thay đổi mật độ ghế `seating_density`.
- **Mở thêm cửa hàng mới:** Khi chuỗi mở thêm cửa hàng mới, dữ liệu chưa có lịch sử bán hàng; hệ thống kích hoạt chế độ fallback an toàn về median theo loại hình cửa hàng (`store_type`).

### 10.2. Giám sát Hiệu năng Mô hình (Concept & Performance Drift)
- Trong bài toán dự báo doanh thu hàng ngày, nhãn thực tế (`revenue`) có thể được ghi nhận ngay vào cuối mỗi ngày từ hệ thống máy tính tiền POS.
- Hệ thống tự động tính toán **Rolling 7-day RMSLE** và **Rolling 30-day RMSLE**.
- **Ngưỡng cảnh báo suy giảm hiệu năng (Performance Alert):**
  - Trạng thái Bình thường (Healthy): $\\text{RMSLE} \\le 0.1800$.
  - Cảnh báo Nhẹ (Warning): $0.1800 < \\text{RMSLE} \\le 0.2200$.
  - Báo động Đỏ (Critical Alert): $\\text{RMSLE} > 0.2200$ (kích hoạt kiểm tra dữ liệu đầu vào và xem xét retrain khẩn cấp).

### 10.3. Chiến lược Tái huấn luyện Tự động (Retraining Triggers)
1. **Tái huấn luyện định kỳ (Scheduled Retraining):**
   - Tần suất: **Hàng quý (mỗi 3 tháng)** khi chu kỳ kinh doanh của quý vừa kết thúc.
   - Bổ sung dữ liệu quý mới vào tập Train Set, cập nhật lại bảng thống kê lịch sử `log_hist_sw_med`, fit lại LightGBM Huber và Linear Ridge.
2. **Tái huấn luyện theo sự kiện (Event-driven Retraining):**
   - Khi chuỗi cà phê thay đổi menu toàn hệ thống, điều chỉnh giá bán niêm yết trên 15%, hoặc triển khai chương trình loyalty mới làm thay đổi vĩnh viễn thói quen chi tiêu của khách hàng.

### 10.4. Kiến trúc Triển khai An toàn (Safe Deployment Strategy)
- **Shadow Deployment:** Chạy mô hình mới song song cùng mô hình hiện tại trong 14 ngày. Chỉ thay thế mô hình cũ nếu RMSLE thực tế của mô hình mới thấp hơn mô hình cũ.
- **Failover / Fallback Mechanism:** Nếu server mô hình gặp sự cố độ trễ hoặc thiếu đặc trưng thời tiết, hệ thống tự động fallback về **Model 0 (WeekdayMedianBaseline)** đảm bảo dịch vụ luôn thông suốt 99.99%.
"""))

# ==============================================================================
# Cell 25: Checklist & Nguyên Tắc Quan Trọng
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 11. Checklist Nghiệm Thu Kỹ Thuật và Các Nguyên Tắc Cốt Lõi

Bảng đối chiếu các tiêu chí kỹ thuật đã hoàn thành trong toàn bộ vòng đời phát triển mô hình:

| STT | Tiêu chí Kiểm định Chất lượng | Trạng thái | Minh chứng Thực tế trong Dự án |
|---|---|:---:|---|
| 1 | Xác định rõ bài toán, biến mục tiêu và horizon | Đạt | Dự báo `revenue` hàng ngày, horizon 90 ngày, metric RMSLE |
| 2 | Kiểm tra chất lượng dữ liệu và thực hiện EDA | Đạt | Phân tích 4 bảng, phát hiện độ lệch +11.71, boxplot thứ/khuyến mãi |
| 3 | Khử trùng lặp và làm sạch dữ liệu | Đạt | Khử 132 dòng trùng, sửa ngoại lệ S12, khóa `(date, store_id)` duy nhất |
| 4 | Kiểm soát nguy cơ Data Leakage | Đạt | Không dùng thông tin tương lai, fit imputer/stats thuần túy trên train |
| 5 | Chia dữ liệu phù hợp cấu trúc chuỗi thời gian | Đạt | Temporal Split (Train < 2025-10-01, Val Q4/2025 đồng bộ Starter Notebook) |
| 6 | Preprocessing và Feature Engineering đa tầng | Đạt | E1 lịch/store, E2 khuyến mãi, E3 thời tiết, out-of-fold target stats |
| 7 | Xây dựng Baseline chuẩn mực | Đạt | `WeekdayMedianBaseline` với 2 tầng fallback (RMSLE = 0.2391 clean / 0.2387 raw) |
| 8 | Lựa chọn và so sánh các họ mô hình ứng viên | Đạt | Tuyến tính (Ridge) $\\to$ Cây (LGBM) $\\to$ Huber Loss $\\to$ Weighted Ensemble |
| 9 | Chọn metric tối ưu phù hợp bài toán | Đạt | RMSLE phản ánh sai số tương đối, kết hợp MAE và Log Residual |
| 10 | Hyperparameter Tuning & Cross-validation | Đạt | Walk-Forward 4 chu kỳ (Fold A, B, C, Val), tối ưu hóa trọng số SLSQP |
| 11 | Đánh giá độc lập trên Primary Validation | Đạt | Đánh giá trên Q4/2025 (3.832 dòng): Champion đạt RMSLE = 0.1603 (+32.98%) |
| 12 | Đóng gói model artifact và xuất submission | Đạt | Tệp `best_production_model.joblib`, hàm `predict_cafe_revenue()` và `submission.csv` |
| 13 | Lập kế hoạch Monitoring & Retraining | Đạt | Giám sát Data Drift (PSI), Concept Drift, ngưỡng cảnh báo RMSLE > 0.22 |

---

### 4 Nguyên Tắc Cốt Lõi Được Bảo Toàn:
1. **Không để rò rỉ dữ liệu:** Toàn bộ đặc trưng thống kê lịch sử đều tuân thủ nguyên tắc out-of-fold và tính toán trước thời điểm cắt $T_0$.
2. **Không chỉ nhìn vào một con số:** Báo cáo chi tiết trên cả RMSLE, MAE, RMSE, độ lệch log residual và phân tích subgroup theo loại cửa hàng, khuyến mãi, thành phố.
3. **Không coi training score là chất lượng thực tế:** Mọi quyết định đều dựa trên Walk-Forward Validation và tập kiểm định Q4/2025 độc lập hoàn toàn.
4. **Ưu tiên khả năng tái lập (Reproducibility):** Cố định random seed 42, lưu trữ đầy đủ artifact, code được module hóa rõ ràng, có thể chạy lại từ đầu đến cuối một cách nhất quán.
"""))

# Thiết lập metadata cho notebook
nb["cells"] = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.9",
    },
}

out_path = project_root / "notebooks" / "end_to_end_training_pipeline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Đã tạo khung notebook cấu trúc hoàn chỉnh tại: {out_path}")
print("Đang tiến hành thực thi notebook từ đầu đến cuối...")

client = NotebookClient(nb, timeout=1200, kernel_name="python3")
client.execute()

with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Hoàn thành thực thi và lưu notebook tại: {out_path}")
