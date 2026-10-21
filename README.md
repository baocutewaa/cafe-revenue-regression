# Cafe Revenue Regression — Dự Báo Doanh Thu Cửa Hàng Cà Phê

Dự án Machine Learning dự báo doanh thu hàng ngày theo từng cửa hàng cà phê (*panel time-series regression*).

- **Biến mục tiêu:** `revenue` (triệu VND/ngày, không âm)
- **Metric đánh giá:** RMSLE (Root Mean Squared Logarithmic Error)
- **Kế hoạch tổng thể:** xem chi tiết tại [`Regression_Plan.md`](Regression_Plan.md)

---

## 1. Cấu trúc thư mục

```text
cafe-revenue-regression/
├── data/
│   ├── raw/                 # train.csv, stores.csv, weather.csv, holidays.csv
│   └── processed/           # train.csv, val.csv, test.csv đã xử lý trùng lặp & predictions
├── regression_splits/       # Thư mục split chuẩn (train.csv, val.csv, test.csv)
├── models/
│   ├── baseline_median.json # Model artifact lưu các bảng lookup trung vị (Giai đoạn 1)
│   ├── best_linear_model.joblib # Pipeline mô hình tuyến tính Ridge tốt nhất (Giai đoạn 2)
│   └── best_tree_model.joblib   # Pipeline mô hình LightGBM Tuned tốt nhất (Giai đoạn 3)
├── notebooks/
│   ├── 01_baseline_reproduction.ipynb  # Notebook tái lập và phân tích baseline (GĐ 1)
│   ├── 02_linear_models.ipynb          # Notebook thực nghiệm & phân tích hệ số tuyến tính (GĐ 2)
│   └── 03_tree_models.ipynb            # Notebook thực nghiệm & phân tích phi tuyến cây (GĐ 3)
├── reports/
│   ├── baseline_report.md              # Báo cáo chi tiết Giai đoạn 1
│   ├── baseline_val_predictions.csv    # Kết quả dự báo baseline trên Q3/2025
│   ├── linear_models_report.md         # Báo cáo chi tiết Giai đoạn 2 (Mô hình tuyến tính)
│   ├── linear_val_predictions.csv      # Kết quả dự báo và residual của Ridge trên Q3/2025
│   ├── tree_models_report.md           # Báo cáo chi tiết Giai đoạn 3 (Tree-based models)
│   ├── tree_val_predictions.csv        # Kết quả dự báo và residual của LightGBM trên Q3/2025
│   └── validation_results.csv          # Bảng tổng hợp điểm số qua các fold và mô hình
├── src/
│   ├── config.py            # Cấu hình đường dẫn, thời gian split và hằng số
│   ├── data/
│   │   ├── load.py          # Load dữ liệu, deduplication, sinh temporal splits
│   │   └── join.py          # Ghép nối an toàn đa nguồn (stores, weather, holidays)
│   ├── features/
│   │   ├── calendar.py      # Đặc trưng lịch và chu kỳ điều hòa sin/cos
│   │   ├── store.py         # Tuổi cửa hàng, mật độ ghế, tương tác store_dow
│   │   ├── weather_holiday.py # Tương tác khuyến mãi cuối tuần, cờ mưa, ngày lễ
│   │   ├── builder.py       # Pipeline tổng hợp và registry các tầng feature (E1, E2, E3)
│   │   └── tree_builder.py  # Xử lý ngoại lệ S12 và thống kê target out-of-fold cho cây
│   ├── evaluation/
│   │   ├── metrics.py       # RMSLE, MAE, RMSE, log residual
│   │   └── backtest.py      # Walk-forward backtesting và phân tích nhóm
│   ├── models/
│   │   ├── baseline.py      # WeekdayMedianBaseline với 2 tầng fallback
│   │   ├── linear.py        # LogTransformedLinearModel (Ridge, Huber, ElasticNet, OLS)
│   │   └── tree.py          # LogTransformedTreeModel (LightGBM, HistGBR, RF, ExtraTrees)
│   └── experiments/
│       ├── run_baseline.py      # Pipeline chính thực thi Giai đoạn 1
│       ├── run_linear_models.py  # Pipeline chính thực thi Giai đoạn 2
│       └── run_tree_models.py    # Pipeline chính thực thi Giai đoạn 3
├── tests/
│   ├── test_metrics.py       # Unit test cho hàm metric
│   ├── test_baseline.py      # Unit test cho baseline model
│   ├── test_splits.py        # Unit test cho hợp đồng dữ liệu và splits
│   ├── test_features.py      # Unit test cho feature engineering & join an toàn
│   ├── test_linear_models.py # Unit test cho pipeline mô hình tuyến tính
│   └── test_tree_models.py   # Unit test cho pipeline mô hình cây quyết định
├── scripts/
│   ├── build_notebook.py         # Script dựng notebook 01
│   ├── build_linear_notebook.py  # Script dựng notebook 02
│   └── build_tree_notebook.py    # Script dựng notebook 03
├── .gitignore               # Cấu hình bỏ qua các file tạm, logs, venv
├── requirements.txt         # Danh sách thư viện phụ thuộc (pinned versions)
├── venv/                    # Môi trường ảo Python (cục bộ, không commit git)
└── README.md
```

---

## 2. Kết quả Giai đoạn 1 — Baseline có thể tái lập

> Báo cáo phân tích chi tiết: [`reports/baseline_report.md`](reports/baseline_report.md)

- **Train:** `< 2025-07-01` (18.982 dòng, 71,63%)
- **Validation (Q3/2025):** `2025-07-01 <= date < 2025-10-01` (3.686 dòng, 13,91%) -> **RMSLE = 0.3101** | **MAE = 4.31 triệu VND**
- **Test Holdout (Q4/2025):** `2025-10-01 <= date < 2026-01-01` (3.832 dòng, 14,46%)

---

## 3. Kết quả Giai đoạn 2 — Mô hình tuyến tính (Linear Regression Models)

> Báo cáo phân tích chi tiết: [`reports/linear_models_report.md`](reports/linear_models_report.md)  
> Notebook thực nghiệm: [`notebooks/02_linear_models.ipynb`](notebooks/02_linear_models.ipynb)  
> Model Artifact: [`models/best_linear_model.joblib`](models/best_linear_model.joblib)

### 3.1. Bảng tiến trình thực nghiệm trên Primary Validation (Q3/2025)

| Thí nghiệm | Họ mô hình | Tầng Feature | Siêu tham số | RMSLE | MAE (Tr. VND) | Cải thiện so với Baseline |
|---|---|---|---|---|---|---|
| **E0_baseline** | MedianBaseline | store_weekday_median | Không | **0.3101** | 4.31 | 0.0% (Mốc chuẩn) |
| **E1_calendar_store** | Ridge (`log1p`) | E1 (Store + Calendar) | $\alpha = 1.0$ | **0.2962** | 4.41 | +4.48% |
| **E2_promo** | Ridge (`log1p`) | E2 (E1 + Promotion) | $\alpha = 1.0$ | **0.2894** | 4.34 | **+6.68%** *(Vượt KPI $\le 0.2946$)* |
| **E3_weather_holiday** | Ridge (`log1p`) | E3 (E2 + Env) | $\alpha = 1.0$ | **0.2631** | **3.75** | **+15.16%** *(Mô hình tốt nhất)* |
| **E4_ridge** | Ridge (`log1p`) | E3 (E2 + Env) | $\alpha = 10.0$ | **0.2646** | 3.79 | +14.67% |
| **E4_ols** | OLS (`log1p`) | E3 (E2 + Env) | Không phạt | **0.2639** | 3.75 | +14.90% |
| **E4_huber** | HuberRegressor | E3 (E2 + Env) | $\alpha=1.0, \epsilon=1.35$ | **0.2727** | **3.58** | +12.06% *(Tối ưu MAE)* |
| **E4_elastic_net** | ElasticNet | E3 (E2 + Env) | $\alpha=0.001, \ell_1=0.1$ | **0.2726** | 3.96 | +12.09% |
| **E4_lasso** | Lasso | E3 (E2 + Env) | $\alpha=0.0005$ | **0.2872** | 4.26 | +7.38% |

### 3.2. Độ ổn định Walk-Forward qua các Fold

| Fold | Khoảng Train | Khoảng Val | Baseline RMSLE | Ridge RMSLE | Mức cải thiện (%) |
|---|---|---|---|---|---|
| **Fold A (Q1/2025)** | Khởi đầu đến 2025-01-01 | Q1/2025 | 0.3037 | **0.2874** | **+5.37%** |
| **Fold B (Q2/2025)** | Khởi đầu đến 2025-04-01 | Q2/2025 | 0.2684 | **0.2367** | **+11.81%** |
| **Holdout chính (Q3/2025)** | Khởi đầu đến 2025-07-01 | Q3/2025 | 0.3101 | **0.2631** | **+15.16%** |

---

## 4. Kết quả Giai đoạn 3 — Mô hình phi tuyến dựa trên cây (Tree-based Models)

> Báo cáo phân tích chi tiết: [`reports/tree_models_report.md`](reports/tree_models_report.md)  
> Notebook thực nghiệm: [`notebooks/03_tree_models.ipynb`](notebooks/03_tree_models.ipynb)  
> Model Artifact: [`models/best_tree_model.joblib`](models/best_tree_model.joblib)

### 4.1. Bảng tiến trình thực nghiệm trên Primary Validation (Q3/2025)

| Thí nghiệm | Kiến trúc mô hình | Biến đổi Target | Siêu tham số | RMSLE | MAE (Tr. VND) | Cải thiện vs Baseline | Cải thiện vs Ridge GĐ2 |
|---|---|---|---|---|---|---|---|
| **T0_baseline** | Median (store, dow) | Không | Mốc chuẩn Giai đoạn 1 | **0.3101** | 4.31 | 0.0% | -17.86% |
| **T0_ridge** | Ridge Linear Model | `log1p` | $\alpha = 1.0$ (Mốc Giai đoạn 2) | **0.2631** | 3.75 | +15.16% | 0.0% |
| **T1_random_forest** | RandomForest | `log1p` | $n=150, \text{depth}=14$ | **0.2512** | 3.35 | +18.99% | +4.52% |
| **T2_extra_trees** | ExtraTrees | `log1p` | $n=150, \text{depth}=16$ | **0.2293** | 3.09 | +26.06% | +12.85% |
| **T3_hist_gradient_boosting** | HistGradientBoosting | `log1p` | $\text{iter}=200, \text{lr}=0.05, d=6$ | **0.2215** | 2.97 | +28.57% | +15.81% |
| **T4_lightgbm_default** | LightGBM (default) | `log1p` | $n=150, \text{lr}=0.05, \text{leaves}=31$ | **0.2231** | 3.02 | +28.06% | +15.20% |
| **T5_lightgbm_tuned** | **LightGBM (tuned)** | `log1p` | $n=300, \text{lr}=0.05, \text{leaves}=20, \text{sub}=0.8$ | **0.2163** | **2.92** | **+30.25%** | **+17.79%** |
| **T6_target_raw** | LightGBM (raw revenue) | Không (Raw) | Giống T5 nhưng fit trên doanh thu gốc | **0.2716** | 3.92 | +12.42% | -3.23% |

### 4.2. Độ ổn định Walk-Forward qua các Fold

| Fold | Khoảng Train | Khoảng Val | Baseline RMSLE | Ridge RMSLE | LightGBM Tuned RMSLE | vs Baseline (%) | vs Ridge (%) | LightGBM MAE |
|---|---|---|---|---|---|---|---|---|
| **Fold A (Q1/2025)** | Khởi đầu đến 2025-01-01 | Q1/2025 | 0.3037 | 0.2874 | **0.2904** | +4.38% | -1.04% | 4.00 |
| **Fold B (Q2/2025)** | Khởi đầu đến 2025-04-01 | Q2/2025 | 0.2684 | 0.2367 | **0.1982** | **+26.15%** | **+16.27%** | **2.43** |
| **Primary Val (Q3/2025)** | Khởi đầu đến 2025-07-01 | Q3/2025 | 0.3101 | 0.2631 | **0.2163** | **+30.25%** | **+17.79%** | **2.92** |

### 4.3. Các phát hiện kỹ thuật cốt lõi:
1. **Kiểm toán Ngoại lệ Đơn vị S12 (Tháng 03/2024):** 31 bản ghi bị ghi nhầm Nghìn VND (22.000 - 53.600) thay vì Triệu VND. Việc phát hiện và hiệu chỉnh chia 1.000 đã giúp cứu vãn Walk-Forward Fold A khỏi việc cây quyết định học ngưỡng chia ảo (MAE giảm từ 137.9 triệu VND xuống 4.00 triệu VND).
2. **Khắc phục triệt để tính phi tuyến mùa hè:**
   - Nhóm cửa hàng trường học (`university`) vào mùa hè (tháng 7-8): Baseline RMSLE = **0.5329** -> Ridge RMSLE = **0.3957** -> LightGBM RMSLE = **0.2520** (giảm hơn **52.7%** sai số). Cây tự động chia nhánh `(store_type == university) & (month in [7, 8])` chính xác tuyệt đối.
   - Ngày khuyến mãi (`is_promo`): Baseline = **0.4424** -> Ridge = **0.3423** -> LightGBM = **0.2525** (giảm 42,9% sai số).
3. **Độ quan trọng của Target Transformation:** Huấn luyện trên `log1p(revenue)` giúp khớp toán học hàm lỗi với metric RMSLE, cho kết quả vượt trội 0.0553 RMSLE so với huấn luyện trên doanh thu gốc (0.2163 vs 0.2716).

---

## 5. Cài đặt môi trường & Hướng dẫn thực thi

### 5.1. Thiết lập môi trường ảo (`venv`)
```powershell
# 1. Kích hoạt môi trường ảo (PowerShell)
.\venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện phụ thuộc (nếu chưa cài)
pip install -r requirements.txt
```

### 5.2. Chạy pipeline Giai đoạn 1 (Baseline)
```powershell
python src/experiments/run_baseline.py
```

### 5.3. Chạy pipeline Giai đoạn 2 (Mô hình tuyến tính)
```powershell
python src/experiments/run_linear_models.py
```

### 5.4. Chạy pipeline Giai đoạn 3 (Tree-based Models)
```powershell
python src/experiments/run_tree_models.py
```

### 5.5. Chạy toàn bộ Unit Tests
```powershell
python -m unittest discover tests
```

### 5.6. Mở và Xem Notebooks Tương tác
- [`notebooks/01_baseline_reproduction.ipynb`](notebooks/01_baseline_reproduction.ipynb): Tái lập Baseline & Phân tích lỗi sơ bộ.
- [`notebooks/02_linear_models.ipynb`](notebooks/02_linear_models.ipynb): Toàn bộ quy trình Giai đoạn 2 với biểu đồ so sánh mô hình tuyến tính và phân tích hệ số.
- [`notebooks/03_tree_models.ipynb`](notebooks/03_tree_models.ipynb): Toàn bộ quy trình Giai đoạn 3 với biểu đồ so sánh Random Forest, ExtraTrees, HistGBR, LightGBM, phân tích độ quan trọng và chẩn đoán residual.

