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
│   └── best_linear_model.joblib # Pipeline mô hình tuyến tính Ridge tốt nhất (Giai đoạn 2)
├── notebooks/
│   ├── 01_baseline_reproduction.ipynb  # Notebook tái lập và phân tích baseline (GĐ 1)
│   └── 02_linear_models.ipynb          # Notebook thực nghiệm & phân tích hệ số tuyến tính (GĐ 2)
├── reports/
│   ├── baseline_report.md              # Báo cáo chi tiết Giai đoạn 1
│   ├── baseline_val_predictions.csv    # Kết quả dự báo baseline trên Q3/2025
│   ├── linear_models_report.md         # Báo cáo chi tiết Giai đoạn 2 (Mô hình tuyến tính)
│   ├── linear_val_predictions.csv      # Kết quả dự báo và residual của Ridge trên Q3/2025
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
│   │   └── builder.py       # Pipeline tổng hợp và registry các tầng feature (E1, E2, E3)
│   ├── evaluation/
│   │   ├── metrics.py       # RMSLE, MAE, RMSE, log residual
│   │   └── backtest.py      # Walk-forward backtesting và phân tích nhóm
│   ├── models/
│   │   ├── baseline.py      # WeekdayMedianBaseline với 2 tầng fallback
│   │   └── linear.py        # LogTransformedLinearModel (Ridge, Huber, ElasticNet, OLS)
│   └── experiments/
│       ├── run_baseline.py      # Pipeline chính thực thi Giai đoạn 1
│       └── run_linear_models.py  # Pipeline chính thực thi Giai đoạn 2
├── tests/
│   ├── test_metrics.py       # Unit test cho hàm metric
│   ├── test_baseline.py      # Unit test cho baseline model
│   ├── test_splits.py        # Unit test cho hợp đồng dữ liệu và splits
│   ├── test_features.py      # Unit test cho feature engineering & join an toàn
│   └── test_linear_models.py # Unit test cho pipeline mô hình tuyến tính
├── scripts/
│   ├── build_notebook.py         # Script dựng notebook 01
│   └── build_linear_notebook.py  # Script dựng notebook 02
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

### 3.3. Các phát hiện kinh doanh & Khả năng giải thích (Interpretability)
1. **Hiệu ứng Khuyến mãi (`is_promo`):** Thúc đẩy doanh thu tăng trung bình **+45.2%**, giải quyết triệt để vấn đề dự báo thấp của baseline.
2. **Khắc phục lỗi nhóm yếu thế:**
   - Ngày có khuyến mãi: RMSLE giảm từ **0.4424 xuống 0.3423** (giảm 22,6%).
   - Cửa hàng trường học nghỉ hè: RMSLE giảm từ **0.5329 xuống 0.3957** (giảm 25,7%).
   - Cửa hàng mới mở (< 100 mẫu): RMSLE giảm từ **0.4714 xuống 0.3439** (giảm 27,0%).

---

## 4. Cài đặt môi trường & Hướng dẫn thực thi

### 4.1. Thiết lập môi trường ảo (`venv`)
```powershell
# 1. Kích hoạt môi trường ảo (PowerShell)
.\venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện phụ thuộc (nếu chưa cài)
pip install -r requirements.txt
```

### 4.2. Chạy pipeline Giai đoạn 1 (Baseline)
```powershell
python src/experiments/run_baseline.py
```

### 4.3. Chạy pipeline Giai đoạn 2 (Mô hình tuyến tính)
```powershell
python src/experiments/run_linear_models.py
```

### 4.4. Chạy toàn bộ Unit Tests
```powershell
python -m unittest discover tests
```

### 4.5. Mở và Xem Notebooks Tương tác
- [`notebooks/01_baseline_reproduction.ipynb`](notebooks/01_baseline_reproduction.ipynb): Tái lập Baseline & Phân tích lỗi sơ bộ.
- [`notebooks/02_linear_models.ipynb`](notebooks/02_linear_models.ipynb): Toàn bộ quy trình Giai đoạn 2 với biểu đồ so sánh mô hình, phân tích hệ số và chẩn đoán sai số.
