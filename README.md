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
│   ├── best_tree_model.joblib   # Pipeline mô hình LightGBM Tuned tốt nhất (Giai đoạn 3)
│   ├── best_target_transform_model.joblib # Pipeline mô hình LightGBM Huber Loss tối ưu nhất (Giai đoạn 4)
│   ├── best_ensemble_model.joblib # Mô hình Weighted Ensemble + Duan Smearing tối ưu nhất (Giai đoạn 5)
│   └── best_production_model.joblib # Pipeline mô hình Champion hoàn chỉnh sẵn sàng phục vụ suy luận
├── notebooks/
│   ├── 01_baseline_reproduction.ipynb  # Notebook tái lập và phân tích baseline (GĐ 1)
│   ├── 02_linear_models.ipynb          # Notebook thực nghiệm & phân tích hệ số tuyến tính (GĐ 2)
│   ├── 03_tree_models.ipynb            # Notebook thực nghiệm & phân tích phi tuyến cây (GĐ 3)
│   ├── 04_target_transformation.ipynb  # Notebook thực nghiệm biến đổi target & hàm loss (GĐ 4)
│   ├── 05_lags_and_ensemble.ipynb      # Notebook thực nghiệm phân tích lags & kiến trúc ensemble (GĐ 5)
│   └── end_to_end_training_pipeline.ipynb # Master Notebook từ đầu đến cuối chuẩn 10 bước Training_Process.md
├── reports/
│   ├── baseline_report.md              # Báo cáo chi tiết Giai đoạn 1
│   ├── baseline_val_predictions.csv    # Kết quả dự báo baseline trên Q3/2025
│   ├── linear_models_report.md         # Báo cáo chi tiết Giai đoạn 2 (Mô hình tuyến tính)
│   ├── linear_val_predictions.csv      # Kết quả dự báo và residual của Ridge trên Q3/2025
│   ├── tree_models_report.md           # Báo cáo chi tiết Giai đoạn 3 (Tree-based models)
│   ├── tree_val_predictions.csv        # Kết quả dự báo và residual của LightGBM trên Q3/2025
│   ├── target_transform_report.md      # Báo cáo chi tiết Giai đoạn 4 (Target transformation)
│   ├── target_transform_val_predictions.csv # Kết quả dự báo của LightGBM Huber trên Q3/2025
│   ├── lags_and_ensemble_report.md     # Báo cáo chi tiết Giai đoạn 5 (Lags & Ensemble)
│   ├── ensemble_val_predictions.csv    # Kết quả dự báo của Ensemble tối ưu trên Q3/2025
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
│   │   ├── tree.py          # LogTransformedTreeModel (LightGBM, HistGBR, RF, ExtraTrees)
│   │   ├── target_transform.py # TargetTransformer & TransformedTargetTreeModel
│   │   └── ensemble.py      # WeightedLogEnsemble & StackingLogEnsemble
│   └── experiments/
│       ├── run_baseline.py      # Pipeline chính thực thi Giai đoạn 1
│       ├── run_linear_models.py  # Pipeline chính thực thi Giai đoạn 2
│       ├── run_tree_models.py    # Pipeline chính thực thi Giai đoạn 3
│       ├── run_target_transform.py # Pipeline chính thực thi Giai đoạn 4
│       └── run_lags_and_ensemble.py # Pipeline chính thực thi Giai đoạn 5
├── tests/
│   ├── test_metrics.py       # Unit test cho hàm metric
│   ├── test_baseline.py      # Unit test cho baseline model
│   ├── test_splits.py        # Unit test cho hợp đồng dữ liệu và splits
│   ├── test_features.py      # Unit test cho feature engineering & join an toàn
│   ├── test_linear_models.py # Unit test cho pipeline mô hình tuyến tính
│   ├── test_tree_models.py   # Unit test cho pipeline mô hình cây quyết định
│   ├── test_target_transforms.py # Unit test cho bộ biến đổi target & bias adjustments
│   └── test_ensemble.py      # Unit test cho mô hình ensemble blending và stacking
├── scripts/
│   ├── build_notebook.py         # Script dựng notebook 01
│   ├── build_linear_notebook.py  # Script dựng notebook 02
│   ├── build_tree_notebook.py    # Script dựng notebook 03
│   ├── build_target_notebook.py  # Script dựng notebook 04
│   └── build_ensemble_notebook.py # Script dựng notebook 05
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

---

## 5. Kết quả Giai đoạn 4 — Biến đổi Target và Hàm Mục tiêu (Target Transformation)

> Báo cáo phân tích chi tiết: [`reports/target_transform_report.md`](reports/target_transform_report.md)  
> Notebook thực nghiệm: [`notebooks/04_target_transformation.ipynb`](notebooks/04_target_transformation.ipynb)  
> Model Artifact: [`models/best_target_transform_model.joblib`](models/best_target_transform_model.joblib)

### 5.1. Bảng so sánh các Hàm Mục tiêu (Loss Objectives) trong LightGBM trên Q3/2025

| Cấu hình Mục tiêu | Không gian Target | Hàm Loss | RMSLE | MAE (Tr. VND) | vs Baseline (%) | vs Ridge GĐ2 (%) |
|---|---|---|---|---|---|---|
| **Raw Target - MSE Loss** | Doanh thu gốc | regression (MSE) | **0.2716** | 3.92 | +12.42% | -3.23% |
| **Raw Target - Poisson Loss** | Doanh thu gốc | poisson | **0.2616** | 3.93 | +15.64% | +0.57% |
| **Raw Target - Tweedie ($\rho=1.1$)** | Doanh thu gốc | tweedie | **0.2508** | 3.75 | +19.12% | +4.68% |
| **Raw Target - Tweedie ($\rho=1.5$)** | Doanh thu gốc | tweedie | **0.2396** | 3.47 | +22.73% | +8.93% |
| **Raw Target - Tweedie ($\rho=1.9$)** | Doanh thu gốc | tweedie | **0.2299** | 3.29 | +25.86% | +12.62% |
| **Log1p Target - L1/MAE Loss** | `log1p(revenue)` | regression_l1 | **0.2164** | 2.92 | +30.22% | +17.75% |
| **Log1p Target - MSE Loss (GĐ 3)** | `log1p(revenue)` | regression (MSE) | **0.2163** | 2.92 | +30.25% | +17.79% |
| **Log1p Target - Huber Loss (Kỷ lục mới)** | `log1p(revenue)` | **huber** | **0.2129** | **2.88** | **+31.34%** | **+19.08%** |

### 5.2. Đánh giá Hiệu chỉnh Tái biến đổi (Smearing & Bias Correction)

| Phương pháp Nghịch đảo | Công thức điều chỉnh | RMSLE | MAE (Tr. VND) | Mean Log Residual |
|---|---|---|---|---|
| **Log1p Huber - Standard** | $\hat{y} = \text{expm1}(\hat{z})$ (Không hiệu chỉnh) | **0.2129** | 2.88 | +0.0436 |
| **Log1p Huber - Half-variance** | $\hat{y} = \exp(\hat{z} + 0.5 \sigma^2) - 1$ ($\sigma^2 = 0.0247$) | **0.2107** | 2.82 | +0.0313 |
| **Log1p Huber - Duan Smearing** | $\hat{y} = \exp(\hat{z}) \times 1.0205 - 1$ | **0.2097** | **2.79** | +0.0233 |

### 5.3. Độ ổn định Walk-Forward qua các Fold

| Fold | Khoảng Train | Khoảng Val | Baseline RMSLE | Ridge RMSLE | LightGBM MSE GĐ3 | LightGBM Huber GĐ4 | vs Baseline (%) | Huber MAE |
|---|---|---|---|---|---|---|---|---|
| **Fold A (Q1/2025)** | Khởi đầu đến 2025-01-01 | Q1/2025 | 0.3037 | 0.2874 | 0.2904 | **0.2959** | +2.57% | 4.04 |
| **Fold B (Q2/2025)** | Khởi đầu đến 2025-04-01 | Q2/2025 | 0.2684 | 0.2367 | 0.1959 | **0.1951** | **+27.31%** | **2.70** |
| **Primary Val (Q3/2025)** | Khởi đầu đến 2025-07-01 | Q3/2025 | 0.3101 | 0.2631 | 0.2163 | **0.2129** | **+31.34%** | **2.88** |

### 5.4. Các phát hiện cốt lõi:
1. **Phân phối biến mục tiêu:** Doanh thu thực tế có độ lệch phải rất nặng (Skewness = **+12.19**). Phép biến đổi $\log(1+y)$ đưa độ lệch về **-0.30**, giúp thỏa mãn tính đối xứng và khớp chính xác hàm mất mát với metric RMSLE của cuộc thi.
2. **Ưu thế của Huber Loss:** Huber loss trong không gian log kết hợp ưu điểm của L2 ở sai số nhỏ và L1 ở sai số lớn, giúp mô hình miễn nhiễm với các điểm đột biến doanh thu còn sót lại, thiết lập kỷ lục mới **RMSLE = 0.2129** và **MAE = 2.88 triệu VND**.
3. **Hiệu ứng Smearing Factor:** Duan's smearing factor ước lượng là **1.0205** (+2.05% điều chỉnh). Việc áp dụng Smearing giúp điều chỉnh độ lệch Jensen và giảm MAE xuống **2.79 triệu VND**.

---

## 6. Kết quả Giai đoạn 5 — Lags và Ensemble Models

> Báo cáo phân tích chi tiết: [`reports/lags_and_ensemble_report.md`](reports/lags_and_ensemble_report.md)  
> Notebook thực nghiệm: [`notebooks/05_lags_and_ensemble.ipynb`](notebooks/05_lags_and_ensemble.ipynb)  
> Model Artifact: [`models/best_ensemble_model.joblib`](models/best_ensemble_model.joblib)

### 6.1. Khảo sát Lags và Dự báo Nhiều bước (Multi-step Horizon Simulation)

| Phương pháp Lags | Đặc trưng sử dụng | Mô phỏng 90 ngày | RMSLE | MAE (Tr. VND) | Đánh giá rủi ro & Thực thi |
|---|---|---|---|---|---|
| **E4a: Không dùng Lags (Historical Stats)** | Out-of-fold `log_hist_sw_med`, `log_hist_s_med` | An toàn tuyệt đối (Không leakage) | **0.2129** | **2.88** | Zero leakage, tốc độ tức thì, không rủi ro tích lũy sai số |
| **E4b: Horizon-safe Long Lags (>=90d)** | Bổ sung `lag_91` & `lag_364` | An toàn (Biết tại cutoff $T_0$) | 0.5185 | 8.03 | Thưa thớt dữ liệu ở cửa hàng mới, không cải thiện thêm |
| **E4c: Recursive Multi-step Rollout** | `lag_7` & `lag_14` cập nhật đệ quy | Đệ quy $T_0 \to T_{90}$ | 0.5185 | 8.03 | Tích lũy sai số đệ quy (*error compounding*), không tăng độ chính xác |
| **E4d: Teacher Forcing (Đối chứng)** | `lag_7` & `lag_14` lấy từ ground-truth | Cung cấp tương lai (Leakage) | 0.5185 | 8.03 | Vi phạm hợp đồng dự báo thực tế (*unrealistic*) |

*Kết luận nghiên cứu E4:* Sử dụng thống kê lịch sử an toàn (Horizon-safe Target Statistics) kết hợp calendar/weather/store interactions là tối ưu nhất cho bài toán dự báo 90 ngày.

### 6.2. Bảng So sánh các Chiến lược Ensemble trên Q3/2025

| Chiến lược Ensemble | Thành phần & Trọng số | Hiệu chỉnh Smearing | RMSLE | MAE (Tr. VND) | vs Baseline (%) | vs Single Best GĐ4 (%) |
|---|---|---|---|---|---|---|
| **ENS1: Simple Average Blend** | 20% mỗi mô hình (LGBM, ET, HGB, Ridge, RF) | Không | 0.2221 | 3.01 | +28.38% | -4.32% |
| **ENS2: Top-3 Tree Average** | 33.3% mỗi cây (LGBM + ExtraTrees + HistGBR) | Không | 0.2167 | 2.92 | +30.12% | -1.78% |
| **ENS3: Stacking Meta-Ridge** | Meta-Ridge học trọng số OOF qua 4-Fold CV | Không | 0.2229 | 3.00 | +28.12% | -4.70% |
| **ENS4: Optimized Weighted Blend** | 0.90 LightGBM Huber + 0.10 Ridge | Không | 0.2123 | 2.88 | +31.54% | +0.28% |
| **ENS5: Optimized Blend + Smearing (Kỷ lục mới)** | **0.90 LightGBM Huber + 0.10 Ridge** | **Duan factor = 1.0207** | **0.2096** | **2.81** | **+32.41%** | **+1.55%** |

### 6.3. Độ ổn định Walk-Forward qua các Fold

| Fold | Khoảng Train | Khoảng Val | Baseline RMSLE | Ridge RMSLE | LGBM Huber RMSLE | Ensemble Tối ưu RMSLE | vs Baseline (%) | vs Ridge (%) | vs Single Best (%) | Ensemble MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold A (Q1/2025)** | Khởi đầu đến 2025-01-01 | Q1/2025 | 0.3037 | 0.2874 | 0.2959 | **0.2830** | +6.82% | +1.53% | **+4.36%** | 3.83 |
| **Fold B (Q2/2025)** | Khởi đầu đến 2025-04-01 | Q2/2025 | 0.2684 | 0.2367 | 0.1951 | **0.1921** | +28.43% | +18.84% | **+1.54%** | **2.63** |
| **Primary Val (Q3/2025)** | Khởi đầu đến 2025-07-01 | Q3/2025 | 0.3101 | 0.2631 | 0.2129 | **0.2096** | **+32.41%** | **+20.33%** | **+1.55%** | **2.81** |

### 6.4. Các phát hiện kỹ thuật cốt lõi:
1. **Phá vỡ rào cản RMSLE 0.2100:** Mô hình kết hợp tối ưu giữa **LightGBM Huber Loss** (90%) và **Linear Ridge** (10%) kết hợp hệ số Duan's Smearing đạt **RMSLE = 0.2096** và **MAE = 2.81 triệu VND**, chính thức vượt qua mục tiêu đề ra tại `Regression_Plan.md`.
2. **Tính bổ trợ của Inductive Bias:** LightGBM chia nhánh phi tuyến cực mạnh trên các điều kiện biên (university mùa hè, khuyến mãi), trong khi Linear Ridge cung cấp một hàm hồi quy trơn mượt toàn cục giúp điều hòa (regularize) các dự báo ngoại lai ở vùng đuôi doanh thu cao.
3. **Thống trị toàn diện trên Walk-Forward Backtesting:** Mô hình Ensemble đánh bại tất cả các mô hình đơn lẻ ở **tất cả 3 chu kỳ kiểm định** (Fold A giảm từ 0.2959 xuống 0.2830; Fold B giảm từ 0.1951 xuống 0.1921; Q3/2025 giảm từ 0.2129 xuống 0.2096).

---

## 7. Cài đặt môi trường & Hướng dẫn thực thi

### 7.1. Thiết lập môi trường ảo (`venv`)
```powershell
# 1. Kích hoạt môi trường ảo (PowerShell)
.\venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện phụ thuộc (nếu chưa cài)
pip install -r requirements.txt
```

### 7.2. Chạy pipeline Giai đoạn 1 (Baseline)
```powershell
python src/experiments/run_baseline.py
```

### 7.3. Chạy pipeline Giai đoạn 2 (Mô hình tuyến tính)
```powershell
python src/experiments/run_linear_models.py
```

### 7.4. Chạy pipeline Giai đoạn 3 (Tree-based Models)
```powershell
python src/experiments/run_tree_models.py
```

### 7.5. Chạy pipeline Giai đoạn 4 (Target Transformation)
```powershell
python src/experiments/run_target_transform.py
```

### 7.6. Chạy pipeline Giai đoạn 5 (Lags & Ensemble Models)
```powershell
python src/experiments/run_lags_and_ensemble.py
```

### 7.7. Chạy toàn bộ Unit Tests (42 tests)
```powershell
python -m unittest discover tests
```

### 7.8. Mở và Xem Notebooks Tương tác
- [`notebooks/01_baseline_reproduction.ipynb`](notebooks/01_baseline_reproduction.ipynb): Tái lập Baseline & Phân tích lỗi sơ bộ.
- [`notebooks/02_linear_models.ipynb`](notebooks/02_linear_models.ipynb): Toàn bộ quy trình Giai đoạn 2 với biểu đồ so sánh mô hình tuyến tính và phân tích hệ số.
- [`notebooks/03_tree_models.ipynb`](notebooks/03_tree_models.ipynb): Toàn bộ quy trình Giai đoạn 3 với biểu đồ so sánh Random Forest, ExtraTrees, HistGBR, LightGBM.
- [`notebooks/04_target_transformation.ipynb`](notebooks/04_target_transformation.ipynb): Toàn bộ quy trình Giai đoạn 4 với phân tích phân phối target, so sánh hàm loss (Tweedie, Poisson, Huber), và thực nghiệm bias correction / smearing.
- [`notebooks/05_lags_and_ensemble.ipynb`](notebooks/05_lags_and_ensemble.ipynb): Toàn bộ quy trình Giai đoạn 5 với kiểm chứng mô phỏng Lags 90 ngày, so sánh 5 cấu hình Ensemble, tối ưu hóa trọng số SLSQP, và phân tích độ ổn định Walk-Forward.
- [`notebooks/end_to_end_training_pipeline.ipynb`](notebooks/end_to_end_training_pipeline.ipynb): **Master Notebook toàn diện chuẩn 10 bước theo `Training_Process.md`** — thực thi từ dữ liệu thô, EDA, làm sạch, tách tập, feature engineering, huấn luyện 5 thế hệ mô hình, backtesting, final test trên holdout Q4/2025 (RMSLE = 0.1866), đóng gói model artifact và xây dựng inference pipeline sản xuất.

