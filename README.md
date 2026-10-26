# Cafe Revenue Regression — Dự Báo Doanh Thu Cửa Hàng Cà Phê

Dự án Machine Learning dự báo doanh thu hàng ngày theo từng cửa hàng cà phê (*panel time-series regression*).

- **Biến mục tiêu:** `revenue` (triệu VND/ngày, không âm)
- **Metric đánh giá:** RMSLE (Root Mean Squared Logarithmic Error)
- **Kế hoạch tổng thể:** xem chi tiết tại [`Regression_Plan.md`](Regression_Plan.md)

---

## Executive Summary

Dự án đã chuẩn hóa 100% chiến lược phân chia dữ liệu đồng bộ tuyệt đối với Notebook cơ sở của Ban tổ chức ([`data/raw/starter_cafe.ipynb`](data/raw/starter_cafe.ipynb)):
- **Tập Train:** Dữ liệu trước 01/10/2025 (`date < 2025-10-01`), bao gồm **22.668 dòng sạch** sau khi loại 132 bản ghi trùng lặp (21 tháng lịch sử: 01/01/2024 – 30/09/2025).
- **Tập Primary Validation:** Quý 4/2025 (`01/10/2025 <= date <= 31/12/2025`), bao gồm **3.832 dòng sạch** làm mốc đánh giá chuẩn mực.
- **Tập Test chính thức:*### Bảng thành tích lũy tiến qua các giai đoạn mô hình trên Primary Validation (Q4/2025)

| Giai đoạn | Thế hệ mô hình | Cấu hình & Kỹ thuật cốt lõi | RMSLE | MAE (Tr. VND) | Cải thiện vs Baseline | Trạng thái kỹ thuật |
|---|---|---|---|---|---|---|
| **GĐ 0** | **Làm sạch có kiểm chứng** | Khử trùng lặp (132), sửa S12 (/1000), cờ 75 đột biến, chuẩn hóa thời tiết (°F->°C, lưới 1.642 dòng) & stores, ablation C0–C5 | — | — | — | **Nền tảng dữ liệu sạch chuẩn mực** |
| **GĐ 1** | **Baseline Median** | Lookup trung vị `(store_id, weekday)` + 2 tầng fallback | **0.2391** *(0.2387 raw)* | 3.57 | 0.0% | Mốc chuẩn đối chiếu ban đầu |
| **GĐ 2** | **Linear Model** | Ridge Regression (`log1p` target + E3 features: Lịch, Store, Promo, Weather, Holiday) | **0.1978** | 2.94 | **+17.27%** | Vượt baseline rõ rệt |
| **GĐ 3** | **Tree Model** | LightGBM Tuned (max_depth=5, 20 leaves, E3 features + target statistics OOF) | **0.1667** | 2.49 | **+30.28%** | Bắt phi tương tác phi tuyến |
| **GĐ 4** | **Target Transform** | LightGBM Huber Loss + Duan's Smearing Factor ($s = 1.0220$) | **0.1604** | 2.34 | **+32.91%** | Tối ưu hóa hàm loss & khử lệch Jensen |
| **GĐ 5** | **Champion Ensemble** | **Weighted Blend (91% LGBM Huber + 9% Linear Ridge) + Duan Smearing ($s = 1.0222$)** | **0.1603** | **2.33** | **+32.98%** | **Mô hình Tối ưu Nhất Toàn Dự án** |

---

## 1. Cấu trúc thư mục

```text
cafe-revenue-regression/
├── data/
│   ├── raw/                 # train.csv, stores.csv, weather.csv, holidays.csv
│   ├── clean/               # train_clean, stores_clean, weather_clean, holidays_daily (.parquet & .csv)
│   └── processed/           # train.csv, val.csv đã xử lý trùng lặp & predictions
├── regression_splits/       # Thư mục split chuẩn (train.csv, val.csv) đồng bộ starter_cafe.ipynb
├── models/
│   ├── baseline_median.json # Model artifact lưu các bảng lookup trung vị (Giai đoạn 1)
│   ├── best_linear_model.joblib # Pipeline mô hình tuyến tính Ridge tốt nhất (Giai đoạn 2)
│   ├── best_tree_model.joblib   # Pipeline mô hình LightGBM Tuned tốt nhất (Giai đoạn 3)
│   ├── best_target_transform_model.joblib # Pipeline mô hình LightGBM Huber Loss tối ưu nhất (Giai đoạn 4)
│   ├── best_ensemble_model.joblib # Mô hình Weighted Ensemble + Duan Smearing tối ưu nhất (Giai đoạn 5)
│   └── best_production_model.joblib # Pipeline mô hình Champion hoàn chỉnh sẵn sàng phục vụ suy luận
├── notebooks/
│   ├── 00_data_cleaning_and_verification.ipynb # Notebook Giai đoạn 0: Làm sạch có kiểm chứng & kiểm thử hợp đồng
│   ├── 01_baseline_reproduction.ipynb  # Notebook tái lập và phân tích baseline (GĐ 1)
│   ├── 02_linear_models.ipynb          # Notebook thực nghiệm & phân tích hệ số tuyến tính (GĐ 2)
│   ├── 03_tree_models.ipynb            # Notebook thực nghiệm & phân tích phi tuyến cây (GĐ 3)
│   ├── 04_target_transformation.ipynb  # Notebook thực nghiệm biến đổi target & hàm loss (GĐ 4)
│   ├── 05_lags_and_ensemble.ipynb      # Notebook thực nghiệm phân tích lags & kiến trúc ensemble (GĐ 5)
│   ├── 07_test.ipynb                   # Thực nghiệm xử lý ngoại lệ đa quy mô (1000x, 10x, 0.1x) & Champion Model
│   └── end_to_end_training_pipeline.ipynb # Master Notebook từ đầu đến cuối chuẩn 10 bước
├── reports/
│   ├── data_quality_report.md          # Báo cáo chi tiết Giai đoạn 0 (Chất lượng dữ liệu & Ablation C0-C5)
│   ├── outliers_flagged.csv            # Sổ đăng ký 110 dòng ngoại lệ & đột biến đã gắn cờ
│   ├── data_cleaning_ablation.csv      # Bảng số liệu chi tiết nghiệm thu Ablation C0-C5
│   ├── baseline_report.md              # Báo cáo chi tiết Giai đoạn 1
│   ├── baseline_val_predictions.csv    # Kết quả dự báo baseline trên Q4/2025
│   ├── linear_models_report.md         # Báo cáo chi tiết Giai đoạn 2 (Mô hình tuyến tính)
│   ├── linear_val_predictions.csv      # Kết quả dự báo và residual của Ridge trên Q4/2025
│   ├── tree_models_report.md           # Báo cáo chi tiết Giai đoạn 3 (Tree-based models)
│   ├── tree_val_predictions.csv        # Kết quả dự báo và residual của LightGBM trên Q4/2025
│   ├── target_transform_report.md      # Báo cáo chi tiết Giai đoạn 4 (Target transformation)
│   ├── target_transform_val_predictions.csv # Kết quả dự báo của LightGBM Huber trên Q4/2025
│   ├── lags_and_ensemble_report.md     # Báo cáo chi tiết Giai đoạn 5 (Lags & Ensemble)
│   ├── ensemble_val_predictions.csv    # Kết quả dự báo của Ensemble tối ưu trên Q4/2025
│   └── validation_results.csv          # Bảng tổng hợp điểm số qua các fold và mô hình
├── src/
│   ├── config.py            # Cấu hình đường dẫn, thời gian split và hằng số
│   ├── data/
│   │   ├── clean.py         # Pipeline làm sạch, sửa lỗi đơn vị, phát hiện đột biến & contracts (GĐ 0)
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
│       ├── run_data_cleaning.py # Pipeline chính thực thi Giai đoạn 0 (Làm sạch có kiểm chứng)
│       ├── run_baseline.py      # Pipeline chính thực thi Giai đoạn 1
│       ├── run_linear_models.py  # Pipeline chính thực thi Giai đoạn 2
│       ├── run_tree_models.py    # Pipeline chính thực thi Giai đoạn 3
│       ├── run_target_transform.py # Pipeline chính thực thi Giai đoạn 4
│       └── run_lags_and_ensemble.py # Pipeline chính thực thi Giai đoạn 5
├── tests/
│   ├── test_data_contracts.py# Unit test tự động kiểm thử hợp đồng dữ liệu Phase 0
│   ├── test_metrics.py       # Unit test cho hàm metric
│   ├── test_baseline.py      # Unit test cho baseline model
│   ├── test_splits.py        # Unit test cho hợp đồng dữ liệu và splits
│   ├── test_features.py      # Unit test cho feature engineering & join an toàn
│   ├── test_linear_models.py # Unit test cho pipeline mô hình tuyến tính
│   ├── test_tree_models.py   # Unit test cho pipeline mô hình cây quyết định
│   ├── test_target_transforms.py # Unit test cho bộ biến đổi target & bias adjustments
│   └── test_ensemble.py      # Unit test cho mô hình ensemble blending và stacking
├── scripts/
│   ├── build_00_data_cleaning_notebook.py # Script dựng notebook 00
│   ├── build_notebook.py         # Script dựng notebook 01
│   ├── build_linear_notebook.py  # Script dựng notebook 02
│   ├── build_tree_notebook.py    # Script dựng notebook 03
│   ├── build_target_notebook.py  # Script dựng notebook 04
│   ├── build_ensemble_notebook.py # Script dựng notebook 05
│   ├── build_07_test_notebook.py # Script dựng và chạy thực nghiệm Notebook 07
│   └── build_end_to_end_notebook.py # Script dựng Master Pipeline Notebook
├── requirements.txt         # Danh sách thư viện phụ thuộc (pinned versions)
├── venv/                    # Môi trường ảo Python (cục bộ)
└── README.md
```

---

## 1.1. Kết quả Giai đoạn 0 — Làm sạch có kiểm chứng (Verified Data Cleaning)

Giai đoạn 0 thiết lập nền tảng dữ liệu sạch tuân thủ nghiêm ngặt **Kiến trúc ba lớp dữ liệu**:
- **Khử trùng lặp:** 132 bản ghi trùng lặp hoàn toàn được loại bỏ, đưa `train.csv` về đúng **26.500 dòng sạch**.
- **Sửa lỗi đơn vị S12:** 30 bản ghi tháng 03/2024 của S12 bị ghi nhầm theo Nghìn VND (> 10.000 - 53.600) được chia cho 1.000, đưa trung vị về 29.17 triệu VND (khớp tự nhiên với tháng 2 và tháng 4, không xóa dòng).
- **Phân loại 75 đột biến doanh thu:** 50 đột biến cao (~10x) và 25 đột biến thấp (~0.1x) được gắn cờ và loại khỏi tập fit. 6 đột biến nằm trong Q4/2025 tạo thành **sàn nhiễu ~0.010 - 0.012 RMSLE**.
- **Hiệu chuẩn thời tiết Fahrenheit:** 22 ngày liên tiếp tại TP.HCM (10/06–01/07/2025) đo bằng °F (84–89.2 °F) được chuyển về °C (27.9–31.8 °C). Dựng lưới thời tiết đủ 1.642 dòng, điền thiếu an toàn.
- **Store & Holiday:** Impute `area_m2` cho S19 và S24 theo tỷ lệ $2.607 \text{ m}^2/\text{ghế}$. Tổng hợp 53 ngày lễ duy nhất và tạo cờ Tết chuyên biệt.
- **Ablation C0–C5:** Xác nhận cấu hình **C4** tối ưu nhất, ngăn chặn tình trạng nổ tung sai số tại Fold A Tết (LightGBM RMSLE giảm từ 0.7164 xuống 0.3340).
- **Artifacts:**
  * Báo cáo toàn diện: [`reports/data_quality_report.md`](reports/data_quality_report.md)
  * Sổ đăng ký ngoại lệ: [`reports/outliers_flagged.csv`](reports/outliers_flagged.csv)
  * Số liệu ablation: [`reports/data_cleaning_ablation.csv`](reports/data_cleaning_ablation.csv)
  * Notebook thực nghiệm: [`notebooks/00_data_cleaning_and_verification.ipynb`](notebooks/00_data_cleaning_and_verification.ipynb)
  * Bộ test hợp đồng tự động: [`tests/test_data_contracts.py`](tests/test_data_contracts.py) (6/6 tests passed)

---

## 2. Chiến lược phân chia dữ liệu & Baseline (Giai đoạn 1)

> Báo cáo phân tích chi tiết: [`reports/baseline_report.md`](reports/baseline_report.md)  
> Notebook thực nghiệm: [`notebooks/01_baseline_reproduction.ipynb`](notebooks/01_baseline_reproduction.ipynb)  
> Model Artifact: [`models/baseline_median.json`](models/baseline_median.json)

Để đảm bảo tính nhất quán tuyệt đối và đối chiếu chuẩn xác với Notebook cơ sở của ban tổ chức ([`data/raw/starter_cafe.ipynb`](data/raw/starter_cafe.ipynb)), phân vai dữ liệu được quy định thống nhất như sau:

- **Tập Train:** `date < 2025-10-01` — **22.668 dòng sạch** (sau khi loại 132 dòng trùng lặp và xử lý ngoại lai đơn vị), bao phủ 21 tháng lịch sử từ 01/01/2024 đến 30/09/2025.
- **Tập Primary Validation:** `2025-10-01 <= date < 2026-01-01` (Quý 4/2025) — **3.832 dòng sạch**, dùng làm mốc so sánh chính thức.
  - Baseline Median `(store_id, weekday)` đạt **RMSLE = 0.2387** trên dữ liệu thô (khớp hoàn toàn số liệu trong `starter_cafe.ipynb`).
  - Baseline đạt **RMSLE = 0.2391** | **MAE = 3.57 triệu VND** trên dữ liệu sạch sau khử trùng.
- **Tập Test chính thức:** `01/01/2026 <= date <= 31/03/2026` (Quý 1/2026) do giảng viên công bố trên Kaggle (không tách tệp test nội bộ từ `train.csv`).

---

## 3. Kết quả Giai đoạn 2 — Mô hình tuyến tính (Linear Regression Models)

> Báo cáo phân tích chi tiết: [`reports/linear_models_report.md`](reports/linear_models_report.md)  
> Notebook thực nghiệm: [`notebooks/02_linear_models.ipynb`](notebooks/02_linear_models.ipynb)  
> Model Artifact: [`models/best_linear_model.joblib`](models/best_linear_model.joblib)

### 3.1. Bảng tiến trình thực nghiệm trên Primary Validation (Q4/2025)

| Thí nghiệm | Họ mô hình | Tầng Feature | Siêu tham số | RMSLE | MAE (Tr. VND) | Cải thiện so với Baseline |
|---|---|---|---|---|---|---|
| **E0_baseline** | MedianBaseline | store_weekday_median | Không | **0.2391** | 3.57 | 0.0% (Mốc chuẩn) |
| **E1_calendar_store** | Ridge (`log1p`) | E1 (Store + Calendar) | $\alpha = 1.0$ | **0.2508** | 3.80 | -4.89% |
| **E2_promo** | Ridge (`log1p`) | E2 (E1 + Promotion) | $\alpha = 1.0$ | **0.2268** | 3.42 | **+5.14%** |
| **E3_weather_holiday** | Ridge (`log1p`) | E3 (E2 + Env) | $\alpha = 1.0$ | **0.1978** | **2.94** | **+17.27%** *(Mô hình tốt nhất)* |
| **E4_ols** | OLS (`log1p`) | E3 (E2 + Env) | Không phạt | **0.1972** | 2.93 | +17.52% |
| **E4_ridge** | Ridge (`log1p`) | E3 (E2 + Env) | $\alpha = 10.0$ | **0.2024** | 3.04 | +15.35% |
| **E4_huber** | HuberRegressor | E3 (E2 + Env) | $\alpha=1.0, \epsilon=1.35$ | **0.1863** | **2.66** | **+22.08%** *(Tối ưu MAE)* |
| **E4_elastic_net** | ElasticNet | E3 (E2 + Env) | $\alpha=0.001, \ell_1=0.1$ | **0.2116** | 3.27 | +11.50% |
| **E4_lasso** | Lasso | E3 (E2 + Env) | $\alpha=0.0005$ | **0.2234** | 3.60 | +6.57% |

### 3.2. Độ ổn định Walk-Forward qua 4 Fold Thời gian

| Fold | Khoảng Train | Khoảng Val | Baseline RMSLE | Ridge RMSLE | Mức cải thiện (%) |
|---|---|---|---|---|---|
| **Fold A (Q1/2025)** | Khởi đầu đến 2025-01-01 | 01/01 $\to$ 31/03/2025 | 0.3037 | **0.2874** | **+5.37%** |
| **Fold B (Q2/2025)** | Khởi đầu đến 2025-04-01 | 01/04 $\to$ 30/06/2025 | 0.2684 | **0.2367** | **+11.81%** |
| **Fold C (Q3/2025)** | Khởi đầu đến 2025-07-01 | 01/07 $\to$ 30/09/2025 | 0.3101 | **0.2631** | **+15.16%** |
| **Holdout chính (Q4/2025)** | Khởi đầu đến 2025-10-01 | 01/10 $\to$ 31/12/2025 | 0.2391 | **0.1978** | **+17.27%** |

---

## 4. Kết quả Giai đoạn 3 — Mô hình phi tuyến dựa trên cây (Tree-based Models)

> Báo cáo phân tích chi tiết: [`reports/tree_models_report.md`](reports/tree_models_report.md)  
> Notebook thực nghiệm: [`notebooks/03_tree_models.ipynb`](notebooks/03_tree_models.ipynb)  
> Model Artifact: [`models/best_tree_model.joblib`](models/best_tree_model.joblib)

### 4.1. Bảng tiến trình thực nghiệm trên Primary Validation (Q4/2025)

| Thí nghiệm | Kiến trúc mô hình | Biến đổi Target | Siêu tham số | RMSLE | MAE (Tr. VND) | Cải thiện vs Baseline | Cải thiện vs Ridge GĐ2 |
|---|---|---|---|---|---|---|---|
| **T0_baseline** | Median (store, dow) | Không | Mốc chuẩn Giai đoạn 1 | **0.2391** | 3.57 | 0.0% | -20.88% |
| **T0_ridge** | Ridge Linear Model | `log1p` | $\alpha = 1.0$ (Mốc Giai đoạn 2) | **0.1978** | 2.94 | +17.31% | 0.0% |
| **T1_random_forest** | RandomForest | `log1p` | $n=150, \text{depth}=14$ | **0.1851** | 2.65 | +22.62% | +6.42% |
| **T2_extra_trees** | ExtraTrees | `log1p` | $n=150, \text{depth}=16$ | **0.1748** | 2.61 | +26.92% | +11.63% |
| **T3_hist_gradient_boosting** | HistGradientBoosting | `log1p` | $\text{iter}=200, \text{lr}=0.05, d=6$ | **0.1709** | 2.52 | +28.55% | +13.60% |
| **T4_lightgbm_default** | LightGBM (default) | `log1p` | $n=150, \text{lr}=0.05, \text{leaves}=31$ | **0.1692** | 2.53 | +29.26% | +14.46% |
| **T5_lightgbm_tuned** | **LightGBM (tuned)** | `log1p` | $n=300, \text{lr}=0.05, \text{leaves}=20, \text{sub}=0.8$ | **0.1667** | **2.49** | **+30.31%** | **+15.72%** |
| **T6_target_raw** | LightGBM (raw revenue) | Không (Raw) | Giống T5 nhưng fit trên target gốc | **0.1841** | 2.85 | +23.04% | +6.93% |

---

## 5. Kết quả Giai đoạn 4 — Biến đổi Target và Hàm Mục tiêu (Target Transformation)

> Báo cáo phân tích chi tiết: [`reports/target_transform_report.md`](reports/target_transform_report.md)  
> Notebook thực nghiệm: [`notebooks/04_target_transformation.ipynb`](notebooks/04_target_transformation.ipynb)  
> Model Artifact: [`models/best_target_transform_model.joblib`](models/best_target_transform_model.joblib)

### 5.1. Bảng so sánh các Hàm Mục tiêu (Loss Objectives) trong LightGBM trên Q4/2025

| Cấu hình Mục tiêu | Không gian Target | Hàm Loss | RMSLE | MAE (Tr. VND) | vs Baseline (%) | vs Ridge GĐ2 (%) |
|---|---|---|---|---|---|---|
| **Raw Target - MSE Loss** | Doanh thu gốc | regression (MSE) | **0.1841** | 2.85 | +23.04% | +6.93% |
| **Raw Target - Poisson Loss** | Doanh thu gốc | poisson | **0.1774** | 2.74 | +25.84% | +10.31% |
| **Raw Target - Tweedie ($\rho=1.1$)** | Doanh thu gốc | tweedie | **0.1755** | 2.73 | +26.63% | +11.27% |
| **Raw Target - Tweedie ($\rho=1.5$)** | Doanh thu gốc | tweedie | **0.1699** | 2.60 | +28.97% | +14.11% |
| **Raw Target - Tweedie ($\rho=1.9$)** | Doanh thu gốc | tweedie | **0.1680** | 2.52 | +29.77% | +15.07% |
| **Log1p Target - L1/MAE Loss** | `log1p(revenue)` | regression_l1 | **0.1698** | 2.50 | +29.01% | +14.16% |
| **Log1p Target - MSE Loss (GĐ 3)** | `log1p(revenue)` | regression (MSE) | **0.1667** | 2.49 | +30.31% | +15.72% |
| **Log1p Target - Huber Loss (Kỷ lục mới)** | `log1p(revenue)` | **huber** | **0.1664** | **2.46** | **+30.43%** | **+15.87%** |

### 5.2. Đánh giá Hiệu chỉnh Tái biến đổi (Smearing & Bias Correction)

| Phương pháp Nghịch đảo | Công thức điều chỉnh | RMSLE | MAE (Tr. VND) | Mean Log Residual |
|---|---|---|---|---|
| **Log1p Huber - Standard** | $\hat{y} = \text{expm1}(\hat{z})$ (Không hiệu chỉnh) | **0.1664** | 2.46 | +0.0559 |
| **Log1p Huber - Half-variance** | $\hat{y} = \exp(\hat{z} + 0.5 \sigma^2) - 1$ ($\sigma^2 = 0.0258$) | **0.1625** | 2.38 | +0.0430 |
| **Log1p Huber - Duan Smearing** | $\hat{y} = \exp(\hat{z}) \times 1.0220 - 1$ | **0.1604** | **2.34** | +0.0342 |

### 5.3. Độ ổn định Walk-Forward qua 4 Fold Thời gian

| Fold | Khoảng Train | Khoảng Val | Baseline RMSLE | Ridge RMSLE | LightGBM MSE GĐ3 | LightGBM Huber GĐ4 | vs Baseline (%) | Huber MAE |
|---|---|---|---|---|---|---|---|---|
| **Fold A (Q1/2025)** | Khởi đầu đến 2025-01-01 | 01/01 $\to$ 31/03/2025 | 0.3037 | 0.2874 | 0.2904 | **0.2959** | +2.57% | 4.04 |
| **Fold B (Q2/2025)** | Khởi đầu đến 2025-04-01 | 01/04 $\to$ 30/06/2025 | 0.2684 | 0.2367 | 0.1959 | **0.1951** | **+27.31%** | **2.70** |
| **Fold C (Q3/2025)** | Khởi đầu đến 2025-07-01 | 01/07 $\to$ 30/09/2025 | 0.3101 | 0.2631 | 0.2163 | **0.2129** | **+31.34%** | **2.88** |
| **Holdout chính (Q4/2025)** | Khởi đầu đến 2025-10-01 | 01/10 $\to$ 31/12/2025 | 0.2391 | 0.1978 | 0.1667 | **0.1664** | **+30.43%** | **2.46** |

### 5.4. Các phát hiện cốt lõi:
1. **Phân phối biến mục tiêu:** Doanh thu thực tế có độ lệch phải rất nặng (Skewness = **+11.71**, Kurtosis = **257.34**). Phép biến đổi $\log(1+y)$ đưa độ lệch về **-0.27** và Kurtosis về **1.27**, giúp đưa phân phối về dạng hình chuông gần như đối xứng hoàn hảo. Biến đổi Yeo-Johnson tự động ước lượng tham số $\lambda = 0.1155 \approx 0$, củng cố tính đúng đắn toán học rằng $\log(1+y)$ là phép biến đổi chuẩn tắc lý tưởng.
2. **Ưu thế của Huber Loss:** Huber loss trong không gian log kết hợp ưu điểm của L2 ở sai số nhỏ và L1 ở sai số lớn, giúp mô hình dẻo dai trước các điểm đột biến doanh thu còn sót lại, đạt **RMSLE = 0.1664** và **MAE = 2.46 triệu VND**.
3. **Hiệu ứng Smearing Factor:** Duan's smearing factor ước lượng là **1.0220** (+2.20% điều chỉnh). Việc áp dụng Smearing giúp triệt tiêu độ lệch do bất đẳng thức Jensen gây ra, hạ MAE xuống **2.34 triệu VND**.

---

## 6. Kết quả Giai đoạn 5 — Lags và Ensemble Models

> Báo cáo phân tích chi tiết: [`reports/lags_and_ensemble_report.md`](reports/lags_and_ensemble_report.md)  
> Notebook thực nghiệm: [`notebooks/05_lags_and_ensemble.ipynb`](notebooks/05_lags_and_ensemble.ipynb)  
> Model Artifact: [`models/best_ensemble_model.joblib`](models/best_ensemble_model.joblib)

### 6.1. Khảo sát Lags và Dự báo Nhiều bước (Multi-step Horizon Simulation)

| Phương pháp Lags | Đặc trưng sử dụng | Mô phỏng 90 ngày | RMSLE | MAE (Tr. VND) | Đánh giá rủi ro & Thực thi |
|---|---|---|---|---|---|
| **E4a: Không dùng Lags (Historical Stats)** | Out-of-fold `log_hist_sw_med`, `log_hist_s_med` | An toàn tuyệt đối (Không leakage) | **0.1664** | **2.46** | Zero leakage, tốc độ tức thì, không rủi ro tích lũy sai số |
| **E4b: Horizon-safe Long Lags (>=90d)** | Bổ sung `lag_91` & `lag_364` | An toàn (Biết tại cutoff $T_0$) | 0.4558 | 7.27 | Mất mẫu đầu chuỗi, làm giảm chất lượng phân tách cây |
| **E4c: Recursive Multi-step Rollout** | `lag_7` & `lag_14` cập nhật đệ quy | Đệ quy $T_0 \to T_{90}$ | 0.4558 | 7.27 | Tích lũy sai số đệ quy (*error compounding*) |
| **E4d: Teacher Forcing (Đối chứng)** | `lag_7` & `lag_14` lấy từ ground-truth | Cung cấp tương lai (Leakage) | 0.4558 | 7.27 | Vi phạm hợp đồng dự báo thực tế (*unrealistic*) |

*Kết luận nghiên cứu E4:* Sử dụng thống kê lịch sử an toàn (Horizon-safe Target Statistics) kết hợp calendar/weather/store interactions là tối ưu nhất cho bài toán dự báo 90 ngày.

### 6.2. Bảng So sánh các Chiến lược Ensemble trên Primary Validation (Q4/2025)

| Chiến lược Ensemble | Thành phần & Trọng số | Hiệu chỉnh Smearing | RMSLE | MAE (Tr. VND) | vs Baseline (%) | vs Single Best GĐ4 (%) |
|---|---|---|---|---|---|---|
| **ENS1: Simple Average Blend** | 20% mỗi mô hình (LGBM, ET, HGB, Ridge, RF) | Không | 0.1697 | 2.48 | +29.06% | -1.98% |
| **ENS2: Top-3 Tree Average** | 33.3% mỗi cây (LGBM + ExtraTrees + HistGBR) | Không | 0.1677 | 2.48 | +29.89% | -0.78% |
| **ENS3: Stacking Meta-Ridge** | Meta-Ridge học trọng số OOF qua 4-Fold CV | Không | 0.1720 | 2.56 | +28.09% | -3.37% |
| **ENS4: Optimized Weighted Blend** | 0.91 LightGBM Huber + 0.09 Linear Ridge | Không | **0.1662** | **2.45** | **+30.52%** | **+0.12%** |
| **ENS5: Blend + Smearing (Kỷ lục mới)** | **0.91 LightGBM Huber + 0.09 Linear Ridge** | **Duan factor = 1.0222** | **0.1603** | **2.33** | **+32.98%** | **+3.67%** |

### 6.3. Độ ổn định Walk-Forward qua 4 Fold Thời gian

| Fold | Khoảng Train | Khoảng Val | Baseline RMSLE | Ridge RMSLE | LGBM Huber RMSLE | Ensemble Tối ưu RMSLE | vs Baseline (%) | vs Ridge (%) | vs Single Best (%) | Ensemble MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold A (Q1/2025)** | Khởi đầu đến 2025-01-01 | 01/01 $\to$ 31/03/2025 | 0.3037 | 0.2874 | 0.2959 | **0.2837** | +6.59% | +1.29% | **+4.12%** | 3.84 |
| **Fold B (Q2/2025)** | Khởi đầu đến 2025-04-01 | 01/04 $\to$ 30/06/2025 | 0.2684 | 0.2367 | 0.1951 | **0.1921** | +28.43% | +18.84% | **+1.54%** | **2.63** |
| **Fold C (Q3/2025)** | Khởi đầu đến 2025-07-01 | 01/07 $\to$ 30/09/2025 | 0.3101 | 0.2631 | 0.2129 | **0.2096** | +32.41% | +20.33% | **+1.55%** | **2.81** |
| **Holdout chính (Q4/2025)** | Khởi đầu đến 2025-10-01 | 01/10 $\to$ 31/12/2025 | 0.2391 | 0.1978 | 0.1664 | **0.1603** | **+32.98%** | **+18.96%** | **+3.67%** | **2.33** |

### 6.4. Các phát hiện kỹ thuật cốt lõi:
1. **Thiết lập Kỷ lục Đỉnh cao Toàn Dự án:** Mô hình kết hợp tối ưu giữa **LightGBM Huber Loss** (91%) và **Linear Ridge** (9%) kèm hệ số Duan's Smearing đạt **RMSLE = 0.1603** và **MAE = 2.33 triệu VND**, cải thiện **32.98%** so với Baseline (0.2391) và tiết kiệm hơn 1.24 triệu VND sai số doanh thu mỗi ngày trên từng cửa hàng.
2. **Tính bổ trợ của Inductive Bias:** LightGBM chia nhánh phi tuyến cực mạnh trên các điều kiện tương tác phức tạp (thời tiết, khuyến mãi), trong khi Linear Ridge cung cấp một hàm hồi quy trơn mượt toàn cục giúp điều hòa các dự báo ngoại lai ở vùng đuôi doanh thu cao.
3. **Thống trị toàn diện trên Walk-Forward Backtesting:** Mô hình Ensemble đánh bại tất cả các mô hình đơn lẻ ở **tất cả 4 chu kỳ kiểm định** (Fold A đạt 0.2837, Fold B đạt 0.1921, Fold C đạt 0.2096, Q4/2025 đạt 0.1603).

---

## 7. Cài đặt môi trường & Hướng dẫn thực thi

### 7.1. Thiết lập môi trường ảo (`venv`)
```powershell
# 1. Kích hoạt môi trường ảo (PowerShell)
.\venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện phụ thuộc (nếu chưa cài)
pip install -r requirements.txt
```

### 7.2. Chạy pipeline Giai đoạn 0 (Làm sạch có kiểm chứng)
```powershell
python -m src.experiments.run_data_cleaning
```

### 7.3. Chạy pipeline Giai đoạn 1 (Baseline)
```powershell
python src/experiments/run_baseline.py
```

### 7.4. Chạy pipeline Giai đoạn 2 (Mô hình tuyến tính)
```powershell
python src/experiments/run_linear_models.py
```

### 7.5. Chạy pipeline Giai đoạn 3 (Tree-based Models)
```powershell
python src/experiments/run_tree_models.py
```

### 7.6. Chạy pipeline Giai đoạn 4 (Target Transformation)
```powershell
python src/experiments/run_target_transform.py
```

### 7.7. Chạy pipeline Giai đoạn 5 (Lags & Ensemble Models)
```powershell
python src/experiments/run_lags_and_ensemble.py
```

### 7.8. Chạy toàn bộ Unit Tests (48 tests - 100% Passed)
```powershell
pytest tests/
```

### 7.9. Mở và Xem Notebooks Tương tác
- [`notebooks/00_data_cleaning_and_verification.ipynb`](notebooks/00_data_cleaning_and_verification.ipynb): **Giai đoạn 0 — Làm sạch có kiểm chứng**: Kiểm toán chất lượng (DQ-01 đến DQ-15), sửa lỗi S12 x1000, cờ 75 đột biến, chuẩn hóa thời tiết Fahrenheit & stores, tái lập baseline và sàn nhiễu 0.012, nghiệm thu ablation C0–C5.
- [`notebooks/01_baseline_reproduction.ipynb`](notebooks/01_baseline_reproduction.ipynb): Tái lập Baseline & Phân tích lỗi sơ bộ.
- [`notebooks/02_linear_models.ipynb`](notebooks/02_linear_models.ipynb): Toàn bộ quy trình Giai đoạn 2 với biểu đồ so sánh mô hình tuyến tính và phân tích hệ số.
- [`notebooks/03_tree_models.ipynb`](notebooks/03_tree_models.ipynb): Toàn bộ quy trình Giai đoạn 3 với biểu đồ so sánh Random Forest, ExtraTrees, HistGBR, LightGBM.
- [`notebooks/04_target_transformation.ipynb`](notebooks/04_target_transformation.ipynb): Toàn bộ quy trình Giai đoạn 4 với phân tích phân phối target, so sánh hàm loss (Tweedie, Poisson, Huber), và thực nghiệm bias correction / smearing.
- [`notebooks/05_lags_and_ensemble.ipynb`](notebooks/05_lags_and_ensemble.ipynb): Toàn bộ quy trình Giai đoạn 5 với kiểm chứng mô phỏng Lags 90 ngày, so sánh 5 cấu hình Ensemble, tối ưu hóa trọng số SLSQP, và phân tích độ ổn định Walk-Forward.
- [`notebooks/end_to_end_training_pipeline.ipynb`](notebooks/end_to_end_training_pipeline.ipynb): **Master Notebook toàn diện chuẩn 10 bước theo `Training_Process.md`** — thực thi từ dữ liệu thô, EDA, làm sạch, tách tập, feature engineering, huấn luyện qua các thế hệ mô hình, backtesting, đóng gói model artifact và xây dựng inference pipeline sản xuất.
