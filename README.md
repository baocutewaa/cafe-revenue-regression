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
│   └── processed/           # train.csv, val.csv, test.csv đã xử lý trùng lặp
├── regression_splits/       # Thư mục split chuẩn (train.csv, val.csv, test.csv)
├── models/
│   └── baseline_median.json # Model artifact lưu các bảng lookup trung vị
├── notebooks/
│   └── 01_baseline_reproduction.ipynb  # Notebook tái lập và phân tích baseline
├── reports/
│   ├── baseline_report.md              # Báo cáo chi tiết Giai đoạn 1
│   ├── validation_results.csv          # Bảng tổng hợp điểm số qua các fold
│   └── baseline_val_predictions.csv    # Kết quả dự báo và residual trên Q3/2025
├── src/
│   ├── config.py            # Cấu hình đường dẫn, thời gian split và hằng số
│   ├── data/
│   │   └── load.py          # Load dữ liệu, deduplication, sinh temporal splits
│   ├── evaluation/
│   │   ├── metrics.py       # RMSLE, MAE, RMSE, log residual
│   │   └── backtest.py      # Walk-forward backtesting và phân tích nhóm
│   ├── models/
│   │   └── baseline.py      # WeekdayMedianBaseline với 2 tầng fallback
│   └── experiments/
│       └── run_baseline.py  # Pipeline chính thực thi Giai đoạn 1
├── tests/
│   ├── test_metrics.py      # Unit test cho hàm metric
│   ├── test_baseline.py     # Unit test cho baseline model
│   └── test_splits.py       # Unit test cho hợp đồng dữ liệu và splits
├── scripts/
│   └── build_notebook.py    # Script khởi tạo và cập nhật notebook
├── requirements.txt
└── README.md
```

---

## 2. Kết quả Giai đoạn 1 — Baseline có thể tái lập

> Báo cáo phân tích chi tiết: [`reports/baseline_report.md`](reports/baseline_report.md)

### 2.1. Làm sạch & Chia tập dữ liệu
- Dữ liệu `train.csv` ban đầu: **26.632** dòng.
- Đã loại bỏ **132** bản ghi trùng lặp hoàn toàn (*exact duplicates*), còn lại **26.500** dòng sạch.
- Phân chia theo thời gian (chuẩn hóa theo quý):
  - **Train:** `< 2025-07-01` (18.982 dòng, 71,63%)
  - **Validation (Q3/2025):** `2025-07-01 <= date < 2025-10-01` (3.686 dòng, 13,91%)
  - **Test Holdout (Q4/2025):** `2025-10-01 <= date < 2026-01-01` (3.832 dòng, 14,46%)

### 2.2. Điểm số Backtesting qua các Fold (RMSLE)

| Fold / Thí nghiệm | Dữ liệu | Khoảng Train | Khoảng Val | RMSLE | MAE (Tr. VND) | Ý nghĩa |
|---|---|---|---|---|---|---|
| **Holdout chính (Q3/2025)** | clean_dedup | < 2025-07-01 | Q3/2025 | **0.3101** | 4.31 | Baseline chuẩn cho các vòng sau |
| **Fold A (Q1/2025)** | clean_dedup | < 2025-01-01 | Q1/2025 | **0.3037** | 4.03 | Độ ổn định walk-forward |
| **Fold B (Q2/2025)** | clean_dedup | < 2025-04-01 | Q2/2025 | **0.2684** | 3.92 | Độ ổn định walk-forward |
| **Notebook benchmark (clean)** | clean_dedup | < 2025-10-01 | Q4/2025 | **0.2391** | 3.57 | Đối chiếu sau dedup |
| **Notebook benchmark (raw)** | raw (132 dups) | < 2025-10-01 | Q4/2025 | **0.2387** | 3.56 | Tái lập chính xác `starter_cafe.ipynb` |

---

## 3. Hướng dẫn chạy lại mã nguồn

### Chạy toàn bộ pipeline Baseline
```powershell
python src/experiments/run_baseline.py
```

### Chạy bộ Unit Tests
```powershell
python -m unittest discover tests
```

### Mở Notebook tương tác
Mở file [`notebooks/01_baseline_reproduction.ipynb`](notebooks/01_baseline_reproduction.ipynb) trong VS Code / Jupyter để xem chi tiết biểu đồ trực quan hóa phân phối residual và so sánh doanh thu thực tế vs dự báo.
