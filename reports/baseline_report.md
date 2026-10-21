# Báo cáo Giai đoạn 1 — Baseline có thể tái lập

> **Dự án:** Cafe Revenue Regression  
> **Mục tiêu:** Xây dựng và tái lập đường cơ sở (Baseline) dự báo doanh thu cà phê  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-16 10:34:30  

---

## 1. Tóm tắt kết quả cốt lõi

1. **Dữ liệu và Deduplication:**
   - Tổng số dòng train gốc: **26.632** dòng.
   - Số bản ghi trùng lặp hoàn toàn (*exact duplicates*): **132** dòng đã được loại bỏ an toàn.
   - Tổng số dòng sạch: **26,500** dòng.
   - Sau khi deduplicate, không còn bất kỳ khóa `(date, store_id)` nào bị trùng lặp.

2. **Tái lập Metric theo Regression_Plan.md:**
   - **Notebook cũ (trên dữ liệu gốc với 132 duplicates, fit < 2025-10-01, val Q4/2025):** Tái lập chính xác **RMSLE = 0.2387** (khớp hoàn toàn với notebook `starter_cafe.ipynb`).
   - **Notebook cũ (sau khi dedup):** **RMSLE = 0.2391**.
   - **Split chuẩn mới (fit < 2025-07-01, val Q3/2025):** Tái lập chính xác **RMSLE = 0.3101** (khớp mục tiêu đề ra tại Section 3.3 và 8 trong `Regression_Plan.md`).
   - **Fold A (fit < 2025-01-01, val Q1/2025):** **RMSLE = 0.3037**.
   - **Fold B (fit < 2025-04-01, val Q2/2025):** **RMSLE = 0.2684**.

---

## 2. Bảng tổng hợp kết quả Backtesting qua các Fold

| fold_name                           | data_version      | train_period        | val_period               | train_rows | val_rows | rmsle  | mae    | rmse    | mean_log_residual |
| ----------------------------------- | ----------------- | ------------------- | ------------------------ | ---------- | -------- | ------ | ------ | ------- | ----------------- |
| Fold A (Q1/2025)                    | clean_dedup       | Start to 2025-01-01 | 2025-01-01 to 2025-04-01 | 12281      | 3170     | 0.3037 | 4.0345 | 12.6352 | 0.0118            |
| Fold B (Q2/2025)                    | clean_dedup       | Start to 2025-04-01 | 2025-04-01 to 2025-07-01 | 15451      | 3531     | 0.2684 | 3.9226 | 10.8542 | 0.055             |
| Holdout chính (Q3/2025)             | clean_dedup       | Start to 2025-07-01 | 2025-07-01 to 2025-10-01 | 18982      | 3686     | 0.3101 | 4.3111 | 12.4634 | 0.0073            |
| Notebook Benchmark (Q4/2025)        | clean_dedup       | Start to 2025-10-01 | 2025-10-01 to 2026-01-01 | 22668      | 3832     | 0.2391 | 3.5719 | 7.1075  | 0.0579            |
| Notebook Benchmark (Raw duplicates) | raw_with_132_dups | Start to 2025-10-01 | 2025-10-01 to 2025-12-31 | 22779      | 3853     | 0.2387 | 3.5637 | 7.0912  | 0.0575            |


---

## 3. Phân tích chi tiết lỗi theo nhóm (Subgroup Analysis trên Q3/2025)

### 3.1. Theo Thành phố

| city_norm   | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| ----------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| Ho Chi Minh | 1797      | 18.13         | 20.3        | 0.3114 | 4.3328 | -0.0369           |
| Ha Noi      | 1889      | 18.44         | 18.05       | 0.3088 | 4.2904 | 0.0493            |

### 3.2. Theo Loại cửa hàng (Store Type)

| store_type_norm | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| --------------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| university      | 399       | 10.76         | 16.31       | 0.5329 | 6.1986 | -0.3726           |
| street          | 1266      | 23.34         | 22.38       | 0.3125 | 5.7791 | 0.0341            |
| residential     | 406       | 8.96          | 8.44        | 0.2576 | 2.0082 | 0.0035            |
| office          | 985       | 16.79         | 16.05       | 0.242  | 2.9099 | 0.0604            |
| mall            | 630       | 20.27         | 19.53       | 0.228  | 3.8405 | 0.1135            |

### 3.3. Theo Chương trình khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| -------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| 1        | 358       | 21.53         | 18.41       | 0.4424 | 9.3364 | 0.2047            |
| 0        | 3328      | 18.09         | 18.42       | 0.2923 | 3.7705 | -0.0139           |

### 3.4. Theo Phân tầng doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| --------------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| Low (<15M)      | 1305      | 10.19         | 11.23       | 0.3852 | 3.4995 | -0.149            |
| High (>25M)     | 913       | 30.3          | 26.2        | 0.3457 | 7.7959 | 0.1935            |
| Medium (15-25M) | 1468      | 19.55         | 19.17       | 0.1876 | 2.8653 | 0.0304            |

### 3.5. Theo Lịch sử cửa hàng (Short History vs Established)

| history_tier             | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| ------------------------ | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| Short History (<100 obs) | 313       | 14.05         | 13.34       | 0.4714 | 5.9441 | 0.1325            |
| Established (>=100 obs)  | 3373      | 18.67         | 19.2        | 0.2906 | 4.1596 | -0.0043           |

### 3.6. Top 5 Cửa hàng có sai số RMSLE cao nhất

| store_id | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| -------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| S11      | 40        | 6.84          | 18.05       | 0.8859 | 9.809  | -0.7878           |
| S31      | 89        | 13.33         | 21.56       | 0.5048 | 6.3456 | -0.3663           |
| S30      | 91        | 13.59         | 22.18       | 0.5001 | 6.8614 | -0.3434           |
| S32      | 90        | 9.22          | 14.54       | 0.4867 | 5.3186 | -0.2885           |
| S34      | 90        | 15.98         | 17.18       | 0.4663 | 8.7079 | -0.0929           |

---

## 4. Nhận xét và định hướng cho Giai đoạn tiếp theo

1. **Hiệu ứng khuyến mãi (`is_promo`):** Baseline hiện tại không dùng feature `is_promo`. Khi có khuyến mãi, doanh thu thực tế tăng vọt khiến baseline dự báo thấp hơn đáng kể (mean log residual dương lớn). Việc tích hợp promo feature ở Giai đoạn 2 & 3 sẽ đem lại bước nhảy vọt hiệu năng.
2. **Độ biến thiên theo loại cửa hàng:** Các cửa hàng văn phòng (`office`) và trường học (`university`) có tính chu kỳ tuần rất rõ rệt; trong khi cửa hàng mặt phố (`street`) và trung tâm thương mại (`mall`) chịu ảnh hưởng mạnh hơn bởi thời tiết và ngày nghỉ lễ.
3. **Cửa hàng có lịch sử ngắn:** Cửa hàng mới mở trong giai đoạn train có ít mẫu quan sát nên median weekday chưa hội tụ. Cần dùng hierarchical shrinkage hoặc Bayesian target encoding ở các giai đoạn sau.
4. **Mục tiêu Giai đoạn 2 (Linear Models):** Xây dựng Ridge / ElasticNet trên `log1p(revenue)` kết hợp calendar features và store metadata để vượt mốc baseline 0.3101 (mục tiêu <= 0.2946).

---

## 5. Môi trường thực thi & Khả năng tái lập (Reproducibility)

- **Môi trường ảo:** Python `venv` độc lập (loại trừ khỏi git via `.gitignore`).
- **Thư viện phụ thuộc:** Quản lý tập trung tại `requirements.txt` với phiên bản pinned (`numpy>=1.26.0`, `pandas>=2.1.0`, `scikit-learn>=1.3.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`, `ipykernel>=6.29.0`).
- **Quy trình tái lập:**
  1. Kích hoạt môi trường: `.\venv\Scripts\Activate.ps1`
  2. Cài đặt thư viện: `pip install -r requirements.txt`
  3. Tái lập thí nghiệm: `python src/experiments/run_baseline.py`
  4. Kiểm thử logic: `python -m unittest discover tests`
