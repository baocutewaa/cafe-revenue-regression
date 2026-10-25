# Báo cáo Giai đoạn 1 — Baseline có thể tái lập

> **Dự án:** Cafe Revenue Regression  
> **Mục tiêu:** Xây dựng và tái lập đường cơ sở (Baseline) dự báo doanh thu cà phê  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-25 10:49:59  

---

## 1. Tóm tắt kết quả cốt lõi

1. **Dữ liệu và Deduplication:**
   - Tổng số dòng train gốc: **26.632** dòng.
   - Số bản ghi trùng lặp hoàn toàn (*exact duplicates*): **132** dòng đã được loại bỏ an toàn.
   - Tổng số dòng sạch: **26,500** dòng.
   - Sau khi deduplicate, không còn bất kỳ khóa `(date, store_id)` nào bị trùng lặp.

2. **Tái lập Metric đồng bộ Starter Notebook (`starter_cafe.ipynb`):**
   - **Primary Baseline (fit < 2025-10-01, val Q4/2025 trên dữ liệu sạch):** **RMSLE = 0.2391** | **MAE = 3.57 triệu VND**.
   - **Notebook Benchmark (fit < 2025-10-01, val Q4/2025 trên dữ liệu thô 132 dups):** Tái lập chính xác **RMSLE = 0.2387** (khớp 100% với `starter_cafe.ipynb`).
   - **Fold A (fit < 2025-01-01, val Q1/2025):** **RMSLE = 0.3037**.
   - **Fold B (fit < 2025-04-01, val Q2/2025):** **RMSLE = 0.2684**.
   - **Fold C (fit < 2025-07-01, val Q3/2025):** **RMSLE = 0.3101**.

---

## 2. Bảng tổng hợp kết quả Backtesting qua các Fold

| fold_name                           | data_version      | train_period        | val_period               | train_rows | val_rows | rmsle  | mae    | rmse    | mean_log_residual |
| ----------------------------------- | ----------------- | ------------------- | ------------------------ | ---------- | -------- | ------ | ------ | ------- | ----------------- |
| Fold A (Q1/2025)                    | clean_dedup       | Start to 2025-01-01 | 2025-01-01 to 2025-04-01 | 12281      | 3170     | 0.3037 | 4.0345 | 12.6352 | 0.0118            |
| Fold B (Q2/2025)                    | clean_dedup       | Start to 2025-04-01 | 2025-04-01 to 2025-07-01 | 15451      | 3531     | 0.2684 | 3.9226 | 10.8542 | 0.055             |
| Fold C (Q3/2025)                    | clean_dedup       | Start to 2025-07-01 | 2025-07-01 to 2025-10-01 | 18982      | 3686     | 0.3101 | 4.3111 | 12.4634 | 0.0073            |
| Holdout chính (Q4/2025)             | clean_dedup       | Start to 2025-10-01 | 2025-10-01 to 2026-01-01 | 22668      | 3832     | 0.2391 | 3.5719 | 7.1075  | 0.0579            |
| Notebook Benchmark (Raw duplicates) | raw_with_132_dups | Start to 2025-10-01 | 2025-10-01 to 2025-12-31 | 22779      | 3853     | 0.2387 | 3.5637 | 7.0912  | 0.0575            |


---

## 3. Phân tích chi tiết lỗi theo nhóm (Subgroup Analysis trên Primary Val Q4/2025)

### 3.1. Theo Thành phố

| city_norm   | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| ----------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| Ha Noi      | 2011      | 19.09         | 18.1        | 0.2476 | 3.5655 | 0.0812            |
| Ho Chi Minh | 1821      | 19.79         | 20.12       | 0.2294 | 3.579  | 0.0321            |

### 3.2. Theo Loại cửa hàng (Store Type)

| store_type_norm | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| --------------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| university      | 478       | 16.24         | 14.71       | 0.3467 | 4.2144 | 0.1373            |
| street          | 1341      | 22.68         | 22.31       | 0.2407 | 4.4092 | 0.0155            |
| residential     | 361       | 8.97          | 8.71        | 0.2263 | 1.7858 | -0.0278           |
| mall            | 658       | 21.54         | 19.73       | 0.1995 | 3.4794 | 0.1026            |
| office          | 994       | 18.13         | 16.16       | 0.1989 | 2.8433 | 0.0782            |

### 3.3. Theo Chương trình khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| -------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| 1        | 488       | 25.83         | 18.56       | 0.3907 | 8.4908 | 0.3241            |
| 0        | 3344      | 18.83         | 18.52       | 0.208  | 2.8541 | 0.019             |

### 3.4. Theo Phân tầng doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| --------------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| High (>25M)     | 990       | 30.09         | 26.22       | 0.277  | 6.2519 | 0.1805            |
| Low (<15M)      | 1048      | 10.66         | 10.68       | 0.2626 | 2.2961 | -0.051            |
| Medium (15-25M) | 1794      | 19.52         | 18.55       | 0.1988 | 2.8383 | 0.0537            |

### 3.5. Theo Lịch sử cửa hàng (Short History vs Established)

| history_tier             | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| ------------------------ | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| Short History (<100 obs) | 218       | 14.92         | 18.1        | 0.4829 | 6.2184 | 0.1741            |
| Established (>=100 obs)  | 3614      | 19.65         | 19.15       | 0.2158 | 3.4123 | 0.0508            |

### 3.6. Top 5 Cửa hàng có sai số RMSLE cao nhất

| store_id | n_samples | actual_median | pred_median | rmsle  | mae    | mean_log_residual |
| -------- | --------- | ------------- | ----------- | ------ | ------ | ----------------- |
| S11      | 91        | 12.87         | 7.25        | 0.5642 | 5.3312 | 0.432             |
| S01      | 24        | 11.9          | 18.1        | 0.559  | 7.9737 | -0.3974           |
| S45      | 21        | 13.91         | 18.1        | 0.4172 | 6.9429 | -0.1254           |
| S19      | 82        | 20.78         | 18.1        | 0.3615 | 6.5038 | 0.132             |
| S13      | 92        | 22.42         | 20.06       | 0.3151 | 6.2253 | 0.0747            |

---

## 4. Nhận xét và định hướng cho Giai đoạn tiếp theo

1. **Hiệu ứng khuyến mãi (`is_promo`):** Baseline hiện tại không dùng feature `is_promo`. Khi có khuyến mãi, doanh thu thực tế tăng vọt khiến baseline dự báo thấp hơn đáng kể (mean log residual dương lớn). Việc tích hợp promo feature ở Giai đoạn 2 & 3 sẽ đem lại bước nhảy vọt hiệu năng.
2. **Độ biến thiên theo loại cửa hàng:** Các cửa hàng văn phòng (`office`) và trường học (`university`) có tính chu kỳ tuần rất rõ rệt; trong khi cửa hàng mặt phố (`street`) và trung tâm thương mại (`mall`) chịu ảnh hưởng mạnh hơn bởi thời tiết và ngày nghỉ lễ.
3. **Cửa hàng có lịch sử ngắn:** Cửa hàng mới mở trong giai đoạn train có ít mẫu quan sát nên median weekday chưa hội tụ. Cần dùng hierarchical shrinkage hoặc Bayesian target encoding ở các giai đoạn sau.
4. **Mục tiêu Giai đoạn 2 (Linear Models):** Xây dựng Ridge / ElasticNet trên `log1p(revenue)` kết hợp calendar features và store metadata để vượt mốc baseline 0.2391 (mục tiêu KPI <= 0.2270, cải thiện >= 5%).
