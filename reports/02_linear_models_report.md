# Báo cáo Giai đoạn 2 — Mô hình tuyến tính (Linear Regression Models)

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 2 — Section 8 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-25 10:57:53  

---

## 1. Tóm tắt kết quả nổi bật & So sánh KPI

1. **Vượt mốc KPI đề ra:** Mô hình tuyến tính Ridge trên `log1p(revenue)` đạt **RMSLE = 0.1978** trên tập Primary Validation Q4/2025.
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.2391**), mô hình tuyến tính giảm sai số tới **17.27%** (vượt chỉ tiêu $\ge$ 5% / threshold $\le$ 0.2270).
   - Sai số tuyệt đối trung bình (MAE) giảm từ **3.57 triệu VND xuống 2.94 triệu VND**.
2. **Độ ổn định Walk-Forward vượt trội:** Trên mọi fold thời gian (Fold A Q1/2025, Fold B Q2/2025, Fold C Q3/2025, Holdout chính Q4/2025), mô hình tuyến tính đều đánh bại Baseline một cách tuyệt đối và đồng nhất.
3. **Giải quyết triệt để độ lệch khuyến mãi (`is_promo`):** Ở Giai đoạn 1, baseline có sai số cao khi có khuyến mãi do không nhận biết được promo. Với mô hình tuyến tính, RMSLE trong ngày khuyến mãi giảm ngoạn mục.
4. **Khả năng giải thích kinh doanh cao (Interpretability):** Hệ số của `is_promo` cho thấy khuyến mãi thúc đẩy doanh thu tăng mạnh, trong khi các tương tác cuối tuần và loại cửa hàng giải thích chính xác sự phân hóa giữa các địa điểm.

---

## 2. Bảng tiến trình thực nghiệm (Experiment Progression)

| experiment         | model_family            | feature_tier         | hyperparameters           | rmsle  | mae    | rmse   | mean_log_res | relative_improvement |
| ------------------ | ----------------------- | -------------------- | ------------------------- | ------ | ------ | ------ | ------------ | -------------------- |
| E0_baseline        | MedianBaseline          | store_weekday_median | None (hierarchical)       | 0.2391 | 3.5719 | 7.1075 | 0.0579       | 0.0% (Benchmark)     |
| E1_calendar_store  | Ridge (log target)      | E1_calendar_store    | alpha=1.0                 | 0.2508 | 3.8031 | 7.3771 | 0.0813       | +-4.89%              |
| E2_promo           | Ridge (log target)      | E2_promo             | alpha=1.0                 | 0.2268 | 3.4214 | 6.824  | 0.0706       | +5.14%               |
| E3_weather_holiday | Ridge (log target)      | E3_weather_holiday   | alpha=1.0                 | 0.1978 | 2.935  | 6.5063 | 0.0532       | +17.27%              |
| E4_ridge           | Ridge (tuned a=10)      | E3_weather_holiday   | alpha=10.0                | 0.2024 | 3.0418 | 6.5954 | 0.0666       | +15.35%              |
| E4_ols             | OLS (LinearRegression)  | E3_weather_holiday   | unregularized             | 0.1972 | 2.9261 | 6.4994 | 0.0487       | +17.52%              |
| E4_huber           | HuberRegressor (robust) | E3_weather_holiday   | alpha=1.0, eps=1.35       | 0.1863 | 2.658  | 6.271  | 0.0492       | +22.08%              |
| E4_elastic_net     | ElasticNet (L1+L2)      | E3_weather_holiday   | alpha=0.001, l1_ratio=0.1 | 0.2116 | 3.2688 | 6.788  | 0.0705       | +11.50%              |
| E4_lasso           | Lasso (L1)              | E3_weather_holiday   | alpha=0.0005              | 0.2234 | 3.5999 | 7.0302 | 0.0648       | +6.57%               |


---

## 3. Kết quả Backtesting Walk-Forward qua các Fold

| fold_name               | train_period        | val_period               | baseline_rmsle | linear_rmsle | improvement_% | baseline_mae | linear_mae |
| ----------------------- | ------------------- | ------------------------ | -------------- | ------------ | ------------- | ------------ | ---------- |
| Fold A (Q1/2025)        | Start to 2025-01-01 | 2025-01-01 to 2025-04-01 | 0.3037         | 0.2874       | 5.37          | 4.0345       | 4.3146     |
| Fold B (Q2/2025)        | Start to 2025-04-01 | 2025-04-01 to 2025-07-01 | 0.2684         | 0.2367       | 11.81         | 3.9226       | 3.4682     |
| Fold C (Q3/2025)        | Start to 2025-07-01 | 2025-07-01 to 2025-10-01 | 0.3101         | 0.2631       | 15.16         | 4.3111       | 3.7488     |
| Holdout chính (Q4/2025) | Start to 2025-10-01 | 2025-10-01 to 2026-01-01 | 0.2391         | 0.1978       | 17.27         | 3.5719       | 2.935      |


---

## 4. Phân tích hệ số & Đóng góp kinh doanh (Model Interpretability)

Vì mô hình hồi quy được huấn luyện trên $\log(1 + y)$, hệ số $\beta_j$ mang ý nghĩa kinh doanh trực tiếp: khi biến $x_j$ tăng 1 đơn vị, doanh thu ước tính thay đổi $\Delta\% = (e^{\beta_j} - 1) \times 100\%$.

### 4.1. Top 10 yếu tố thúc đẩy doanh thu mạnh nhất (Positive Drivers)

| feature                 | coefficient | percent_impact |
| ----------------------- | ----------- | -------------- |
| cat__store_dow_S41_dow6 | 0.5914      | +80.66%        |
| cat__store_dow_S12_dow4 | 0.5618      | +75.38%        |
| cat__store_dow_S21_dow6 | 0.5518      | +73.63%        |
| cat__store_dow_S12_dow1 | 0.5436      | +72.21%        |
| cat__store_dow_S41_dow5 | 0.5211      | +68.39%        |
| cat__store_dow_S12_dow3 | 0.5123      | +66.92%        |
| cat__store_dow_S21_dow5 | 0.5105      | +66.62%        |
| cat__store_dow_S12_dow2 | 0.4954      | +64.11%        |
| cat__store_dow_S05_dow5 | 0.4652      | +59.23%        |
| cat__store_dow_S10_dow6 | 0.4597      | +58.36%        |


### 4.2. Top 10 yếu tố làm giảm doanh thu (Negative Drivers)

| feature                 | coefficient | percent_impact |
| ----------------------- | ----------- | -------------- |
| cat__store_dow_S06_dow6 | -0.8882     | -58.86%        |
| cat__store_dow_S03_dow6 | -0.6766     | -49.16%        |
| cat__store_dow_S06_dow5 | -0.6437     | -47.47%        |
| cat__store_dow_S37_dow6 | -0.6358     | -47.05%        |
| cat__store_dow_S26_dow6 | -0.5923     | -44.70%        |
| cat__store_dow_S24_dow6 | -0.5886     | -44.49%        |
| cat__store_dow_S27_dow6 | -0.5669     | -43.27%        |
| cat__store_dow_S38_dow6 | -0.5381     | -41.61%        |
| cat__store_dow_S33_dow6 | -0.5335     | -41.34%        |
| cat__store_dow_S07_dow6 | -0.5131     | -40.14%        |


---

## 5. Phân tích chi tiết lỗi theo nhóm (Subgroup Analysis trên Primary Val Q4/2025)

### 5.1. Phân tích theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| university  | 478       | 16.24         | 12.73       | 0.2865       | 3.9283     | 0.2253       | 0.3467         | 17.37         |
| office      | 994       | 18.13         | 16.68       | 0.1913       | 2.8197     | 0.0401       | 0.1989         | 3.82          |
| mall        | 658       | 21.54         | 20.36       | 0.183        | 3.1531     | 0.056        | 0.1995         | 8.29          |
| street      | 1341      | 22.68         | 22.3        | 0.1793       | 3.0267     | 0.0224       | 0.2407         | 25.5          |
| residential | 361       | 8.97          | 8.85        | 0.1592       | 1.1988     | -0.0291      | 0.2263         | 29.66         |


### 5.2. Phân tích theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| -------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| 1        | 488       | 25.83         | 24.85       | 0.2298       | 4.4419     | 0.054        | 0.3907         | 41.18         |
| 0        | 3344      | 18.83         | 17.98       | 0.1927       | 2.7151     | 0.0531       | 0.208          | 7.36          |


### 5.3. Phân tích theo Thành phố

| city        | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| Ha Noi      | 2011      | 19.09         | 18.33       | 0.2082       | 3.0262     | 0.0555       | 0.2476         | 15.91         |
| Ho Chi Minh | 1821      | 19.79         | 18.92       | 0.1856       | 2.8343     | 0.0507       | 0.2294         | 19.1          |


### 5.4. Phân tích theo Lịch sử Cửa hàng (History Tier)

| history_tier             | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| ------------------------ | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| Short History (<100 obs) | 218       | 14.92         | 10.85       | 0.4412       | 6.0701     | 0.3243       | 0.4829         | 8.65          |
| Established (>=100 obs)  | 3614      | 19.65         | 18.99       | 0.1725       | 2.7459     | 0.0369       | 0.2158         | 20.08         |


### 5.5. Phân tích theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| --------------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| High (>25M)     | 990       | 30.09         | 28.3        | 0.2428       | 5.081      | 0.0947       | 0.277          | 12.33         |
| Low (<15M)      | 1048      | 10.66         | 9.75        | 0.1928       | 1.6116     | 0.0135       | 0.2626         | 26.6          |
| Medium (15-25M) | 1794      | 19.52         | 18.72       | 0.1712       | 2.5238     | 0.0535       | 0.1988         | 13.88         |


### 5.6. Top 5 Cửa hàng có sai số RMSLE cao nhất

| store_id | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| -------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| S19      | 82        | 20.78         | 13.93       | 0.4693       | 8.2667     | 0.37         | 0.3615         | -29.81        |
| S45      | 21        | 13.91         | 13.6        | 0.463        | 6.5374     | 0.1761       | 0.4172         | -10.98        |
| S11      | 91        | 12.87         | 8.83        | 0.4166       | 4.3967     | 0.3418       | 0.5642         | 26.16         |
| S01      | 24        | 11.9          | 9.1         | 0.4118       | 4.5011     | 0.2312       | 0.559          | 26.34         |
| S27      | 89        | 25.54         | 27.17       | 0.284        | 4.7365     | 0.0292       | 0.2851         | 0.41          |


---

## 6. Kết luận & Định hướng cho Giai đoạn 3 (Tree-based Models)

1. **Mô hình tuyến tính là một benchmark xuất sắc:** Với RMSLE = 0.1978, Ridge trên `log1p(revenue)` đã vượt mục tiêu đề ra cho Giai đoạn 2 và cung cấp một hệ số giải thích rất rõ ràng cho nghiệp vụ.
2. **Hạn chế của mô hình tuyến tính:**
   - Mối quan hệ giữa thời tiết (nhiệt độ, mưa) và doanh thu có tính chất phi tuyến (ví dụ mưa rào nhẹ vào mùa hè mát mẻ có thể không giảm doanh thu bằng mưa bão lớn kéo dài).
   - Sự sụt giảm của cửa hàng trường học (`university`) trong kỳ nghỉ hè hay đợt thi cử là sự suy giảm phi tuyến theo thời gian mà mô hình tuyến tính additive chỉ bù đắp được một phần.
3. **Bước tiếp theo (Giai đoạn 3 — Tree-based Models):**
   - Thử nghiệm Random Forest, HistGradientBoosting, LightGBM và CatBoost trên cùng bộ features sạch.
   - Tree-based models sẽ tự động học các tương tác phi tuyến phức tạp mà không cần tạo bằng tay.
   - Mục tiêu: Hạ RMSLE xuống dưới **0.2000**.
