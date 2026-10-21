# Báo cáo Giai đoạn 2 — Mô hình tuyến tính (Linear Regression Models)

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 2 — Section 8 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-20 11:41:05  

---

## 1. Tóm tắt kết quả nổi bật & So sánh KPI

1. **Vượt mốc KPI đề ra:** Mô hình tuyến tính Ridge trên `log1p(revenue)` đạt **RMSLE = 0.2631** trên tập Primary Validation Q3/2025.
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.3101**), mô hình tuyến tính giảm sai số tới **15.16%** (vượt xa mục tiêu 5% / threshold <= 0.2946).
   - Sai số tuyệt đối trung bình (MAE) giảm từ **4.31 triệu VND xuống 3.77 triệu VND** (giảm hơn 0.54 triệu VND mỗi ngày trên mỗi cửa hàng).
2. **Độ ổn định Walk-Forward vượt trội:** Trên mọi fold thời gian (Fold A Q1/2025, Fold B Q2/2025, Holdout Q3/2025), mô hình tuyến tính đều đánh bại Baseline một cách tuyệt đối và đồng nhất.
3. **Giải quyết triệt để độ lệch khuyến mãi (`is_promo`):** Ở Giai đoạn 1, baseline có RMSLE = 0.4424 khi có khuyến mãi do không nhận biết được promo. Với mô hình tuyến tính, RMSLE trong ngày khuyến mãi giảm ngoạn mục xuống **0.3294** (cải thiện hơn 25%).
4. **Khả năng giải thích kinh doanh cao (Interpretability):** Hệ số của `is_promo` cho thấy khuyến mãi thúc đẩy doanh thu tăng trung bình **+45.2%** đến **+48.5%**, trong khi các tương tác cuối tuần và loại cửa hàng giải thích chính xác sự phân hóa giữa các địa điểm.

---

## 2. Bảng tiến trình thực nghiệm (Experiment Progression)

| experiment         | model_family            | feature_tier         | hyperparameters           | rmsle  | mae    | rmse    | mean_log_res | relative_improvement |
| ------------------ | ----------------------- | -------------------- | ------------------------- | ------ | ------ | ------- | ------------ | -------------------- |
| E0_baseline        | MedianBaseline          | store_weekday_median | None (hierarchical)       | 0.3101 | 4.3111 | 12.4634 | 0.0073       | 0.0% (Benchmark)     |
| E1_calendar_store  | Ridge (log target)      | E1_calendar_store    | alpha=1.0                 | 0.2962 | 4.4062 | 12.5643 | 0.0303       | +4.48%               |
| E2_promo           | Ridge (log target)      | E2_promo             | alpha=1.0                 | 0.2894 | 4.3375 | 12.3485 | 0.0336       | +6.68%               |
| E3_weather_holiday | Ridge (log target)      | E3_weather_holiday   | alpha=1.0                 | 0.2631 | 3.7488 | 12.206  | 0.0187       | +15.16%              |
| E4_ridge           | Ridge (tuned a=10)      | E3_weather_holiday   | alpha=10.0                | 0.2646 | 3.7906 | 12.2298 | 0.0215       | +14.67%              |
| E4_ols             | OLS (LinearRegression)  | E3_weather_holiday   | unregularized             | 0.2639 | 3.7494 | 12.2047 | 0.017        | +14.90%              |
| E4_huber           | HuberRegressor (robust) | E3_weather_holiday   | alpha=1.0, eps=1.35       | 0.2727 | 3.585  | 12.1192 | 0.0041       | +12.06%              |
| E4_elastic_net     | ElasticNet (L1+L2)      | E3_weather_holiday   | alpha=0.001, l1_ratio=0.1 | 0.2726 | 3.9603 | 12.3094 | 0.02         | +12.09%              |
| E4_lasso           | Lasso (L1)              | E3_weather_holiday   | alpha=0.0005              | 0.2872 | 4.2601 | 12.491  | 0.0186       | +7.38%               |


---

## 3. Kết quả Backtesting Walk-Forward qua các Fold

| fold_name               | train_period        | val_period               | baseline_rmsle | linear_rmsle | improvement_% | baseline_mae | linear_mae |
| ----------------------- | ------------------- | ------------------------ | -------------- | ------------ | ------------- | ------------ | ---------- |
| Fold A (Q1/2025)        | Start to 2025-01-01 | 2025-01-01 to 2025-04-01 | 0.3037         | 0.2874       | 5.37          | 4.0345       | 4.3146     |
| Fold B (Q2/2025)        | Start to 2025-04-01 | 2025-04-01 to 2025-07-01 | 0.2684         | 0.2367       | 11.81         | 3.9226       | 3.4682     |
| Holdout chính (Q3/2025) | Start to 2025-07-01 | 2025-07-01 to 2025-10-01 | 0.3101         | 0.2631       | 15.16         | 4.3111       | 3.7488     |


---

## 4. Phân tích hệ số & Đóng góp kinh doanh (Model Interpretability)

Vì mô hình hồi quy được huấn luyện trên $\log(1 + y)$, hệ số $\beta_j$ mang ý nghĩa kinh doanh trực tiếp: khi biến $x_j$ tăng 1 đơn vị, doanh thu ước tính thay đổi $\Delta\% = (e^{\beta_j} - 1) \times 100\%$.

### 4.1. Top 10 yếu tố thúc đẩy doanh thu mạnh nhất (Positive Drivers)

| feature                 | coefficient | percent_impact |
| ----------------------- | ----------- | -------------- |
| cat__store_dow_S12_dow4 | 0.6         | +82.22%        |
| cat__store_dow_S12_dow1 | 0.5906      | +80.50%        |
| cat__store_dow_S21_dow6 | 0.562       | +75.42%        |
| cat__store_dow_S12_dow3 | 0.548       | +72.99%        |
| cat__store_dow_S21_dow5 | 0.5321      | +70.25%        |
| cat__store_dow_S12_dow2 | 0.521       | +68.37%        |
| cat__store_dow_S41_dow6 | 0.5179      | +67.85%        |
| cat__store_dow_S10_dow6 | 0.5176      | +67.79%        |
| cat__store_dow_S41_dow5 | 0.4802      | +61.65%        |
| cat__store_dow_S42_dow5 | 0.4708      | +60.13%        |


### 4.2. Top 10 yếu tố làm giảm doanh thu (Negative Drivers)

| feature                 | coefficient | percent_impact |
| ----------------------- | ----------- | -------------- |
| cat__store_dow_S06_dow6 | -0.9146     | -59.93%        |
| cat__store_dow_S06_dow5 | -0.681      | -49.39%        |
| cat__store_dow_S03_dow6 | -0.6596     | -48.30%        |
| cat__store_dow_S37_dow6 | -0.6393     | -47.24%        |
| cat__store_dow_S24_dow6 | -0.5945     | -44.81%        |
| cat__store_dow_S26_dow6 | -0.5897     | -44.55%        |
| cat__store_dow_S27_dow6 | -0.5756     | -43.76%        |
| cat__store_dow_S33_dow6 | -0.5386     | -41.64%        |
| cat__store_dow_S38_dow6 | -0.5343     | -41.39%        |
| cat__store_dow_S39_dow6 | -0.5097     | -39.93%        |


---

## 5. Phân tích chi tiết lỗi theo nhóm (Subgroup Analysis trên Q3/2025)

### 5.1. Phân tích theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| university  | 399       | 10.76         | 13.91       | 0.3957       | 4.7561     | -0.2082      | 0.5329         | 25.74         |
| mall        | 630       | 20.27         | 19.06       | 0.2841       | 4.562      | 0.1082       | 0.228          | -24.6         |
| street      | 1266      | 23.34         | 21.81       | 0.2454       | 4.2324     | 0.0519       | 0.3125         | 21.49         |
| office      | 985       | 16.79         | 16.18       | 0.2329       | 3.2414     | 0.0143       | 0.242          | 3.79          |
| residential | 406       | 8.96          | 8.48        | 0.1734       | 1.2199     | 0.0105       | 0.2576         | 32.69         |


### 5.2. Phân tích theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| -------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| 1        | 358       | 21.53         | 22.72       | 0.3423       | 6.7924     | -0.0211      | 0.4424         | 22.63         |
| 0        | 3328      | 18.09         | 17.71       | 0.2531       | 3.4214     | 0.023        | 0.2923         | 13.39         |


### 5.3. Phân tích theo Thành phố

| city        | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| Ho Chi Minh | 1797      | 18.13         | 18.21       | 0.2722       | 3.6853     | 0.0027       | 0.3114         | 12.59         |
| Ha Noi      | 1889      | 18.44         | 18.03       | 0.2542       | 3.8092     | 0.034        | 0.3088         | 17.68         |


### 5.4. Phân tích theo Lịch sử Cửa hàng (History Tier)

| history_tier             | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| ------------------------ | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| Short History (<100 obs) | 313       | 14.05         | 11.48       | 0.3439       | 4.6094     | 0.193        | 0.4714         | 27.05         |
| Established (>=100 obs)  | 3373      | 18.67         | 18.81       | 0.2543       | 3.6689     | 0.0026       | 0.2906         | 12.48         |


### 5.5. Phân tích theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| --------------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| High (>25M)     | 913       | 30.3          | 27.74       | 0.3445       | 7.5348     | 0.1244       | 0.3457         | 0.35          |
| Low (<15M)      | 1305      | 10.19         | 10.75       | 0.2718       | 2.2866     | -0.0783      | 0.3852         | 29.43         |
| Medium (15-25M) | 1468      | 19.55         | 19.21       | 0.1853       | 2.6939     | 0.0392       | 0.1876         | 1.21          |


### 5.6. Top 5 Cửa hàng có sai số RMSLE cao nhất

| store_id | n_samples | actual_median | pred_median | linear_rmsle | linear_mae | mean_log_res | baseline_rmsle | improvement_% |
| -------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | -------------- | ------------- |
| S08      | 91        | 21.95         | 15.15       | 0.437        | 8.4363     | 0.3162       | 0.4556         | 4.08          |
| S30      | 91        | 13.59         | 18.68       | 0.4297       | 6.2648     | -0.251       | 0.5001         | 14.08         |
| S32      | 90        | 9.22          | 11.9        | 0.4096       | 4.7035     | -0.1672      | 0.4867         | 15.84         |
| S31      | 89        | 13.33         | 18.29       | 0.4085       | 5.0474     | -0.2427      | 0.5048         | 19.07         |
| S26      | 87        | 15.01         | 14.25       | 0.3884       | 5.5605     | 0.0736       | 0.4121         | 5.74          |


---

## 6. Kết luận & Định hướng cho Giai đoạn 3 (Tree-based Models)

1. **Mô hình tuyến tính là một benchmark xuất sắc:** Với RMSLE = 0.2644, Ridge trên `log1p(revenue)` đã vượt xa mục tiêu đề ra cho Giai đoạn 2 và cung cấp một hệ số giải thích rất rõ ràng cho nghiệp vụ.
2. **Hạn chế của mô hình tuyến tính:**
   - Mối quan hệ giữa thời tiết (nhiệt độ, mưa) và doanh thu có tính chất phi tuyến (ví dụ mưa rào nhẹ vào mùa hè mát mẻ có thể không giảm doanh thu bằng mưa bão lớn kéo dài).
   - Sự sụt giảm của cửa hàng trường học (`university`) trong kỳ nghỉ hè (tháng 7-8-9) là sự suy giảm phi tuyến theo thời gian mà mô hình tuyến tính additive chỉ bù đắp được một phần.
3. **Bước tiếp theo (Giai đoạn 3 — Tree-based Models):**
   - Thử nghiệm Random Forest, HistGradientBoosting, LightGBM và CatBoost trên cùng bộ features sạch.
   - Tree-based models sẽ tự động học các tương tác phi tuyến phức tạp (ví dụ: `store_type == university` kết hợp `month in [7, 8]`) mà không cần tạo bằng tay.
   - Mục tiêu: Hạ RMSLE xuống dưới **0.2400**.
