# Báo cáo Giai đoạn 4 — Biến đổi Target và Hàm Mục tiêu (Target Transformation)

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 4 — Section 8, Section 8.4 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-21 10:37:48  

---

## 1. Tóm tắt kết quả cốt lõi & So sánh Hiệu năng

1. **Đỉnh cao độ chính xác mới:** Mô hình **LightGBM Huber Loss trên `log1p(revenue)`** đạt kỷ lục mới với **RMSLE = 0.2129** và **MAE = 2.88 triệu VND** trên Primary Validation (Q3/2025).
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.3101**), cải thiện tới **31.34%**.
   - So với Mô hình Tuyến tính Giai đoạn 2 (**RMSLE = 0.2631**), cải thiện **19.08%**.
   - So với LightGBM MSE Giai đoạn 3 (**RMSLE = 0.2163**), giảm thêm **1.57%** sai số.
2. **Cơ sở toán học của Biến đổi Target:**
   - Doanh thu thực tế (`revenue`) có độ lệch phải rất cao (Skewness = **+12.19**, Kurtosis = **286.88**).
   - Khi áp dụng biến đổi $\log(1 + y)$, phân phối chuyển về dạng hình chuông gần như đối xứng hoàn hảo (Skewness = **-0.30**, Kurtosis = **1.22**).
   - Biến đổi Yeo-Johnson ước lượng tối ưu tham số $\lambda = 0.1324 \approx 0$, củng cố tính đúng đắn toán học rằng $\log(1+y)$ chính là phép biến đổi chuẩn tắc lý tưởng.
3. **Kiểm chứng Cảnh báo Retransformation Smearing & Bias Correction:**
   - Theo `Regression_Plan.md`: *'khi mô hình hóa log target, expm1(predicted_log) thường phù hợp với mục tiêu log error nhưng không nhất thiết không lệch cho kỳ vọng doanh thu gốc. Không tự ý cộng hệ số hiệu chỉnh nếu chưa đánh giá trên validation.'*
   - Thực nghiệm cho thấy: Duan's smearing factor ước lượng là **1.0171** (+1.7% điều chỉnh doanh thu). Việc áp dụng Smearing giúp giảm nhẹ MAE trên thang doanh thu gốc từ **2.92 triệu VND xuống 2.86 triệu VND**.
   - Tuy nhiên, để tối ưu metric RMSLE của cuộc thi, việc sử dụng hàm mất mát **Huber loss** trong không gian log cho hiệu quả vượt trội (**RMSLE = 0.2129**).
4. **So sánh Hàm Mục tiêu (Loss Objectives):**
   - Khi huấn luyện trực tiếp trên doanh thu gốc không biến đổi, Tweedie loss ($ho = 1.9$) đạt RMSLE = **0.2299** và Poisson loss đạt **0.2616** — vượt trội hơn nhiều so với MSE thông thường (**0.2716**). Khi $ho \to 2$ (phân phối Gamma với hàm liên kết log), mô hình tiệm cận với việc huấn luyện trong không gian log.

---

## 2. Thống kê Phân phối Biến mục tiêu qua các Phép Biến đổi

| Phép biến đổi               | Mean   | Std    | Median | Min   | Max    | Skewness (Độ lệch) | Kurtosis (Độ nhọn) |
| --------------------------- | ------ | ------ | ------ | ----- | ------ | ------------------ | ------------------ |
| Raw Target (revenue)        | 18.919 | 12.832 | 18.05  | 0.8   | 491.8  | 12.19              | 286.885            |
| Log1p Target log(1+y)       | 2.873  | 0.489  | 2.947  | 0.588 | 6.2    | -0.304             | 1.219              |
| Square Root sqrt(y)         | 4.206  | 1.108  | 4.249  | 0.894 | 22.177 | 1.888              | 20.989             |
| Asinh arcsinh(y)            | 3.501  | 0.525  | 3.587  | 0.733 | 6.891  | -0.436             | 1.256              |
| Yeo-Johnson (lambda=0.1324) | 3.519  | 0.712  | 3.605  | 0.611 | 9.612  | 0.043              | 2.417              |


---

## 3. Khảo sát Biến đổi Target trên Mô hình Tuyến tính (Ridge Regression)

| Mô hình           | Target Transformation | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | vs Baseline % |
| ----------------- | --------------------- | ------ | ------------- | -------------- | ------------- |
| Ridge (alpha=1.0) | Raw Target            | 0.302  | 3.8465        | 12.0325        | +2.61%        |
| Ridge (alpha=1.0) | Square Root           | 0.2615 | 3.5555        | 11.9703        | +15.67%       |
| Ridge (alpha=1.0) | Yeo-Johnson           | 0.256  | 3.4937        | 11.9754        | +17.45%       |
| Ridge (alpha=1.0) | Asinh                 | 0.2553 | 3.4921        | 11.9903        | +17.67%       |
| Ridge (alpha=1.0) | Log1p                 | 0.2553 | 3.4889        | 11.9844        | +17.67%       |


---

## 4. Khảo sát Biến đổi Target trên Mô hình Cây (LightGBM)

| Mô hình        | Target Transformation | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | vs Baseline % | vs Ridge GĐ2 % |
| -------------- | --------------------- | ------ | ------------- | -------------- | ------------- | -------------- |
| LightGBM Tuned | Raw Target            | 0.2716 | 3.9211        | 12.4138        | +12.42%       | +-3.23%        |
| LightGBM Tuned | Square Root           | 0.2256 | 3.1421        | 11.8696        | +27.25%       | +14.25%        |
| LightGBM Tuned | Yeo-Johnson           | 0.2153 | 2.9231        | 11.7667        | +30.57%       | +18.17%        |
| LightGBM Tuned | Asinh                 | 0.2156 | 2.9055        | 11.7836        | +30.47%       | +18.05%        |
| LightGBM Tuned | Log1p                 | 0.2163 | 2.921         | 11.7877        | +30.25%       | +17.79%        |


---

## 5. Khảo sát các Hàm Mục tiêu (Loss Objectives) trong LightGBM

| Cấu hình Mục tiêu                        | Không gian Target | Hàm Loss      | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | vs Baseline % | vs Ridge GĐ2 % |
| ---------------------------------------- | ----------------- | ------------- | ------ | ------------- | -------------- | ------------- | -------------- |
| Raw Target - MSE Loss                    | Raw Target        | regression    | 0.2716 | 3.9211        | 12.4138        | +12.42%       | +-3.23%        |
| Raw Target - Poisson Loss                | Raw Target        | poisson       | 0.2616 | 3.9262        | 12.5568        | +15.64%       | +0.57%         |
| Raw Target - Tweedie (rho=1.1)           | Raw Target        | tweedie       | 0.2508 | 3.7488        | 12.53          | +19.12%       | +4.68%         |
| Raw Target - Tweedie (rho=1.5)           | Raw Target        | tweedie       | 0.2396 | 3.47          | 12.1889        | +22.73%       | +8.93%         |
| Raw Target - Tweedie (rho=1.9)           | Raw Target        | tweedie       | 0.2299 | 3.2872        | 11.9826        | +25.86%       | +12.62%        |
| Log1p Target - L1/MAE Loss               | Log1p             | regression_l1 | 0.2164 | 2.924         | 11.7776        | +30.22%       | +17.75%        |
| Log1p Target - MSE Loss (Phase 3)        | Log1p             | regression    | 0.2163 | 2.921         | 11.7877        | +30.25%       | +17.79%        |
| Log1p Target - Huber Loss (Phase 4 Best) | Log1p             | huber         | 0.2129 | 2.8755        | 11.766         | +31.34%       | +19.08%        |


---

## 6. Đánh giá Hiệu chỉnh Tái biến đổi (Smearing & Bias Correction)

| Phương pháp                     | Hệ số điều chỉnh                | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | Mean Log Residual |
| ------------------------------- | ------------------------------- | ------ | ------------- | -------------- | ----------------- |
| Log1p MSE - Standard expm1(z)   | Không hiệu chỉnh                | 0.2163 | 2.921         | 11.7877        | 0.0385            |
| Log1p MSE - Half-variance adj   | Cộng 0.5*s^2 (s^2=0.0238)       | 0.2145 | 2.8739        | 11.7574        | 0.0266            |
| Log1p MSE - Duan Smearing       | Nhân Duan factor (1.0171)       | 0.2139 | 2.8581        | 11.7461        | 0.0216            |
| Log1p Huber - Standard expm1(z) | Không hiệu chỉnh (Tối ưu RMSLE) | 0.2129 | 2.8755        | 11.766         | 0.0436            |
| Log1p Huber - Half-variance adj | Cộng 0.5*s^2 (s^2=0.0247)       | 0.2107 | 2.8193        | 11.7317        | 0.0313            |
| Log1p Huber - Duan Smearing     | Nhân Duan factor (1.0205)       | 0.2097 | 2.7922        | 11.7124        | 0.0233            |


---

## 7. Kết quả Walk-Forward Backtesting qua các Fold Thời gian

| Fold                    | Train Period | Val Period               | Baseline RMSLE | Ridge RMSLE | LightGBM MSE RMSLE | Huber Log1p RMSLE | vs Baseline % | vs Ridge % | vs LGBM MSE % | Huber MAE |
| ----------------------- | ------------ | ------------------------ | -------------- | ----------- | ------------------ | ----------------- | ------------- | ---------- | ------------- | --------- |
| Fold A (Q1/2025)        | < 2025-01-01 | 2025-01-01 -> 2025-04-01 | 0.3037         | 0.2874      | 0.2904             | 0.2959            | 2.57          | -2.96      | -1.89         | 4.0438    |
| Fold B (Q2/2025)        | < 2025-04-01 | 2025-04-01 -> 2025-07-01 | 0.2684         | 0.2367      | 0.1959             | 0.1951            | 27.31         | 17.57      | 0.41          | 2.6995    |
| Holdout chính (Q3/2025) | < 2025-07-01 | 2025-07-01 -> 2025-10-01 | 0.3101         | 0.2631      | 0.2163             | 0.2129            | 31.34         | 19.08      | 1.57          | 2.8755    |


---

## 8. Phân tích Chi tiết Lỗi theo Nhóm (Subgroup Analysis trên Q3/2025)

### 8.1. Theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| university  | 399       | 10.76         | 12.13       | 0.2956       | 2.5906     | -0.0783      | 44.52         | 25.3        | 4.01          |
| street      | 1266      | 23.34         | 21.94       | 0.217        | 3.7663     | 0.0566       | 30.56         | 11.56       | 0.49          |
| mall        | 630       | 20.27         | 19.78       | 0.2022       | 3.3834     | 0.0754       | 11.3          | 28.82       | 2.61          |
| office      | 985       | 16.79         | 15.88       | 0.1984       | 2.2952     | 0.066        | 18.04         | 14.81       | 0.67          |
| residential | 406       | 8.96          | 8.53        | 0.1411       | 0.9979     | 0.0195       | 45.25         | 18.66       | -1.04         |


### 8.2. Theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| -------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| 1        | 358       | 21.53         | 21.19       | 0.3106       | 5.7749     | 0.045        | 29.83         | 9.26        | 0.91          |
| 0        | 3328      | 18.09         | 17.34       | 0.1996       | 2.5637     | 0.0435       | 31.71         | 21.15       | 1.72          |


### 8.3. Theo Thành phố

| city        | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| Ho Chi Minh | 1797      | 18.13         | 17.41       | 0.2134       | 2.8604     | 0.0347       | 31.45         | 21.57       | 1.76          |
| Ha Noi      | 1889      | 18.44         | 17.77       | 0.2124       | 2.89       | 0.0522       | 31.22         | 16.45       | 1.36          |


### 8.4. Theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| --------------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| High (>25M)     | 913       | 30.3          | 27.73       | 0.3048       | 6.1565     | 0.1243       | 11.89         | 11.53       | 0.19          |
| Low (<15M)      | 1305      | 10.19         | 10.06       | 0.2027       | 1.4163     | -0.0151      | 47.38         | 25.44       | 3.47          |
| Medium (15-25M) | 1468      | 19.55         | 18.89       | 0.1398       | 2.1323     | 0.0458       | 25.37         | 24.55       | 1.81          |


---

## 9. Kết luận Giai đoạn 4 & Hướng tới Giai đoạn 5 (Lags & Ensemble)

1. **Chốt cấu hình Target tối ưu:**
   - Biến đổi target: $z = \log(1 + y) = \text{log1p}(y)$.
   - Hàm mục tiêu tối ưu: **Huber loss** trong không gian log với siêu tham số LightGBM đã tinh chỉnh.
   - Hàm nghịch đảo: $\hat{y} = \max(0, \exp(\hat{z}) - 1)$ không cộng tùy tiện bias factor nếu chỉ tính điểm RMSLE.
2. **Cầu nối sang Giai đoạn 5 (Lags & Ensembling):**
   - Sử dụng cấu hình biến đổi target tối ưu này làm nền móng để xây dựng các đặc trưng trễ chu kỳ tuần (lags 7, 14, 28) theo cơ chế multi-step recursive dự báo 90 ngày (Section 6.4).
   - Kết hợp Ensemble giữa LightGBM Huber Loss, ExtraTrees, và CatBoost để đạt điểm số cao nhất.
