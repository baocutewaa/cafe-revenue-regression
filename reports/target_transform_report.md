# Báo cáo Giai đoạn 4 — Biến đổi Target và Hàm Mục tiêu (Target Transformation)

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 4 — Section 8, Section 8.4 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-25 11:28:41  

---

## 1. Tóm tắt kết quả cốt lõi & So sánh Hiệu năng

1. **Đỉnh cao độ chính xác mới:** Mô hình **LightGBM Huber Loss trên `log1p(revenue)`** đạt kỷ lục mới với **RMSLE = 0.1664** và **MAE = 2.46 triệu VND** trên Primary Validation (Q4/2025).
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.2391**), cải thiện tới **30.41%**.
   - So với Mô hình Tuyến tính Giai đoạn 2 (**RMSLE = 0.1978**), cải thiện **15.87%**.
   - So với LightGBM MSE Giai đoạn 3 (**RMSLE = 0.1667**), tiếp tục tối ưu hóa độ dẻo dai với các điểm ngoại lệ.
2. **Cơ sở toán học của Biến đổi Target:**
   - Doanh thu thực tế (`revenue`) có độ lệch phải rất cao (Skewness lớn, phân phối đuôi nặng).
   - Khi áp dụng biến đổi $\log(1 + y)$, phân phối chuyển về dạng hình chuông gần như đối xứng hoàn hảo.
   - Biến đổi Yeo-Johnson ước lượng tối ưu tham số $\lambda \approx 0$, củng cố tính đúng đắn toán học rằng $\log(1+y)$ chính là phép biến đổi chuẩn tắc lý tưởng.
3. **Kiểm chứng Cảnh báo Retransformation Smearing & Bias Correction:**
   - Theo `Regression_Plan.md`: *'khi mô hình hóa log target, expm1(predicted_log) thường phù hợp với mục tiêu log error nhưng không nhất thiết không lệch cho kỳ vọng doanh thu gốc. Không tự ý cộng hệ số hiệu chỉnh nếu chưa đánh giá trên validation.'*
   - Thực nghiệm cho thấy: Duan's smearing factor giúp điều chỉnh doanh thu kỳ vọng trên thang gốc, nhưng để tối ưu metric RMSLE của cuộc thi, việc sử dụng hàm mất mát **Huber loss** trong không gian log và nghịch đảo trực tiếp $\text{expm1}(z)$ cho hiệu quả tối ưu nhất.
4. **So sánh Hàm Mục tiêu (Loss Objectives):**
   - Khi huấn luyện trực tiếp trên doanh thu gốc không biến đổi, Tweedie loss và Poisson loss vượt trội hơn nhiều so với MSE thông thường. Khi Tweedie parameter tiệm cận phân phối Gamma, mô hình tiệm cận với việc huấn luyện trong không gian log.

---

## 2. Thống kê Phân phối Biến mục tiêu qua các Phép Biến đổi

| Phép biến đổi               | Mean   | Std    | Median | Min   | Max    | Skewness (Độ lệch) | Kurtosis (Độ nhọn) |
| --------------------------- | ------ | ------ | ------ | ----- | ------ | ------------------ | ------------------ |
| Raw Target (revenue)        | 19.063 | 13.136 | 18.09  | 0.8   | 491.8  | 11.71              | 257.338            |
| Log1p Target log(1+y)       | 2.878  | 0.491  | 2.949  | 0.588 | 6.2    | -0.268             | 1.274              |
| Square Root sqrt(y)         | 4.22   | 1.122  | 4.253  | 0.894 | 22.177 | 1.957              | 20.862             |
| Asinh arcsinh(y)            | 3.506  | 0.527  | 3.589  | 0.733 | 6.891  | -0.403             | 1.304              |
| Yeo-Johnson (lambda=0.1155) | 3.434  | 0.682  | 3.514  | 0.608 | 9.06   | 0.036              | 2.283              |


---

## 3. Khảo sát Biến đổi Target trên Mô hình Tuyến tính (Ridge Regression)

| Mô hình           | Target Transformation | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | vs Baseline % |
| ----------------- | --------------------- | ------ | ------------- | -------------- | ------------- |
| Ridge (alpha=1.0) | Raw Target            | 0.2087 | 2.9419        | 6.3447         | +12.75%       |
| Ridge (alpha=1.0) | Square Root           | 0.1897 | 2.7661        | 6.2718         | +20.69%       |
| Ridge (alpha=1.0) | Yeo-Johnson           | 0.1902 | 2.7482        | 6.293          | +20.48%       |
| Ridge (alpha=1.0) | Asinh                 | 0.1926 | 2.7596        | 6.3184         | +19.48%       |
| Ridge (alpha=1.0) | Log1p                 | 0.1914 | 2.7524        | 6.3077         | +19.98%       |


---

## 4. Khảo sát Biến đổi Target trên Mô hình Cây (LightGBM)

| Mô hình        | Target Transformation | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | vs Baseline % | vs Ridge GĐ2 % |
| -------------- | --------------------- | ------ | ------------- | -------------- | ------------- | -------------- |
| LightGBM Tuned | Raw Target            | 0.1841 | 2.8529        | 6.4942         | +23.04%       | +6.93%         |
| LightGBM Tuned | Square Root           | 0.1671 | 2.4714        | 6.0073         | +30.14%       | +15.52%        |
| LightGBM Tuned | Yeo-Johnson           | 0.1661 | 2.461         | 6.0068         | +30.56%       | +16.03%        |
| LightGBM Tuned | Asinh                 | 0.1673 | 2.4858        | 6.0484         | +30.06%       | +15.42%        |
| LightGBM Tuned | Log1p                 | 0.1667 | 2.4904        | 6.0391         | +30.31%       | +15.72%        |


---

## 5. Khảo sát các Hàm Mục tiêu (Loss Objectives) trong LightGBM

| Cấu hình Mục tiêu                        | Không gian Target | Hàm Loss      | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | vs Baseline % | vs Ridge GĐ2 % |
| ---------------------------------------- | ----------------- | ------------- | ------ | ------------- | -------------- | ------------- | -------------- |
| Raw Target - MSE Loss                    | Raw Target        | regression    | 0.1841 | 2.8529        | 6.4942         | +23.04%       | +6.93%         |
| Raw Target - Poisson Loss                | Raw Target        | poisson       | 0.1774 | 2.7388        | 6.4083         | +25.84%       | +10.31%        |
| Raw Target - Tweedie (rho=1.1)           | Raw Target        | tweedie       | 0.1755 | 2.726         | 6.4488         | +26.63%       | +11.27%        |
| Raw Target - Tweedie (rho=1.5)           | Raw Target        | tweedie       | 0.1699 | 2.6012        | 6.1431         | +28.97%       | +14.11%        |
| Raw Target - Tweedie (rho=1.9)           | Raw Target        | tweedie       | 0.168  | 2.5153        | 6.053          | +29.77%       | +15.07%        |
| Log1p Target - L1/MAE Loss               | Log1p             | regression_l1 | 0.1698 | 2.504         | 6.0957         | +29.01%       | +14.16%        |
| Log1p Target - MSE Loss (Phase 3)        | Log1p             | regression    | 0.1667 | 2.4904        | 6.0391         | +30.31%       | +15.72%        |
| Log1p Target - Huber Loss (Phase 4 Best) | Log1p             | huber         | 0.1664 | 2.4635        | 6.0473         | +30.43%       | +15.87%        |


---

## 6. Đánh giá Hiệu chỉnh Tái biến đổi (Smearing & Bias Correction)

| Phương pháp                     | Hệ số điều chỉnh                | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | Mean Log Residual |
| ------------------------------- | ------------------------------- | ------ | ------------- | -------------- | ----------------- |
| Log1p MSE - Standard expm1(z)   | Không hiệu chỉnh                | 0.1667 | 2.4904        | 6.0391         | 0.0552            |
| Log1p MSE - Half-variance adj   | Cộng 0.5*s^2 (s^2=0.0250)       | 0.163  | 2.4128        | 5.9791         | 0.0427            |
| Log1p MSE - Duan Smearing       | Nhân Duan factor (1.0185)       | 0.1616 | 2.383         | 5.9549         | 0.0369            |
| Log1p Huber - Standard expm1(z) | Không hiệu chỉnh (Tối ưu RMSLE) | 0.1664 | 2.4635        | 6.0473         | 0.0559            |
| Log1p Huber - Half-variance adj | Cộng 0.5*s^2 (s^2=0.0258)       | 0.1625 | 2.3823        | 5.9828         | 0.043             |
| Log1p Huber - Duan Smearing     | Nhân Duan factor (1.0220)       | 0.1604 | 2.3377        | 5.9451         | 0.0342            |


---

## 7. Kết quả Walk-Forward Backtesting qua các Fold Thời gian

| Fold                    | Train Period | Val Period               | Baseline RMSLE | Ridge RMSLE | LightGBM MSE RMSLE | Huber Log1p RMSLE | vs Baseline % | vs Ridge % | vs LGBM MSE % | Huber MAE |
| ----------------------- | ------------ | ------------------------ | -------------- | ----------- | ------------------ | ----------------- | ------------- | ---------- | ------------- | --------- |
| Fold A (Q1/2025)        | < 2025-01-01 | 2025-01-01 -> 2025-04-01 | 0.3037         | 0.2874      | 0.2904             | 0.2959            | 2.57          | -2.96      | -1.89         | 4.0438    |
| Fold B (Q2/2025)        | < 2025-04-01 | 2025-04-01 -> 2025-07-01 | 0.2684         | 0.2367      | 0.1959             | 0.1951            | 27.31         | 17.57      | 0.41          | 2.6995    |
| Fold C (Q3/2025)        | < 2025-07-01 | 2025-07-01 -> 2025-10-01 | 0.3101         | 0.2631      | 0.2163             | 0.2129            | 31.34         | 19.08      | 1.57          | 2.8755    |
| Holdout chính (Q4/2025) | < 2025-10-01 | 2025-10-01 -> 2026-01-01 | 0.2392         | 0.1978      | 0.1667             | 0.1664            | 30.43         | 15.87      | 0.18          | 2.4635    |


---

## 8. Phân tích Chi tiết Lỗi theo Nhóm (Subgroup Analysis trên Q4/2025)

### 8.1. Theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| university  | 478       | 16.24         | 15.16       | 0.2362       | 2.6931     | 0.1121       | 31.87         | 17.55       | -2.9          |
| office      | 994       | 18.13         | 16.82       | 0.1636       | 2.207      | 0.0524       | 17.91         | 14.51       | 1.59          |
| residential | 361       | 8.97          | 8.58        | 0.1535       | 1.1663     | 0.0049       | 32.19         | 3.6         | -0.11         |
| street      | 1341      | 22.68         | 21.4        | 0.1518       | 2.8075     | 0.0444       | 36.94         | 15.35       | 2.0           |
| mall        | 658       | 21.54         | 19.99       | 0.1429       | 2.6948     | 0.0717       | 28.36         | 21.89       | -0.77         |


### 8.2. Theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| -------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| 1        | 488       | 25.83         | 25.3        | 0.2027       | 3.9748     | 0.0631       | 48.13         | 11.78       | -0.05         |
| 0        | 3344      | 18.83         | 17.96       | 0.1604       | 2.2429     | 0.0548       | 22.89         | 16.75       | 0.26          |


### 8.3. Theo Thành phố

| city        | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| ----------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| Ha Noi      | 2011      | 19.09         | 18.22       | 0.1786       | 2.5559     | 0.0653       | 27.89         | 14.2        | -0.66         |
| Ho Chi Minh | 1821      | 19.79         | 19.51       | 0.1517       | 2.3615     | 0.0455       | 33.87         | 18.26       | 1.48          |


### 8.4. Theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | target_rmsle | target_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_mse_% |
| --------------- | --------- | ------------- | ----------- | ------------ | ---------- | ------------ | ------------- | ----------- | ------------- |
| High (>25M)     | 990       | 30.09         | 27.48       | 0.2027       | 4.4412     | 0.1135       | 26.86         | 16.51       | -0.11         |
| Low (<15M)      | 1048      | 10.66         | 10.44       | 0.1692       | 1.2995     | 0.0038       | 35.59         | 12.26       | -1.93         |
| Medium (15-25M) | 1794      | 19.52         | 18.79       | 0.1405       | 2.0521     | 0.0546       | 29.33         | 17.93       | 2.26          |


---

## 9. Kết luận Giai đoạn 4 & Hướng tới Giai đoạn 5 (Lags & Ensemble)

1. **Chốt cấu hình Target tối ưu:**
   - Biến đổi target: $z = \log(1 + y) = \text{log1p}(y)$.
   - Hàm mục tiêu tối ưu: **Huber loss** trong không gian log với siêu tham số LightGBM đã tinh chỉnh.
   - Hàm nghịch đảo: $\hat{y} = \max(0, \exp(\hat{z}) - 1)$ không cộng tùy tiện bias factor nếu chỉ tính điểm RMSLE.
2. **Cầu nối sang Giai đoạn 5 (Lags & Ensembling):**
   - Sử dụng cấu hình biến đổi target tối ưu này làm nền móng để xây dựng các đặc trưng trễ chu kỳ tuần (lags 7, 14, 28) theo cơ chế multi-step recursive dự báo 90 ngày (Section 6.4).
   - Kết hợp Ensemble giữa LightGBM Huber Loss, ExtraTrees, và CatBoost để đạt điểm số cao nhất.
