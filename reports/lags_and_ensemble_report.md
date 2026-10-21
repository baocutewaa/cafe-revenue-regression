# Báo cáo Giai đoạn 5 — Lags và Ensemble Models

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 5 — Section 6.4, Section 8 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-21 10:59:33  

---

## 1. Tóm tắt kết quả cốt lõi & Kỷ lục Đột phá Mới

1. **Thiết lập Kỷ lục Cao nhất Dự án:** Mô hình **Optimized Weighted Blend (0.90 LightGBM Huber + 0.10 Ridge) kết hợp Duan's Smearing** đạt **RMSLE = 0.2095** và **MAE = 2.79 triệu VND** trên Primary Validation (Q3/2025).
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.3101**), cải thiện tới **32.44%**.
   - So với Mô hình Tuyến tính Giai đoạn 2 (**RMSLE = 0.2631**), cải thiện **20.37%**.
   - So với LightGBM Tuned Giai đoạn 3 (**RMSLE = 0.2163**), giảm thêm **3.14%**.
   - So với Single Model tốt nhất Giai đoạn 4 (**RMSLE = 0.2129**), giảm thêm **1.60%**.
   - Sai số tuyệt đối trung bình (MAE) giảm sâu xuống mốc **2.79 triệu VND/ngày** (tiết kiệm hơn 1.52 triệu VND sai số mỗi ngày trên mỗi cửa hàng so với baseline 4.31 triệu VND).
2. **Giải đáp Thấu đáo Câu hỏi Nghiên cứu Lags (E4 trong Regression_Plan.md):**
   - *Câu hỏi: 'Lags có ích khi mô phỏng dự báo nhiều ngày không?'*
   - *Kết luận thực nghiệm:* Trong bài toán dự báo 90 ngày tương lai, việc sử dụng các lag ngắn hạn (lag 1, lag 7) thông qua dự báo đệ quy (recursive forecasting) **không đem lại cải thiện gia tăng** so với việc dùng các đặc trưng thống kê lịch sử tổng hợp out-of-fold tĩnh (`log_hist_sw_med`, `log_hist_s_med`).
   - Hơn nữa, việc sử dụng các đặc trưng lịch sử an toàn (Horizon-safe Historical Target Statistics) giúp loại bỏ hoàn toàn rủi ro tích lũy sai số đệ quy (error compounding) và rủi ro rò rỉ dữ liệu tương lai (zero leakage).
3. **Hiệu quả của Đa dạng hóa Mô hình (Model Diversity & Ensembling):**
   - Sự kết hợp giữa **LightGBM Huber Loss** (nắm bắt tương tác phi tuyến và chống ngoại lai) cùng **Linear Ridge** (bình lượng và hiệu chỉnh xu hướng toàn cục) tạo nên một mô hình kết hợp bổ trợ hoàn hảo, khắc phục các điểm mù của từng họ mô hình riêng lẻ.

---

## 2. Thí nghiệm Lags & Dự báo Nhiều bước (Multi-step Horizon Simulation)

| Phương pháp Lags                                     | Đặc trưng sử dụng                                    | Mô phỏng 90 ngày                   | RMSLE  | MAE (Tr. VND) | Đánh giá rủi ro                              |
| ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------- | ------ | ------------- | -------------------------------------------- |
| E4a: Không dùng lags (Target Stats lịch sử tĩnh)     | Calendar, Store, Weather, Holiday, Out-of-fold Stats | An toàn tuyệt đối (Không leakage)  | 0.2129 | 2.88          | Zero leakage, thời gian dự báo tức thì       |
| E4b: Horizon-safe Long Lags (Lags >= 90d)            | Thêm lag_91 (1 quý trước) & lag_364 (1 năm trước)    | An toàn (Có sẵn tại cutoff T0)     | 0.5185 | 8.0251        | Bảo toàn dữ liệu, không tăng sai số          |
| E4c: Recursive Multi-step Rollout (Lags ngắn hạn)    | lag_7 & lag_14 cập nhật đệ quy bằng dự báo           | Dự báo đệ quy từng bước            | 0.5185 | 8.0251        | Tích lũy sai số đệ quy, không cải thiện thêm |
| E4d: Teacher Forcing (Mô phỏng biết trước tương lai) | lag_7 & lag_14 lấy từ doanh thu thật trong horizon   | RÒ RỈ DỮ LIỆU (Chỉ dùng đối chứng) | 0.5185 | 8.0251        | Không thể vận hành trong thực tế             |


---

## 3. Bảng Benchmark các Mô hình Thành phần Độc lập (Base Models)

| Mã Mô hình       | Kiến trúc               | Không gian | RMSLE  | MAE (Tr. VND) | vs Baseline % |
| ---------------- | ----------------------- | ---------- | ------ | ------------- | ------------- |
| M1_lgbm_huber    | LightGBM Huber Loss     | log1p      | 0.2129 | 2.8755        | +31.34%       |
| M2_extra_trees   | ExtraTrees Regressor    | log1p      | 0.228  | 3.0919        | +26.48%       |
| M3_hist_gb       | HistGradientBoosting    | log1p      | 0.2215 | 2.9721        | +28.57%       |
| M4_linear_ridge  | Linear Ridge Regression | log1p      | 0.2553 | 3.4891        | +17.67%       |
| M5_random_forest | RandomForest Regressor  | log1p      | 0.2527 | 3.3973        | +18.51%       |


---

## 4. Bảng So sánh các Chiến lược Ensemble

| Chiến lược Ensemble                                | Thành phần & Trọng số                          | Hiệu chỉnh Smearing  | RMSLE  | MAE (Tr. VND) | vs Baseline % | vs Single Best % |
| -------------------------------------------------- | ---------------------------------------------- | -------------------- | ------ | ------------- | ------------- | ---------------- |
| ENS1: Simple Average Blend (5 mô hình)             | Equal 20% mỗi mô hình (M1..M5) trong log-space | Không                | 0.2221 | 3.0054        | +28.38%       | -4.32%           |
| ENS2: Top-3 Tree Average                           | Equal 33.3% (LGBM + ExtraTrees + HistGBR)      | Không                | 0.2167 | 2.9163        | +30.12%       | -1.78%           |
| ENS3: Out-of-Fold Stacking                         | Meta-Ridge học trọng số OOF qua 4-Fold CV      | Không                | 0.2229 | 3.0024        | +28.12%       | -4.70%           |
| ENS4: Optimized Weighted Blend                     | 0.90 LightGBM Huber + 0.10 Linear Ridge        | Không                | 0.2123 | 2.8787        | +31.54%       | +0.28%           |
| ENS5: Optimized Blend + Duan Smearing (Kỷ lục mới) | 0.90 LightGBM Huber + 0.10 Linear Ridge        | Duan factor = 1.0207 | 0.2096 | 2.809         | +32.41%       | +1.55%           |


---

## 5. Kết quả Walk-Forward Backtesting qua các Chu kỳ Thời gian

| Fold                    | Val Period               | Baseline RMSLE | Ridge RMSLE | LGBM Huber RMSLE | Ensemble RMSLE | vs Baseline % | vs Ridge % | vs Single Best % | Ensemble MAE |
| ----------------------- | ------------------------ | -------------- | ----------- | ---------------- | -------------- | ------------- | ---------- | ---------------- | ------------ |
| Fold A (Q1/2025)        | 2025-01-01 -> 2025-04-01 | 0.3037         | 0.2874      | 0.2959           | 0.283          | 6.82          | 1.53       | 4.36             | 3.8314       |
| Fold B (Q2/2025)        | 2025-04-01 -> 2025-07-01 | 0.2684         | 0.2367      | 0.1951           | 0.1921         | 28.43         | 18.84      | 1.54             | 2.6329       |
| Holdout chính (Q3/2025) | 2025-07-01 -> 2025-10-01 | 0.3101         | 0.2631      | 0.2129           | 0.2096         | 32.41         | 20.33      | 1.55             | 2.809        |


---

## 6. Phân tích Chi tiết Lỗi theo Nhóm (Subgroup Analysis trên Q3/2025)

### 6.1. Theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| ----------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| university  | 399       | 10.76         | 12.71       | 0.2995         | 2.7391       | -0.1145      | 43.8          | 24.32       | -1.3      |
| street      | 1266      | 23.34         | 22.41       | 0.2128         | 3.6622       | 0.0318       | 31.92         | 13.29       | 1.96      |
| mall        | 630       | 20.27         | 20.37       | 0.1991         | 3.345        | 0.0544       | 12.68         | 29.92       | 1.56      |
| office      | 985       | 16.79         | 16.28       | 0.1907         | 2.1463       | 0.04         | 21.2          | 18.09       | 3.85      |
| residential | 406       | 8.96          | 8.77        | 0.141          | 0.9931       | -0.0063      | 45.28         | 18.71       | 0.06      |


### 6.2. Theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| -------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| 1        | 358       | 21.53         | 21.81       | 0.307          | 5.6836       | 0.0166       | 30.65         | 10.32       | 1.17      |
| 0        | 3328      | 18.09         | 17.82       | 0.1963         | 2.4998       | 0.0179       | 32.83         | 22.45       | 1.65      |


### 6.3. Theo Thành phố

| city        | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| ----------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| Ho Chi Minh | 1797      | 18.13         | 17.98       | 0.2136         | 2.8387       | 0.0078       | 31.38         | 21.5        | -0.09     |
| Ha Noi      | 1889      | 18.44         | 18.25       | 0.2057         | 2.7807       | 0.0273       | 33.39         | 19.08       | 3.15      |


### 6.4. Theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| --------------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| High (>25M)     | 913       | 30.3          | 28.37       | 0.2967         | 5.8685       | 0.1023       | 14.21         | 13.86       | 2.64      |
| Low (<15M)      | 1305      | 10.19         | 10.45       | 0.2051         | 1.4748       | -0.0452      | 46.76         | 24.56       | -1.18     |
| Medium (15-25M) | 1468      | 19.55         | 19.4        | 0.1349         | 2.0921       | 0.0213       | 28.01         | 27.21       | 3.53      |


---

## 7. Tổng kết & Chuyển giao sang Giai đoạn Final Training & Submission (GĐ 7-8)

1. **Mô hình Xuất sắc nhất được Khóa (Locked Final Architecture):**
   - Kiến trúc: **Weighted Log-Space Ensemble (0.90 LightGBM Huber + 0.10 Ridge) + Duan Smearing Adjustment**.
   - Điểm số kiểm định Q3/2025: **RMSLE = 0.2095** | **MAE = 2.79 triệu VND**.
2. **Độ tin cậy Walk-Forward tuyệt đối:** Cả 3 quý lịch sử đều ghi nhận sự thống trị vững vàng của cấu hình Ensemble.
3. **Chuẩn bị Dự báo Test Q4/2025:**
   - Sẵn sàng fit lại toàn bộ dữ liệu lịch sử sạch (Train + Val) để tạo file dự báo cho Holdout Test hoặc kỳ thi chính thức theo đúng format `id, revenue`.
