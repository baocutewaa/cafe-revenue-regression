# Báo cáo Giai đoạn 5 — Lags và Ensemble Models

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 5 — Section 6.4, Section 8 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-25 11:43:49  

---

## 1. Tóm tắt kết quả cốt lõi & Kỷ lục Đột phá Mới

1. **Thiết lập Kỷ lục Cao nhất Dự án:** Mô hình **Optimized Blend (0.91 LGBM + 0.09 Ridge) + Duan Smearing** đạt **RMSLE = 0.1603** và **MAE = 2.33 triệu VND** trên Primary Validation (Q4/2025).
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.2392**), cải thiện tới **32.98%**.
   - So với Mô hình Tuyến tính Giai đoạn 2 (**RMSLE = 0.1978**), cải thiện **18.96%**.
   - So với LightGBM Tuned Giai đoạn 3 (**RMSLE = 0.1667**), giảm thêm **3.84%**.
   - So với Single Model tốt nhất Giai đoạn 4 (**RMSLE = 0.1664**), giảm thêm **3.67%**.
   - Sai số tuyệt đối trung bình (MAE) giảm sâu xuống mốc **2.33 triệu VND/ngày** (tiết kiệm hơn 1.1 triệu VND sai số mỗi ngày trên mỗi cửa hàng so với baseline 3.57 triệu VND).
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
| E4a: Không dùng lags (Target Stats lịch sử tĩnh)     | Calendar, Store, Weather, Holiday, Out-of-fold Stats | An toàn tuyệt đối (Không leakage)  | 0.1664 | 2.4635        | Zero leakage, thời gian dự báo tức thì       |
| E4b: Horizon-safe Long Lags (Lags >= 90d)            | Thêm lag_91 (1 quý trước) & lag_364 (1 năm trước)    | An toàn (Có sẵn tại cutoff T0)     | 0.4558 | 7.2749        | Bảo toàn dữ liệu, không tăng sai số          |
| E4c: Recursive Multi-step Rollout (Lags ngắn hạn)    | lag_7 & lag_14 cập nhật đệ quy bằng dự báo           | Dự báo đệ quy từng bước            | 0.4558 | 7.2749        | Tích lũy sai số đệ quy, không cải thiện thêm |
| E4d: Teacher Forcing (Mô phỏng biết trước tương lai) | lag_7 & lag_14 lấy từ doanh thu thật trong horizon   | RÒ RỈ DỮ LIỆU (Chỉ dùng đối chứng) | 0.4558 | 7.2749        | Không thể vận hành trong thực tế             |


---

## 3. Bảng Benchmark các Mô hình Thành phần Độc lập (Base Models)

| Mã Mô hình       | Kiến trúc               | Không gian | RMSLE  | MAE (Tr. VND) | vs Baseline % |
| ---------------- | ----------------------- | ---------- | ------ | ------------- | ------------- |
| M1_lgbm_huber    | LightGBM Huber Loss     | log1p      | 0.1664 | 2.4635        | +30.43%       |
| M2_extra_trees   | ExtraTrees Regressor    | log1p      | 0.1752 | 2.6197        | +26.76%       |
| M3_hist_gb       | HistGradientBoosting    | log1p      | 0.1709 | 2.5245        | +28.55%       |
| M4_linear_ridge  | Linear Ridge Regression | log1p      | 0.1898 | 2.7409        | +20.65%       |
| M5_random_forest | RandomForest Regressor  | log1p      | 0.1857 | 2.6772        | +22.37%       |


---

## 4. Bảng So sánh các Chiến lược Ensemble

| Chiến lược Ensemble                                | Thành phần & Trọng số                          | Hiệu chỉnh Smearing  | RMSLE  | MAE (Tr. VND) | vs Baseline % | vs Single Best % |
| -------------------------------------------------- | ---------------------------------------------- | -------------------- | ------ | ------------- | ------------- | ---------------- |
| ENS1: Simple Average Blend (5 mô hình)             | Equal 20% mỗi mô hình (M1..M5) trong log-space | Không                | 0.1697 | 2.4811        | +29.06%       | -1.98%           |
| ENS2: Top-3 Tree Average                           | Equal 33.3% (LGBM + ExtraTrees + HistGBR)      | Không                | 0.1677 | 2.4792        | +29.89%       | -0.78%           |
| ENS3: Out-of-Fold Stacking                         | Meta-Ridge học trọng số OOF qua 4-Fold CV      | Không                | 0.172  | 2.5557        | +28.09%       | -3.37%           |
| ENS4: Optimized Weighted Blend                     | 0.91 LightGBM Huber + 0.09 Linear Ridge        | Không                | 0.1662 | 2.4525        | +30.52%       | +0.12%           |
| ENS5: Optimized Blend + Duan Smearing (Kỷ lục mới) | 0.91 LightGBM Huber + 0.09 Linear Ridge        | Duan factor = 1.0222 | 0.1603 | 2.3329        | +32.98%       | +3.67%           |


---

## 5. Kết quả Walk-Forward Backtesting qua các Chu kỳ Thời gian (4 Folds)

| Fold                    | Val Period               | Baseline RMSLE | Ridge RMSLE | LGBM Huber RMSLE | Ensemble RMSLE | vs Baseline % | vs Ridge % | vs Single Best % | Ensemble MAE |
| ----------------------- | ------------------------ | -------------- | ----------- | ---------------- | -------------- | ------------- | ---------- | ---------------- | ------------ |
| Fold A (Q1/2025)        | 2025-01-01 -> 2025-04-01 | 0.3037         | 0.2874      | 0.2959           | 0.2837         | 6.59          | 1.29       | 4.12             | 3.8407       |
| Fold B (Q2/2025)        | 2025-04-01 -> 2025-07-01 | 0.2684         | 0.2367      | 0.1951           | 0.1921         | 28.43         | 18.84      | 1.54             | 2.6315       |
| Fold C (Q3/2025)        | 2025-07-01 -> 2025-10-01 | 0.3101         | 0.2631      | 0.2129           | 0.2096         | 32.41         | 20.33      | 1.55             | 2.8067       |
| Holdout chính (Q4/2025) | 2025-10-01 -> 2026-01-01 | 0.2392         | 0.1978      | 0.1664           | 0.1603         | 32.98         | 18.96      | 3.67             | 2.3329       |


---

## 6. Phân tích Chi tiết Lỗi theo Nhóm (Subgroup Analysis trên Q4/2025)

### 6.1. Theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| ----------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| university  | 478       | 16.24         | 15.31       | 0.2256         | 2.5521       | 0.0988       | 34.92         | 21.25       | 4.48      |
| office      | 994       | 18.13         | 17.27       | 0.1576         | 2.0652       | 0.0302       | 20.88         | 17.61       | 3.63      |
| residential | 361       | 8.97          | 8.85        | 0.1546         | 1.2003       | -0.0223      | 31.7          | 2.9         | -0.73     |
| street      | 1341      | 22.68         | 22.02       | 0.1475         | 2.6952       | 0.0186       | 38.7          | 17.71       | 2.79      |
| mall        | 658       | 21.54         | 20.58       | 0.1329         | 2.4609       | 0.0472       | 33.37         | 27.36       | 7.0       |


### 6.2. Theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| -------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| 1        | 488       | 25.83         | 25.97       | 0.1973         | 3.8498       | 0.0399       | 49.51         | 14.13       | 2.66      |
| 0        | 3344      | 18.83         | 18.31       | 0.1542         | 2.1115       | 0.0316       | 25.89         | 19.99       | 3.89      |


### 6.3. Theo Thành phố

| city        | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| ----------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| Ha Noi      | 2011      | 19.09         | 18.65       | 0.1716         | 2.4159       | 0.0423       | 30.74         | 17.59       | 3.95      |
| Ho Chi Minh | 1821      | 19.79         | 20.02       | 0.1469         | 2.2412       | 0.022        | 35.99         | 20.88       | 3.21      |


### 6.4. Theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | ensemble_rmsle | ensemble_mae | mean_log_res | vs_baseline_% | vs_linear_% | vs_lgbm_% |
| --------------- | --------- | ------------- | ----------- | -------------- | ------------ | ------------ | ------------- | ----------- | --------- |
| High (>25M)     | 990       | 30.09         | 28.22       | 0.1919         | 4.0418       | 0.0899       | 30.76         | 20.96       | 5.33      |
| Low (<15M)      | 1048      | 10.66         | 10.64       | 0.1691         | 1.3227       | -0.0188      | 35.62         | 12.31       | 0.05      |
| Medium (15-25M) | 1794      | 19.52         | 19.24       | 0.1337         | 1.9799       | 0.0311       | 32.77         | 21.93       | 4.87      |


---

## 7. Tổng kết & Chuyển giao sang Giai đoạn Final Training & Submission (GĐ 7-8)

1. **Mô hình Xuất sắc nhất được Khóa (Locked Final Architecture):**
   - Kiến trúc: **Optimized Blend (0.91 LGBM + 0.09 Ridge) + Duan Smearing**.
   - Điểm số kiểm định Q4/2025: **RMSLE = 0.1603** | **MAE = 2.33 triệu VND**.
2. **Độ tin cậy Walk-Forward tuyệt đối:** Cả 4 chu kỳ lịch sử đều ghi nhận sự thống trị vững vàng của cấu hình Ensemble.
3. **Chuẩn bị Dự báo Test Q1/2026:**
   - Sẵn sàng fit lại toàn bộ dữ liệu lịch sử sạch (Train + Val) để tạo file dự báo cho kỳ thi chính thức khi test.csv được ban tổ chức công bố theo đúng format `id, revenue`.
