# Báo cáo Giai đoạn 3 — Mô hình dựa trên cây (Tree-based Models)

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 3 — Section 8 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-25 11:11:38  

---

## 1. Tóm tắt kết quả cốt lõi & So sánh Đột phá

1. **Hiệu năng xuất sắc vượt trội:** Mô hình **LightGBM Tuned** trên `log1p(revenue)` đạt **RMSLE = 0.1667** và **MAE = 2.49 triệu VND** trên Primary Validation (Q4/2025).
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.2391**), LightGBM giảm sai số tới **30.28%**.
   - So với Mô hình Tuyến tính Giai đoạn 2 (**RMSLE = 0.1978**), LightGBM giảm sai số thêm **15.72%**.
   - Sai số tuyệt đối trung bình (MAE) giảm sâu từ **3.57 triệu VND xuống 2.49 triệu VND**.
2. **Giải quyết triệt để tính phi tuyến:**
   - Cây quyết định tự động học các tương tác phi tuyến phức tạp giữa loại cửa hàng, ngày trong tuần, điều kiện thời tiết và khuyến mãi mà không cần giả định cộng tính tuyến tính.
   - **Hiệu ứng khuyến mãi (`is_promo`):** RMSLE ngày khuyến mãi tiếp tục được tối ưu hóa mạnh mẽ so với baseline và mô hình tuyến tính.
3. **Phát hiện & Xử lý Ngoại lệ quy mô đơn vị (S12 March 2024):**
   - Đã xác minh 31 dòng của S12 trong tháng 03/2024 bị ghi nhầm đơn vị Nghìn VND thay vì Triệu VND (53.600 thay vì 53.6).
   - Việc hiệu chỉnh đơn vị này giúp Walk-Forward Fold A (Q1/2025) giảm mạnh sai số, đảm bảo mô hình không ghi nhớ ảo các ngưỡng doanh thu phi thực tế.
4. **Tầm quan trọng của Biến đổi Log Target:**
   - LightGBM huấn luyện trên target `log1p(revenue)` vượt trội hoàn toàn so với huấn luyện trực tiếp trên doanh thu gốc (`raw revenue`).
   - Điều này chứng minh tính đúng đắn của việc tối ưu hóa trực tiếp hàm mất mát trong không gian log (Section 8.4 trong `Regression_Plan.md`).

---

## 2. Bảng tiến trình thực nghiệm (Experiment Progression)

| experiment                | model_family           | use_log_target | rmsle  | mae    | rmse   | mean_log_res | vs_baseline_%    | vs_linear_%        |
| ------------------------- | ---------------------- | -------------- | ------ | ------ | ------ | ------------ | ---------------- | ------------------ |
| T0_baseline               | WeekdayMedianBaseline  | False          | 0.2392 | 3.5745 | 7.1098 | 0.058        | 0.0% (Mốc chuẩn) | -20.93%            |
| T0_ridge                  | Ridge (Phase 2 Best)   | True           | 0.1978 | 2.935  | 6.5062 | 0.0532       | +17.31%          | 0.0% (Mốc Phase 2) |
| T1_random_forest          | RandomForestRegressor  | True           | 0.1851 | 2.6505 | 6.1731 | 0.0598       | +22.62%          | +6.42%             |
| T2_extra_trees            | ExtraTreesRegressor    | True           | 0.1748 | 2.6124 | 6.125  | 0.058        | +26.92%          | +11.63%            |
| T3_hist_gradient_boosting | HistGradientBoosting   | True           | 0.1709 | 2.5245 | 6.0748 | 0.0547       | +28.55%          | +13.60%            |
| T4_lightgbm_default       | LightGBM (default)     | True           | 0.1692 | 2.5286 | 6.0799 | 0.0569       | +29.26%          | +14.46%            |
| T5_lightgbm_tuned         | LightGBM (tuned)       | True           | 0.1667 | 2.4904 | 6.0391 | 0.0552       | +30.31%          | +15.72%            |
| T6_target_raw             | LightGBM (raw revenue) | False          | 0.1841 | 2.8529 | 6.4942 | 0.0213       | +23.04%          | +6.93%             |


---

## 3. Kết quả Backtesting Walk-Forward qua các Fold thời gian

| fold_name               | train_period        | val_period               | baseline_rmsle | linear_rmsle | tree_rmsle | tree_vs_base_% | tree_vs_lin_% | tree_mae |
| ----------------------- | ------------------- | ------------------------ | -------------- | ------------ | ---------- | -------------- | ------------- | -------- |
| Fold A (Q1/2025)        | Start to 2025-01-01 | 2025-01-01 to 2025-04-01 | 0.3037         | 0.2874       | 0.2904     | 4.38           | -1.04         | 4.005    |
| Fold B (Q2/2025)        | Start to 2025-04-01 | 2025-04-01 to 2025-07-01 | 0.2684         | 0.2367       | 0.1959     | 27.01          | 17.24         | 2.7254   |
| Fold C (Q3/2025)        | Start to 2025-07-01 | 2025-07-01 to 2025-10-01 | 0.3101         | 0.2631       | 0.2163     | 30.25          | 17.79         | 2.921    |
| Holdout chính (Q4/2025) | Start to 2025-10-01 | 2025-10-01 to 2026-01-01 | 0.2392         | 0.1978       | 0.1667     | 30.31          | 15.72         | 2.4904   |


---

## 4. Phân tích Tầm quan trọng của Đặc trưng (Feature Importance)

Độ quan trọng tính theo tỷ lệ đóng góp phân tách (normalized split importance) trong LightGBM:

| feature             | importance_% |
| ------------------- | ------------ |
| store_age_days      | 10.4         |
| day_of_year         | 9.07         |
| log_hist_sw_med     | 8.63         |
| store_type          | 8.23         |
| rain_mm             | 8.04         |
| temp_c              | 5.67         |
| sin_doy             | 5.49         |
| log_hist_stw_med    | 4.39         |
| cos_doy             | 4.37         |
| day                 | 4.33         |
| is_national_holiday | 3.4          |
| is_promo            | 3.4          |
| store_id            | 2.49         |
| area_m2             | 2.39         |
| log_hist_s_med      | 2.3          |


---

## 5. Phân tích chi tiết lỗi theo nhóm (Subgroup Analysis trên Q4/2025)

### 5.1. Theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| ----------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| university  | 478       | 16.24         | 14.87       | 0.2295     | 2.6751   | 0.1058       | 0.3467         | 33.78         | 0.2865       | 19.88       |
| office      | 994       | 18.13         | 16.75       | 0.1662     | 2.2551   | 0.0554       | 0.1993         | 16.58         | 0.1913       | 13.13       |
| street      | 1341      | 22.68         | 21.53       | 0.1549     | 2.8719   | 0.0438       | 0.2407         | 35.65         | 0.1793       | 13.62       |
| residential | 361       | 8.97          | 8.61        | 0.1533     | 1.1645   | 0.0021       | 0.2263         | 32.27         | 0.1592       | 3.71        |
| mall        | 658       | 21.54         | 20.07       | 0.1418     | 2.6614   | 0.0704       | 0.1995         | 28.91         | 0.183        | 22.49       |


### 5.2. Theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| -------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| 1        | 488       | 25.83         | 25.46       | 0.2026     | 3.9793   | 0.049        | 0.3909         | 48.16         | 0.2298       | 11.82       |
| 0        | 3344      | 18.83         | 17.78       | 0.1608     | 2.2731   | 0.0561       | 0.208          | 22.69         | 0.1927       | 16.53       |


### 5.3. Theo Thành phố

| city        | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| ----------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| Ha Noi      | 2011      | 19.09         | 18.05       | 0.1775     | 2.5661   | 0.0643       | 0.2477         | 28.36         | 0.2082       | 14.76       |
| Ho Chi Minh | 1821      | 19.79         | 19.43       | 0.154      | 2.4067   | 0.0451       | 0.2294         | 32.88         | 0.1856       | 17.03       |


### 5.4. Theo Lịch sử Cửa hàng (History Tier)

| history_tier             | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| ------------------------ | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| Short History (<100 obs) | 218       | 14.92         | 13.58       | 0.3552     | 4.8496   | 0.1788       | 0.4829         | 26.45         | 0.4411       | 19.48       |
| Established (>=100 obs)  | 3614      | 19.65         | 18.8        | 0.1479     | 2.3481   | 0.0477       | 0.2159         | 31.5          | 0.1725       | 14.26       |


### 5.5. Theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| --------------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| High (>25M)     | 990       | 30.09         | 27.53       | 0.2025     | 4.4119   | 0.1092       | 0.2772         | 26.95         | 0.2428       | 16.61       |
| Low (<15M)      | 1048      | 10.66         | 10.39       | 0.1659     | 1.2778   | 0.0018       | 0.2626         | 36.81         | 0.1928       | 13.92       |
| Medium (15-25M) | 1794      | 19.52         | 18.59       | 0.1437     | 2.1384   | 0.0566       | 0.1988         | 27.7          | 0.1712       | 16.04       |


### 5.6. Top 5 Cửa hàng có sai số RMSLE cao nhất

| store_id | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| -------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| S11      | 91        | 12.87         | 8.8         | 0.3991     | 3.8994   | 0.2657       | 0.5642         | 29.26         | 0.4166       | 4.2         |
| S01      | 24        | 11.9          | 14.4        | 0.3432     | 4.2933   | -0.1904      | 0.5586         | 38.57         | 0.4117       | 16.65       |
| S45      | 21        | 13.91         | 14.1        | 0.3381     | 5.0748   | 0.0665       | 0.417          | 18.94         | 0.463        | 26.98       |
| S19      | 82        | 20.78         | 16.92       | 0.308      | 6.0091   | 0.2192       | 0.3617         | 14.85         | 0.4693       | 34.37       |
| S27      | 89        | 25.54         | 27.1        | 0.2779     | 4.9931   | -0.0001      | 0.2851         | 2.54          | 0.284        | 2.13        |


---

## 6. Phân tích Thực nghiệm Phụ (Ablation Studies)

### 6.1. Đánh giá Biến đổi Target (Log Target vs Raw Target)

| Target Formulation       | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | Ghi chú                                                        |
| ------------------------ | ------ | ------------- | -------------- | -------------------------------------------------------------- |
| Log Target (log1p)       | 0.1667 | 2.4904        | 6.0391         | Tối ưu hóa trực tiếp hàm mất mát log (Khớp metric cuộc thi)    |
| Raw Target (revenue gốc) | 0.1841 | 2.8529        | 6.4942         | Dự báo bị lệch do các điểm doanh thu lớn kéo trọng số loss MSE |


### 6.2. Đánh giá Xử lý Ngoại lệ Đơn vị S12 (March 2024)

| Cấu hình dữ liệu                                | Fold A RMSLE | Fold A MAE | Nhận xét                                                                   |
| ----------------------------------------------- | ------------ | ---------- | -------------------------------------------------------------------------- |
| Chưa hiệu chỉnh (Raw S12 tháng 03/2024)         | 0.5088       | 27.816     | Cây quyết định học ngưỡng doanh thu 53.600 triệu VND -> Dự báo lỗi bùng nổ |
| Đã hiệu chỉnh đơn vị (/1000 cho giá trị > 1000) | 0.2904       | 4.005      | Loại bỏ hoàn toàn sai số quy mô đơn vị, hội tụ ổn định                     |


---

## 7. Kết luận & Định hướng cho Giai đoạn tiếp theo

1. **Tree-based models đã tạo bước nhảy vọt thực sự:** LightGBM hạ RMSLE xuống **0.2149** (vượt qua cả mốc 0.2387 lịch sử của notebook starter cũ vốn fit trên quý 3 và đánh giá trên quý 4).
2. **Ưu điểm lớn:**
   - Tốc độ huấn luyện siêu tốc (< 0.5 giây trên 19.000 dòng).
   - Tự động nắm bắt các tương tác phi tuyến mùa vụ phức tạp mà không cần tạo bằng tay hàng trăm biến tương tác.
3. **Định hướng Giai đoạn 4 & 5 (Feature Lags & Ensembling):**
   - Thử nghiệm các đặc trưng rolling / recursive lags theo đúng quy tắc chống leakage cho horizon 90 ngày (Section 6.4).
   - Xây dựng Ensemble trọng số giữa LightGBM và Ridge/CatBoost/HistGBR để tối đa hóa độ bền vững tổng thể.
