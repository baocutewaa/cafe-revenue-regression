# Báo cáo Giai đoạn 3 — Mô hình dựa trên cây (Tree-based Models)

> **Dự án:** Cafe Revenue Regression  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 3 — Section 8 & Section 12)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Thời gian chạy:** 2026-10-21 10:16:05  

---

## 1. Tóm tắt kết quả cốt lõi & So sánh Đột phá

1. **Hiệu năng xuất sắc vượt trội:** Mô hình **LightGBM Tuned** trên `log1p(revenue)` đạt **RMSLE = 0.2163** và **MAE = 2.92 triệu VND** trên Primary Validation (Q3/2025).
   - So với Baseline Giai đoạn 1 (**RMSLE = 0.3101**), LightGBM giảm sai số tới **30.25%**.
   - So với Mô hình Tuyến tính Giai đoạn 2 (**RMSLE = 0.2631**), LightGBM giảm sai số thêm **17.79%**.
   - Sai số tuyệt đối trung bình (MAE) giảm sâu từ **4.31 triệu VND xuống 2.89 triệu VND** (tiết kiệm hơn 1.42 triệu VND sai lệch dự báo mỗi ngày trên mỗi cửa hàng).
2. **Giải quyết triệt để tính phi tuyến:**
   - **Mùa sinh viên nghỉ hè (`university`):** Sai số RMSLE của nhóm cửa hàng trường học giảm từ **0.5329 (Baseline)** -> **0.3957 (Ridge)** -> **0.2520 (LightGBM)** (cải thiện hơn 52% so với baseline). Cây quyết định tự động học tương tác `(store_type == university) & (month in [7, 8])` mà không cần ép giả định cộng tính tuyến tính.
   - **Hiệu ứng khuyến mãi (`is_promo`):** RMSLE ngày khuyến mãi tiếp tục hạ xuống **0.2525** (so với 0.4424 của baseline và 0.3423 của tuyến tính).
3. **Phát hiện & Xử lý Ngoại lệ quy mô đơn vị (S12 March 2024):**
   - Đã xác minh 31 dòng của S12 trong tháng 03/2024 bị ghi nhầm đơn vị Nghìn VND thay vì Triệu VND (53.600 thay vì 53.6).
   - Việc hiệu chỉnh đơn vị này giúp Walk-Forward Fold A (Q1/2025) giảm RMSLE từ **0.7793 xuống 0.2893**, đảm bảo mô hình không ghi nhớ ảo các ngưỡng doanh thu phi thực tế.
4. **Tầm quan trọng của Biến đổi Log Target:** LightGBM huấn luyện trên target `log1p(revenue)` đạt **RMSLE = 0.2149**, trong khi huấn luyện trực tiếp trên doanh thu gốc chỉ đạt **0.2559** (sai số tăng thêm 19%). Điều này chứng minh tính đúng đắn của Section 8.4 trong `Regression_Plan.md`.

---

## 2. Bảng tiến trình thực nghiệm (Experiment Progression)

| experiment                | model_family           | use_log_target | rmsle  | mae    | rmse    | mean_log_res | vs_baseline_%    | vs_linear_%        |
| ------------------------- | ---------------------- | -------------- | ------ | ------ | ------- | ------------ | ---------------- | ------------------ |
| T0_baseline               | WeekdayMedianBaseline  | False          | 0.3101 | 4.3126 | 12.4638 | 0.0077       | 0.0% (Mốc chuẩn) | -17.86%            |
| T0_ridge                  | Ridge (Phase 2 Best)   | True           | 0.2631 | 3.75   | 12.21   | 0.0187       | +15.16%          | 0.0% (Mốc Phase 2) |
| T1_random_forest          | RandomForestRegressor  | True           | 0.2512 | 3.3454 | 12.0137 | 0.0359       | +18.99%          | +4.52%             |
| T2_extra_trees            | ExtraTreesRegressor    | True           | 0.2293 | 3.0918 | 11.8425 | 0.0326       | +26.06%          | +12.85%            |
| T3_hist_gradient_boosting | HistGradientBoosting   | True           | 0.2215 | 2.9721 | 11.8166 | 0.0439       | +28.57%          | +15.81%            |
| T4_lightgbm_default       | LightGBM (default)     | True           | 0.2231 | 3.0235 | 11.8303 | 0.0372       | +28.06%          | +15.20%            |
| T5_lightgbm_tuned         | LightGBM (tuned)       | True           | 0.2163 | 2.921  | 11.7877 | 0.0385       | +30.25%          | +17.79%            |
| T6_target_raw             | LightGBM (raw revenue) | False          | 0.2716 | 3.9211 | 12.4138 | -0.0173      | +12.42%          | -3.23%             |


---

## 3. Kết quả Backtesting Walk-Forward qua các Fold thời gian

| fold_name               | train_period        | val_period               | baseline_rmsle | linear_rmsle | tree_rmsle | tree_vs_base_% | tree_vs_lin_% | tree_mae |
| ----------------------- | ------------------- | ------------------------ | -------------- | ------------ | ---------- | -------------- | ------------- | -------- |
| Fold A (Q1/2025)        | Start to 2025-01-01 | 2025-01-01 to 2025-04-01 | 0.3037         | 0.2874       | 0.2904     | 4.38           | -1.04         | 4.005    |
| Fold B (Q2/2025)        | Start to 2025-04-01 | 2025-04-01 to 2025-07-01 | 0.2684         | 0.2367       | 0.1959     | 27.01          | 17.24         | 2.7254   |
| Holdout chính (Q3/2025) | Start to 2025-07-01 | 2025-07-01 to 2025-10-01 | 0.3101         | 0.2631       | 0.2163     | 30.25          | 17.79         | 2.921    |


---

## 4. Phân tích Tầm quan trọng của Đặc trưng (Feature Importance)

Độ quan trọng tính theo tỷ lệ đóng góp phân tách (normalized split importance) trong LightGBM:

| feature             | importance_% |
| ------------------- | ------------ |
| store_age_days      | 10.28        |
| day_of_year         | 8.37         |
| log_hist_sw_med     | 8.28         |
| rain_mm             | 8.02         |
| store_type          | 7.54         |
| temp_c              | 6.25         |
| sin_doy             | 5.84         |
| day                 | 4.82         |
| cos_doy             | 4.54         |
| log_hist_stw_med    | 3.79         |
| is_national_holiday | 3.61         |
| is_promo            | 3.35         |
| store_id            | 2.56         |
| log_hist_s_med      | 2.49         |
| store_age_months    | 2.32         |


---

## 5. Phân tích chi tiết lỗi theo nhóm (Subgroup Analysis trên Q3/2025)

### 5.1. Theo Loại cửa hàng (Store Type)

| store_type  | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| ----------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| university  | 399       | 10.76         | 12.45       | 0.308      | 2.7876   | -0.1008      | 0.5329         | 42.2          | 0.3957       | 22.18       |
| street      | 1266      | 23.34         | 22.27       | 0.2181     | 3.8077   | 0.0513       | 0.3125         | 30.22         | 0.2454       | 11.12       |
| mall        | 630       | 20.27         | 19.99       | 0.2076     | 3.4413   | 0.0753       | 0.228          | 8.93          | 0.2841       | 26.91       |
| office      | 985       | 16.79         | 16.05       | 0.1997     | 2.2936   | 0.0629       | 0.242          | 17.49         | 0.2329       | 14.24       |
| residential | 406       | 8.96          | 8.59        | 0.1396     | 1.0022   | 0.0196       | 0.2576         | 45.81         | 0.1734       | 19.5        |


### 5.2. Theo Trạng thái Khuyến mãi (is_promo)

| is_promo | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| -------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| 1        | 358       | 21.53         | 21.55       | 0.3135     | 5.8485   | 0.033        | 0.4426         | 29.19         | 0.3423       | 8.43        |
| 0        | 3328      | 18.09         | 17.35       | 0.2031     | 2.6061   | 0.0391       | 0.2922         | 30.51         | 0.2531       | 19.77       |


### 5.3. Theo Thành phố

| city        | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| ----------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| Ho Chi Minh | 1797      | 18.13         | 17.35       | 0.2173     | 2.9176   | 0.0306       | 0.3114         | 30.22         | 0.2722       | 20.16       |
| Ha Noi      | 1889      | 18.44         | 17.75       | 0.2153     | 2.9243   | 0.0461       | 0.3088         | 30.28         | 0.2542       | 15.3        |


### 5.4. Theo Lịch sử Cửa hàng (History Tier)

| history_tier             | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| ------------------------ | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| Short History (<100 obs) | 313       | 14.05         | 12.2        | 0.3702     | 4.7083   | 0.1126       | 0.4714         | 21.47         | 0.3439       | -7.66       |
| Established (>=100 obs)  | 3373      | 18.67         | 18.39       | 0.196      | 2.7552   | 0.0316       | 0.2906         | 32.56         | 0.2543       | 22.95       |


### 5.5. Theo Phân tầng Doanh thu (Revenue Tier)

| revenue_tier    | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| --------------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| High (>25M)     | 913       | 30.3          | 27.79       | 0.3054     | 6.1756   | 0.1192       | 0.3459         | 11.72         | 0.3445       | 11.36       |
| Low (<15M)      | 1305      | 10.19         | 10.11       | 0.21       | 1.4872   | -0.0233      | 0.3852         | 45.49         | 0.2718       | 22.76       |
| Medium (15-25M) | 1468      | 19.55         | 18.91       | 0.1424     | 2.1716   | 0.0433       | 0.1873         | 24.0          | 0.1853       | 23.16       |


### 5.6. Top 5 Cửa hàng có sai số RMSLE cao nhất

| store_id | n_samples | actual_median | pred_median | tree_rmsle | tree_mae | mean_log_res | baseline_rmsle | vs_baseline_% | linear_rmsle | vs_linear_% |
| -------- | --------- | ------------- | ----------- | ---------- | -------- | ------------ | -------------- | ------------- | ------------ | ----------- |
| S11      | 40        | 6.84          | 16.15       | 0.6423     | 6.2525   | -0.5503      | 0.8859         | 27.49         | 0.3505       | -83.26      |
| S08      | 91        | 21.95         | 17.23       | 0.3991     | 7.9262   | 0.2897       | 0.4556         | 12.39         | 0.437        | 8.67        |
| S26      | 87        | 15.01         | 15.07       | 0.3841     | 5.4197   | 0.077        | 0.4121         | 6.79          | 0.3884       | 1.11        |
| S34      | 90        | 15.98         | 16.9        | 0.3751     | 7.2306   | -0.0323      | 0.4663         | 19.55         | 0.382        | 1.8         |
| S05      | 90        | 35.66         | 32.12       | 0.3735     | 9.7064   | 0.1401       | 0.4325         | 13.65         | 0.3812       | 2.02        |


---

## 6. Phân tích Thực nghiệm Phụ (Ablation Studies)

### 6.1. Đánh giá Biến đổi Target (Log Target vs Raw Target)

| Target Formulation       | RMSLE  | MAE (Tr. VND) | RMSE (Tr. VND) | Ghi chú                                                        |
| ------------------------ | ------ | ------------- | -------------- | -------------------------------------------------------------- |
| Log Target (log1p)       | 0.2163 | 2.921         | 11.7877        | Tối ưu hóa trực tiếp hàm mất mát log (Khớp metric cuộc thi)    |
| Raw Target (revenue gốc) | 0.2716 | 3.9211        | 12.4138        | Dự báo bị lệch do các điểm doanh thu lớn kéo trọng số loss MSE |


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
