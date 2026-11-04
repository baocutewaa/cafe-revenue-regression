import nbformat

NOTEBOOK_PATH = "cafe_revenue_regression.ipynb"
nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
print(f"Current cell count: {len(nb.cells)}")

# Nếu đã có Section 8 thì cắt tỉa lại
has_step8 = any("SECTION 8: [BƯỚC 8A]" in c.source for c in nb.cells)
if has_step8:
    print("Notebook already has Step 8. Trimming back to cell 76...")
    nb.cells = nb.cells[:76]

# 1. Markdown 8.0: Header & Overview
c8_0_md = nbformat.v4.new_markdown_cell("""---

## SECTION 8: [BƯỚC 8A] FULL RETRAINING, KAGGLE SUBMISSION & EXPORT ARTIFACTS

Đúng theo chỉ dẫn tại Bước 8 của bài giảng Chương 4:
> *"Output of this step: someone who does not code can use the model."*  
> *"Retrain on 100% of the training data before final submission to maximize accuracy."*

### 8.1. Tại sao phải Huấn luyện lại trên 100% Dữ liệu (Full Retraining)?
- Trong Bước 4–7, chúng ta giữ riêng tập Validation (Quý 4/2025) để thẩm định khách quan chống rò rỉ dữ liệu.
- Khi chuyển sang tạo file nộp bài cho Quý 1/2026 trên Kaggle, việc huấn luyện lại trên 100% tập dữ liệu sạch (26,500 dòng từ 2024-01-01 đến 2025-12-31) mang lại lợi thế quyết định:
  - Cả 3 cửa hàng mở muộn (**S19** mở tháng 4/2025, **S01** mở tháng 8/2025, **S45** mở tháng 11/2025) đều đã có dữ liệu thực tế lịch sử trong tập huấn luyện.
  - Kết hợp với hệ số đường cong chạy đà `ramp_up_factor`, mô hình sẽ có căn cứ chuẩn xác để dự báo doanh thu Quý 1/2026.
""")

# 2. Code 8.1: Full Retraining
c8_1_code = nbformat.v4.new_code_cell("""# 8.1. Huấn luyện lại toàn bộ mô hình trong Weighted Blend Ensemble trên 100% dữ liệu sạch (26,500 dòng)
from src.step8_deploy import retrain_ensemble_on_full_data, DEFAULT_ENSEMBLE_WEIGHTS

full_ensemble, full_fitted_pipes = retrain_ensemble_on_full_data(
    train_clean_df=train_final,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    ensemble_weights=DEFAULT_ENSEMBLE_WEIGHTS,
    verbose=True
)

print("\\n" + "=" * 80)
print("TRỌNG SỐ ĐÓNG GÓP TRONG ENSEMBLE HUẤN LUYỆN TRÊN 100% DỮ LIỆU:")
print("=" * 80)
for m, w in full_ensemble.weights.items():
    print(f"  * {m:25s}: {w*100:5.2f}%")
print("=" * 80)
""")

# 3. Code 8.2: Submission Generation & Sanity Checks
c8_2_code = nbformat.v4.new_code_cell("""# 8.2. Tạo tệp dự báo submission.csv và thực hiện bộ kiểm tra tự động (Sanity Checks)
from src.step8_deploy import generate_kaggle_submission

sub_df, sanity_results = generate_kaggle_submission(
    fitted_ensemble=full_ensemble,
    test_clean_df=test_final,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    output_path="submission.csv",
    verbose=True
)

# Hiển thị 10 dòng đầu và 5 dòng cuối của tệp submission.csv
print("10 DÒNG ĐẦU TỆP SUBMISSION.CSV:")
display(sub_df.head(10))

print("\\n5 DÒNG CUỐI TỆP SUBMISSION.CSV:")
display(sub_df.tail(5))
""")

# 4. Code 8.3: Package Artifacts Bundle
c8_3_code = nbformat.v4.new_code_cell("""# 8.3. Đóng gói toàn bộ tài nguyên vào artifacts/cafe_revenue_bundle.joblib & JSON Metrics
import os
import json
from src.step8_deploy import package_artifact_bundle
from src.step3_preprocess import DeSeasonalizedStoreStats

# Học StoreStatsEncoder trên 100% dữ liệu lịch sử sạch
store_stats_encoder = DeSeasonalizedStoreStats(shrinkage_m=10.0, alpha_ridge=1.0)
store_stats_encoder.fit(train_final, train_final["revenue"].values)

metrics_summary = {
    "model_name": "Weighted Blend Ensemble (SOTA)",
    "holdout_val_rmsle": 0.1218,
    "seasonal_tet_rmsle": 0.1459,
    "val_mae_m_vnd": 2.03,
    "val_rmse_m_vnd": 2.86,
    "val_r2": 0.8950,
    "starter_baseline_rmsle": 0.2276,
    "relative_improvement_pct": 46.5,
    "ensemble_weights": DEFAULT_ENSEMBLE_WEIGHTS
}

bundle = package_artifact_bundle(
    fitted_ensemble=full_ensemble,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    stores_clean=stores_clean,
    weather_clean=weather_clean,
    holidays_clean=holidays_clean,
    store_stats_encoder=store_stats_encoder,
    metrics_summary=metrics_summary,
    bundle_path="artifacts/cafe_revenue_bundle.joblib",
    json_path="artifacts/model_metrics_summary.json",
    verbose=True
)

# Kiểm tra nội dung tệp JSON chỉ số
with open("artifacts/model_metrics_summary.json", "r", encoding="utf-8") as f:
    saved_metrics = json.load(f)
print("\\nNỘI DUNG TỆP JSON CHỈ SỐ:")
print(json.dumps(saved_metrics, ensure_ascii=False, indent=2))
""")

# 5. Markdown 8.4: In-notebook Interactive Form Demo Header
c8_4_md = nbformat.v4.new_markdown_cell("""---

## SECTION 9: [BƯỚC 8B] INTERACTIVE IN-NOTEBOOK DEMO & WHAT-IF SIMULATOR

Theo đúng thiết kế biểu mẫu tương tác của slide Chương 4 (trang 42 và 58):
> Giảng viên hoặc người dùng không cần biết lập trình có thể trực tiếp điều chỉnh các tham số đầu vào (`store_id`, `date`, `is_promo`, `temp_c`, `rain_mm`) ngay trên notebook thông qua form Google Colab / Jupyter interactive widgets để nhận kết quả dự báo tức thì cùng kịch bản What-If.
""")

# 6. Code 8.5: Interactive Widget Form
c8_5_code = nbformat.v4.new_code_cell("""# 8.4. Biểu mẫu Dự báo Tương tác Trực tiếp & Mô phỏng Kịch bản What-If ngay trong Notebook
# @title ☕ Dự Báo Doanh Thu Quán Cafe & Kịch Bản What-If { run: "auto" }
from src.step8_deploy import predict_single_instance

# Các tham số tương tác qua form widget
demo_store_id = "S01"  # @param ["S01", "S02", "S03", "S04", "S05", "S12", "S19", "S24", "S29", "S45"] {allow-input: true}
demo_date = "2026-02-17"  # @param {type:"string"}
demo_is_promo = 1  # @param [0, 1] {type:"integer"}
demo_temp_c = 28.5  # @param {type:"slider", min:15.0, max:42.0, step:0.5}
demo_rain_mm = 0.0  # @param {type:"slider", min:0.0, max:120.0, step:1.0}

result = predict_single_instance(
    bundle=bundle,
    store_id=demo_store_id,
    date_str=demo_date,
    is_promo=demo_is_promo,
    temp_c=demo_temp_c,
    rain_mm=demo_rain_mm
)

print("=" * 85)
print(f"KẾT QUẢ DỰ BÁO DOANH THU CHO QUÁN {result['store_id']} ({result['store_type'].upper()}, {result['city']})")
print(f"Ngày: {result['date']} ({result['day_of_week']}) | Điều kiện: {result['temp_c']}°C, Mưa: {result['rain_mm']} mm, Khuyến mãi: {'CÓ' if result['is_promo']==1 else 'KHÔNG'}")
print("=" * 85)
print(f"  * DOANH THU DỰ BÁO TỪ ENSEMBLE:   {result['predicted_revenue_m_vnd']:,.2f} TRIỆU VNĐ")
print(f"  * HIỆU ỨNG KHUYẾN MÃI (WHAT-IF):   {result['promo_uplift_m_vnd']:+,.2f} TRIỆU VNĐ ({result['promo_uplift_pct']:+}%)")
print(f"  * DOANH THU KHI ĐỔI TRẠNG THÁI KM: {result['what_if_opposite_promo_pred']:,.2f} TRIỆU VNĐ")
print("-" * 85)
print("CHI TIẾT DỰ BÁO TỪ TỪNG MÔ HÌNH THÀNH PHẦN:")
for model_name, comp_rev in result["component_predictions"].items():
    print(f"  - {model_name:25s}: {comp_rev:5.2f} triệu VNĐ (Trọng số: {bundle['weights'].get(model_name, 0)*100:5.2f}%)")
print("-" * 85)
print(f"💡 KHUYẾN NGHỊ VẬN HÀNH TỰ ĐỘNG:")
print(f"   {result['operational_recommendation']}")
print("=" * 85)
""")

# 7. Markdown 8.6: Deployment Guide
c8_6_md = nbformat.v4.new_markdown_cell("""---

## SECTION 10: HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG FASTAPI BACKEND & STREAMLIT DASHBOARD

Hệ thống đã được đóng gói hoàn chỉnh thành hai ứng dụng Web độc lập sẵn sàng phục vụ thực tế:

### 1. Khởi chạy FastAPI Backend Server:
Mở terminal và chạy lệnh:
```bash
uvicorn app.api_server:app --reload --port 8000
```
- **Tài liệu Swagger UI tương tác:** Truy cập `http://localhost:8000/docs` để kiểm thử trực tiếp các REST API:
  - `GET /health`: Kiểm tra sức khỏe hệ thống và các chỉ số mô hình.
  - `GET /stores`: Danh sách 45 cửa hàng kèm thuộc tính tĩnh.
  - `POST /predict/single`: Dự báo doanh thu 1 ngày + tính What-If.
  - `POST /predict/what-if`: So sánh đối đầu kịch bản Trước vs Sau khi thay đổi khuyến mãi / thời tiết.
  - `POST /predict/batch`: Dự báo theo lô hàng loạt cho tập dữ liệu CSV tùy ý.

### 2. Khởi chạy Giao diện Streamlit Dashboard:
Mở một cửa sổ terminal mới và chạy:
```bash
streamlit run app/streamlit_app.py
```
- **Giao diện người dùng:** Tự động mở tại `http://localhost:8501` gồm 4 tab chuyên biệt:
  - **Tab 1 – Dự báo Ngày & What-If:** Chọn ngày, bật/tắt khuyến mãi, kéo thanh trượt nhiệt độ/mưa, xem 3 thẻ KPI lớn và khuyến nghị vận hành tự động.
  - **Tab 2 – Lập kế hoạch Chuỗi ngày & Mùa Tết:** Chọn khoảng ngày (ví dụ cao điểm Tết 2026), vẽ biểu đồ đường tương tác và tổng doanh thu chuỗi ngày.
  - **Tab 3 – Dự báo Hàng loạt & Xuất Kaggle:** Tải file CSV, nhấn nút dự báo hàng loạt và tải ngay tệp `submission.csv`.
  - **Tab 4 – Bảng điều khiển Chất lượng Mô hình & Khám phá Dữ liệu:** Xem bảng Ablation Study, chẩn đoán phần dư và Top 15 đặc trưng dẫn dắt.
""")

new_cells = [c8_0_md, c8_1_code, c8_2_code, c8_3_code, c8_4_md, c8_5_code, c8_6_md]
nb.cells.extend(new_cells)
print(f"Total cells after adding Step 8: {len(nb.cells)}")

nbformat.write(nb, NOTEBOOK_PATH)
print("Successfully appended Step 8 cells to notebook.")
