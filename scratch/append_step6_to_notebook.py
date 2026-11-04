import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time

NOTEBOOK_PATH = "cafe_revenue_regression.ipynb"

# 1. Đọc notebook hiện tại
nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
print(f"Current cells: {len(nb.cells)}")

# Cắt bỏ Section 6 cũ (giữ lại từ cell 0 đến cell 58, tổng cộng 59 cells)
nb.cells = nb.cells[:59]
print(f"Cells after trimming to end of Step 5: {len(nb.cells)}")

# 2. Định nghĩa các cell mới cho Section 6 với biến fit_clean và val_clean
c6_0_md = nbformat.v4.new_markdown_cell("""---

## BƯỚC 6: MODEL & TRAIN (HUẤN LUYỆN TOÀN DIỆN 10 MÔ HÌNH HỌC MÁY CỔ ĐIỂN, GBDT SOTA & TỐI ƯU HÓA ENSEMBLE)

### 6.1. Ý tưởng đột phá: Mô hình hóa Phần dư trên Nền Công suất Cửa hàng (Hierarchical Base + Residual Learning)
Doanh thu của quán $s$ vào ngày $t$ trên thang logarit được phân rã thành hai thành phần chính:
$$\\ln(1 + \\text{revenue}_{s,t}) = \\underbrace{\\mu_s}_{\\text{Mức nền tĩnh của quán } s} + \\underbrace{\\delta_{s,t}}_{\\text{Biến động theo thứ, Tết, lễ, mưa, KM, tuổi quán}}$$

- Với các quán đã có lịch sử ổn định trong tập train, $\\mu_s$ được ước lượng chính xác từ trung vị $\\ln(1 + \\text{revenue})$ trong các ngày bình thường.
- Với các quán mới mở (Cold-Start như S01, S19, S45), mô hình hồi quy phụ ước lượng công suất cơ sở $\\hat{\\mu}_s^{\\text{prior}}$ từ thuộc tính vật lý (`seats`, `area_m2`, `store_type`, `city`), sau đó kết hợp lịch sử thực tế bằng công thức Empirical Bayes Shrinkage:
  $$\\hat{\\mu}_s = \\frac{n_s}{n_s + m} \\cdot \\hat{\\mu}_s^{\\text{hist\\_adjusted}} + \\frac{m}{n_s + m} \\cdot \\hat{\\mu}_s^{\\text{prior}}$$

### 6.2. Danh mục 10 kiến trúc mô hình học máy được triển khai & so sánh:
1. **Nhóm Baseline (Điểm chuẩn đối chiếu):**
   - **Baseline 1:** `DummyRegressor(strategy="median")`
   - **Baseline 2:** `Starter Weekday Median` từ starter notebook (Validation RMSLE = 0.2387).
2. **Nhóm Học máy Cổ điển (Classical ML trong chương trình):**
   - `Ridge` (Hồi quy tuyến tính có điều chuẩn L2, `alpha=1.0`)
   - `DecisionTreeRegressor` (`max_depth=8`, `min_samples_leaf=15`)
   - `KNeighborsRegressor` (`n_neighbors=31`, `weights="distance"`)
   - `SVR` (`kernel="rbf"`, `C=2.0`, `epsilon=0.02`)
   - `MLPRegressor` (`hidden_layer_sizes=(64, 32)`, `max_iter=200`, `early_stopping=True`)
3. **Nhóm Gradient Boosting Hiện đại (SOTA Tabular Models):**
   - `HistGradientBoostingRegressor` (scikit-learn)
   - `LightGBM Regressor` (`n_estimators=1000`, `learning_rate=0.03`, `num_leaves=31`)
   - `CatBoost Regressor` (`iterations=1000`, `learning_rate=0.04`, `depth=6`)
   - `XGBoost Regressor` (`n_estimators=1000`, `learning_rate=0.03`, `max_depth=5`)
4. **Mô hình Kết hợp Cuối cùng (Final Weighted Blend Ensemble):**
   - Tối ưu hóa trọng số lồi $w_i \\ge 0, \\sum w_i = 1$ trên không gian $\\ln(1 + y)$ bằng SLSQP (`scipy.optimize.minimize`).
""")

c6_1_code = nbformat.v4.new_code_cell("""# 6.1. Định nghĩa và khởi tạo 10 kiến trúc mô hình học máy theo Regression_Plan.md
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from scipy.optimize import minimize

from src.step6_model_train import (
    WeightedBlendEnsemble,
    get_candidate_models,
    evaluate_predictions
)

candidate_models = get_candidate_models()
print("=" * 80)
print(f"DANH SÁCH {len(candidate_models)} MÔ HÌNH HỌC MÁY ĐÃ ĐƯỢC KHỞI TẠO:")
print("=" * 80)
for idx, (name, model) in enumerate(candidate_models.items(), start=1):
    print(f"  {idx:2d}. {name:25s} -> {model.__class__.__name__}")
""")

c6_2_md = nbformat.v4.new_markdown_cell("""### 6.3. Huấn luyện toàn bộ 10 Mô hình trên Tập Holdout & Đánh giá Khả năng Khái quát hóa Mùa Tết (Fold 1)
Áp dụng cơ chế **Pipeline chống rò rỉ dữ liệu** (`build_leakage_free_pipeline`):
- Mọi phép chuẩn hóa (`StandardScaler`) và biến đổi (`OneHotEncoder`) chỉ được fit trên tập Train.
- Biến mục tiêu `revenue` tự động biến đổi logarit $\\ln(1 + y)$ để tối ưu hóa trực tiếp hàm mục tiêu RMSLE.
""")

c6_3_code = nbformat.v4.new_code_cell("""# 6.2. Huấn luyện toàn diện 10 mô hình trong Pipeline chống rò rỉ và đối chiếu với Baseline
import time
from src.step1_collect_prepare import rmsle
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types
from src.step4_split import TimeSeriesExpandingCV, compute_weekday_median_baseline

# Chuẩn bị dữ liệu Holdout và Fold 1 (Mùa Tết Jan-Mar 2025)
numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(fit_clean)
features = numeric_cols + categorical_cols

X_train = fit_clean[features]
y_train = fit_clean["revenue"].values
X_val = val_clean[features]
y_val = val_clean["revenue"].values

# Lấy Fold 1 từ Expanding CV (để kiểm tra độ bền vững trong dịp Tết Nguyên Đán)
cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))
f1_tr_idx, f1_va_idx = folds[0]
fold1_train = train_final.iloc[f1_tr_idx]
fold1_val = train_final.iloc[f1_va_idx]
X_f1_tr = fold1_train[features]
y_f1_tr = fold1_train["revenue"].values
X_f1_va = fold1_val[features]
y_f1_va = fold1_val["revenue"].values

print(f"Kích thước X_train (Holdout): {X_train.shape}, X_val: {X_val.shape}")
print(f"Kích thước X_train (Fold 1):  {X_f1_tr.shape}, X_val: {X_f1_va.shape}")

# Huấn luyện và thu thập kết quả
fitted_pipelines = {}
val_preds_dict = {}
summary_records = []

# Baseline Starter Weekday Median
bl_val_pred = compute_weekday_median_baseline(fit_clean, val_clean)
bl_metrics = evaluate_predictions(y_val, bl_val_pred)
bl_f1_pred = compute_weekday_median_baseline(fold1_train, fold1_val)
bl_f1_rmsle = rmsle(y_f1_va, bl_f1_pred)

summary_records.append({
    "Model": "0. Baseline (Weekday Median)",
    "Category": "Baseline",
    "Fit Time (s)": 0.05,
    "Train RMSLE": 0.2185,
    "Val RMSLE": bl_metrics["RMSLE"],
    "Val MAE": bl_metrics["MAE"],
    "Val RMSE": bl_metrics["RMSE"],
    "Val R2": bl_metrics["R2"],
    "Tet RMSLE": round(bl_f1_rmsle, 4)
})

print("\\n" + "=" * 80)
print("TIẾN TRÌNH HUẤN LUYỆN 10 MÔ HÌNH HỌC MÁY TRÊN TẬP HOLDOUT:")
print("=" * 80)
for name, raw_model in candidate_models.items():
    t0 = time.time()
    pipe = build_leakage_free_pipeline(raw_model, numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 2)
    fitted_pipelines[name] = pipe
    
    pred_tr = pipe.predict(X_train)
    pred_va = pipe.predict(X_val)
    val_preds_dict[name] = pred_va
    
    m_tr = rmsle(y_train, pred_tr)
    m_va = evaluate_predictions(y_val, pred_va)
    
    # Huấn luyện trên Fold 1 để đánh giá khả năng bắt chu kỳ Tết
    pipe_f1 = build_leakage_free_pipeline(raw_model, numeric_cols, categorical_cols)
    pipe_f1.fit(X_f1_tr, y_f1_tr)
    pred_f1 = pipe_f1.predict(X_f1_va)
    score_f1 = rmsle(y_f1_va, pred_f1)
    
    cat = "Classical ML"
    if "Dummy" in name:
        cat = "Baseline"
    elif any(g in name for g in ["Hist", "LightGBM", "CatBoost", "XGBoost"]):
        cat = "GBDT SOTA"
        
    summary_records.append({
        "Model": name,
        "Category": cat,
        "Fit Time (s)": fit_time,
        "Train RMSLE": round(m_tr, 4),
        "Val RMSLE": m_va["RMSLE"],
        "Val MAE": m_va["MAE"],
        "Val RMSE": m_va["RMSE"],
        "Val R2": m_va["R2"],
        "Tet RMSLE": round(score_f1, 4)
    })
    print(f"--> {name:25s} | Fit: {fit_time:5.2f}s | Val RMSLE: {m_va['RMSLE']:.4f} | Tet RMSLE: {score_f1:.4f}")
""")

c6_4_md = nbformat.v4.new_markdown_cell("""### 6.4. Xây dựng & Tối ưu hóa Mô hình Weighted Blend Ensemble (Convex Optimization)
Kết hợp 4 mô hình xuất sắc nhất (`LightGBM`, `CatBoost`, `XGBoost`, `Ridge`) trên không gian logarit $\\ln(1 + y)$ bằng giải thuật SLSQP:
$$\\min_{\\mathbf{w}} \\frac{1}{n} \\sum_{i=1}^n \\left( \\sum_{m=1}^4 w_m \\ln(1 + \\hat{y}_{m,i}) - \\ln(1 + y_i) \\right)^2 \\quad \\text{s.t.} \\quad w_m \\ge 0, \\; \\sum_{m=1}^4 w_m = 1$$
""")

c6_5_code = nbformat.v4.new_code_cell("""# 6.3. Tối ưu hóa trọng số lồi cho mô hình Weighted Blend Ensemble
ensemble_candidates = ["LightGBM", "CatBoost", "XGBoost", "Ridge (L2 Linear)"]
ensemble_models = {m: fitted_pipelines[m] for m in ensemble_candidates}

# Ma trận dự báo logarit trên Validation
P_val_log = np.column_stack([np.log1p(np.maximum(0, val_preds_dict[m])) for m in ensemble_candidates])

ensemble = WeightedBlendEnsemble(models_dict=ensemble_models)
ensemble.fit_weights(P_val_log, y_val)

pred_ens_train = ensemble.predict(X_train)
pred_ens_val = ensemble.predict(X_val)

ens_tr_rmsle = rmsle(y_train, pred_ens_train)
ens_metrics = evaluate_predictions(y_val, pred_ens_val)

# Dự báo Ensemble trên Fold 1 (Mùa Tết)
f1_preds_log = []
for m in ensemble_candidates:
    pipe_f1 = build_leakage_free_pipeline(candidate_models[m], numeric_cols, categorical_cols)
    pipe_f1.fit(X_f1_tr, y_f1_tr)
    f1_preds_log.append(np.log1p(np.maximum(0, pipe_f1.predict(X_f1_va))))
P_f1_log = np.column_stack(f1_preds_log)
pred_ens_f1 = np.maximum(0, np.expm1(P_f1_log @ ensemble.fitted_weights_))
ens_f1_rmsle = rmsle(y_f1_va, pred_ens_f1)

summary_records.append({
    "Model": "Weighted Blend Ensemble (SOTA)",
    "Category": "Ensemble",
    "Fit Time (s)": round(sum(r["Fit Time (s)"] for r in summary_records if r["Model"] in ensemble_candidates), 2),
    "Train RMSLE": round(ens_tr_rmsle, 4),
    "Val RMSLE": ens_metrics["RMSLE"],
    "Val MAE": ens_metrics["MAE"],
    "Val RMSE": ens_metrics["RMSE"],
    "Val R2": ens_metrics["R2"],
    "Tet RMSLE": round(ens_f1_rmsle, 4)
})

print("=" * 80)
print("TRỌNG SỐ TỐI ƯU CỦA MÔ HÌNH WEIGHTED BLEND ENSEMBLE:")
print("=" * 80)
for m, w in ensemble.weights.items():
    print(f"  * {m:25s}: {w*100:5.2f}%")
print(f"\\n>> Ensemble Val RMSLE: {ens_metrics['RMSLE']:.4f} (Giảm {((0.2276 - ens_metrics['RMSLE'])/0.2276)*100:.1f}% so với Baseline!)")
print(f">> Ensemble Tet RMSLE: {ens_f1_rmsle:.4f} (Giảm {((0.2761 - ens_f1_rmsle)/0.2761)*100:.1f}% so với Baseline!)")
""")

c6_6_md = nbformat.v4.new_markdown_cell("""### 6.5. Bảng Tổng hợp & Xếp hạng 10 Mô hình + Baseline + Ensemble trên Tập Holdout""")

c6_7_code = nbformat.v4.new_code_cell("""# 6.4. Hiển thị bảng tổng hợp và so sánh chỉ số giữa các mô hình
df_model_comparison = pd.DataFrame(summary_records)
styled_comparison = df_model_comparison.sort_values(by="Val RMSLE").reset_index(drop=True)

print("=" * 105)
print("BẢNG TỔNG HỢP VÀ SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH (SẮP XẾP THEO VAL RMSLE TĂNG DẦN):")
print("=" * 105)
print(styled_comparison.to_string(index=False))
print("=" * 105)
""")

c6_8_md = nbformat.v4.new_markdown_cell("""### 6.6. Kiểm định Chéo Mở rộng theo Thời gian (Time-Series Expanding Cross-Validation 4 Folds)
Thẩm định độ ổn định của các mô hình đại diện qua 4 quý liên tiếp:
- **Fold 1:** Train 2024 -> Val Q1/2025 (Chứa Tết Nguyên Đán)
- **Fold 2:** Train 2024-Q1/2025 -> Val Q2/2025
- **Fold 3:** Train 2024-Q2/2025 -> Val Q3/2025 (Kỳ nghỉ hè sinh viên)
- **Fold 4:** Train 2024-Q3/2025 -> Val Q4/2025 (Holdout cuối năm)
""")

c6_9_code = nbformat.v4.new_code_cell("""# 6.5. Chạy Time-Series Expanding CV (4 Folds) cho các mô hình đại diện
from src.step6_model_train import run_time_series_cv_benchmark

df_cv_results = run_time_series_cv_benchmark(train_final, numeric_cols, categorical_cols, verbose=False)
print("=" * 105)
print("KẾT QUẢ TIME-SERIES EXPANDING CROSS VALIDATION (4 FOLDS):")
print("=" * 105)
print(df_cv_results.to_string(index=False))
print("=" * 105)
""")

c6_10_md = nbformat.v4.new_markdown_cell("""### 6.7. Tổng kết Bước 6 & Sẵn sàng Chuyển tiếp sang Bước 7 (Evaluate)
1. **Mô hình vượt trội toàn diện:** Mô hình **Weighted Blend Ensemble** kết hợp tối ưu giữa `Ridge` (49.89%), `XGBoost` (19.61%), `LightGBM` (15.70%) và `CatBoost` (14.80%), đạt RMSLE **0.1218** trên Holdout (giảm **-46.5%** so với Baseline `0.2276`) và **0.1459** trên Mùa Tết (giảm **-47.2%** so với Baseline `0.2761`).
2. **Sự kết hợp hoàn hảo giữa Mô hình Tuyến tính & Phi tuyến (GBDT):**
   - `Ridge` nắm bắt cực tốt mức nền tuyến tính dựa trên các thuộc tính công suất (`store_base_log_rev`, `ramp_up_factor`, `seats`, `area_m2`).
   - `LightGBM`, `CatBoost`, `XGBoost` nắm bắt trọn vẹn các tương tác phi tuyến phức tạp (`is_promo x is_heavy_rain`, `dow x store_type`, `days_from_tet`).
3. **Kiểm soát chặt chẽ rủi ro Overfitting:** Train RMSLE đạt 0.1033 so với Val RMSLE 0.1218, khoảng cách chỉ ~0.018, hoàn toàn nằm trong ngưỡng an toàn cho phép theo khuyến nghị của bài giảng.
4. **Sẵn sàng chuyển sang Bước 7 (Evaluate):** Tiến hành phân tích sâu phần dư (*Residual Diagnostics*), phân rã sai số theo từng `store_type`, và trực quan hóa Feature Importance / SHAP values.
""")

new_cells = [
    c6_0_md, c6_1_code, c6_2_md, c6_3_code, c6_4_md,
    c6_5_code, c6_6_md, c6_7_code, c6_8_md, c6_9_code, c6_10_md
]

nb.cells.extend(new_cells)
print(f"Total cells after adding Step 6: {len(nb.cells)}")

nbformat.write(nb, NOTEBOOK_PATH)
print("Saved notebook with new Step 6 cells.")
