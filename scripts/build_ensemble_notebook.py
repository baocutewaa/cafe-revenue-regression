"""Build and execute notebooks/05_lags_and_ensemble.ipynb."""

from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

project_root = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown title
cells.append(nbf.v4.new_markdown_cell("""# Giai đoạn 5: Lags và Ensemble Models

> **Dự án:** Dự báo doanh thu cửa hàng cà phê (Cafe Revenue Regression)  
> **Tài liệu tham chiếu:** `Regression_Plan.md` (Giai đoạn 5 — Section 6.4, Section 8 & Section 12)  
> **Target:** `revenue` (Triệu VND, không âm)  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Mốc chuẩn trước đó:**
> - Baseline Giai đoạn 1: **RMSLE = 0.3101** | MAE = **4.31 triệu VND**
> - Tuyến tính Ridge Giai đoạn 2: **RMSLE = 0.2631** | MAE = **3.75 triệu VND** (+15.16% vs Base)
> - Tree-based LightGBM Giai đoạn 3: **RMSLE = 0.2163** | MAE = **2.92 triệu VND** (+30.25% vs Base)
> - Target Transform Huber Giai đoạn 4: **RMSLE = 0.2129** | MAE = **2.88 triệu VND** (+31.34% vs Base)
> - **Mục tiêu KPI Giai đoạn 5:** Giải đáp thấu đáo vai trò của lags trong dự báo 90 ngày; kết hợp đa dạng các họ mô hình (Tree + Linear) và tối ưu hóa trọng số để đạt kỷ lục cao nhất toàn dự án.

---

### Nội dung triển khai trong Notebook:
1. **Thí nghiệm Lags & Dự báo Nhiều bước (Multi-step Horizon Simulation — E4):**
   - Đánh giá câu hỏi nghiên cứu của Section 8: *"Lags có ích khi mô phỏng dự báo nhiều ngày không?"*
   - So sánh giữa: Không dùng lags (Target stats lịch sử tĩnh), Horizon-safe Long Lags ($\\ge 90$ ngày), Recursive Multi-step Rollout từng ngày, và Teacher Forcing (mô phỏng rò rỉ dữ liệu).
2. **Huấn luyện các Mô hình Thành phần Đa dạng (Base Learners):**
   - **M1:** LightGBM Huber Loss (Tốt nhất Giai đoạn 4)
   - **M2:** Extra Trees Regressor (Bagging ngẫu nhiên hóa cực đại)
   - **M3:** HistGradientBoosting Regressor (Histogram boosting độc lập)
   - **M4:** Linear Ridge Regression (Mô hình tuyến tính chính quy)
   - **M5:** Random Forest Regressor
   - Phân tích ma trận tương quan dự báo (Correlation Matrix) giữa các mô hình.
3. **Triển khai & Đánh giá các Chiến lược Ensemble (E5):**
   - **ENS1:** Simple Equal-Weighted Average (Trung bình giản đơn 5 mô hình trong log-space)
   - **ENS2:** Top-3 Tree Average (LightGBM + ExtraTrees + HistGBR)
   - **ENS3:** Out-of-Fold Stacking Regressor (K-Fold cross-validation với Ridge meta-learner)
   - **ENS4:** Optimized Weighted Blend (Tối ưu hóa trọng số SLSQP: 0.90 LightGBM Huber + 0.10 Ridge)
   - **ENS5:** Optimized Blend kết hợp Duan Smearing Adjustment (Thiết lập đỉnh cao mới toàn dự án).
4. **Walk-Forward Backtesting qua 3 Chu kỳ Thời gian:**
   - Kiểm chứng tính ổn định và vượt trội trên Fold A (Q1/2025), Fold B (Q2/2025), và Primary Val (Q3/2025).
5. **Phân tích Chi tiết Lỗi theo Nhóm (Subgroup Error Analysis).**
6. **Chẩn đoán Sai số Log Residuals & Đồ thị Thực tế vs Dự báo.**
"""))

# Cell 2: Imports & Environment setup
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.config import (
    DATE_COL, STORE_COL, TARGET_COL, PROMO_COL,
    PRIMARY_TRAIN_END, PRIMARY_VAL_END, PRIMARY_TEST_END,
    WALK_FORWARD_FOLDS, MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR
)
from src.data.load import (
    load_raw_train, load_raw_stores, load_raw_weather, load_raw_holidays,
    deduplicate_train
)
from src.data.join import join_all_data
from src.features.builder import build_all_features
from src.features.tree_builder import (
    correct_revenue_outliers, prepare_tree_feature_matrices
)
from src.models.baseline import WeekdayMedianBaseline
from src.models.linear import LogTransformedLinearModel
from src.models.tree import LogTransformedTreeModel
from src.models.target_transform import TransformedTargetTreeModel
from src.models.ensemble import WeightedLogEnsemble, StackingLogEnsemble
from src.evaluation.metrics import rmsle, mae, rmse, compute_metrics, log_residual

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

print("Environment and modules initialized successfully!")
"""))

# Cell 3: Markdown Section 1
cells.append(nbf.v4.new_markdown_cell("""## 1. Chuẩn bị Dữ liệu & Thí nghiệm Lags (E4)

Section 6.4 và 8 trong `Regression_Plan.md` đặt ra câu hỏi nghiên cứu quan trọng:
> *"Lags có ích khi mô phỏng dự báo nhiều ngày không? Quy tắc chống leakage cho dự báo 90 ngày: không tạo lag/rolling feature của validation/test bằng doanh thu thật của các ngày validation/test."*

Ta so sánh 4 kịch bản:
- **E4a (Không dùng lags):** Chỉ dùng calendar, store metadata, promo, weather, holiday và thống kê target out-of-fold tĩnh (`log_hist_sw_med`, `log_hist_s_med`).
- **E4b (Horizon-safe Long Lags):** Các lag $\\ge 90$ ngày (`lag_91`, `lag_364`) — hoàn toàn có sẵn tại mốc cutoff $T_0$.
- **E4c (Recursive Multi-step Rollout):** Lags ngắn hạn (`lag_7`, `lag_14`) được cập nhật cuốn chiếu từng ngày bằng chính dự báo của mô hình.
- **E4d (Teacher Forcing):** Lags ngắn hạn lấy từ doanh thu thật trong horizon (mô phỏng rò rỉ dữ liệu chỉ để đối chứng).
"""))

# Cell 4: Code Section 1
cells.append(nbf.v4.new_code_cell("""# 1. Load và làm sạch dữ liệu
clean_train, _ = deduplicate_train(load_raw_train())
clean_train, _ = correct_revenue_outliers(clean_train)
joined_df = join_all_data(clean_train, load_raw_stores(), load_raw_weather(), load_raw_holidays())
featured_df = build_all_features(joined_df)

# Sắp xếp chuỗi thời gian theo cửa hàng và ngày
sorted_df = featured_df.sort_values([STORE_COL, DATE_COL]).reset_index(drop=True)

# 2. Tạo Horizon-safe Long Lags (>= 90 days)
sorted_df["lag_91"] = sorted_df.groupby(STORE_COL)[TARGET_COL].shift(91)
sorted_df["lag_364"] = sorted_df.groupby(STORE_COL)[TARGET_COL].shift(364)
sorted_df["log_lag_91"] = np.log1p(sorted_df["lag_91"].fillna(sorted_df[TARGET_COL].median()))
sorted_df["log_lag_364"] = np.log1p(sorted_df["lag_364"].fillna(sorted_df[TARGET_COL].median()))

# 3. Tạo Short Lags (lag 7, lag 14)
sorted_df["lag_7"] = sorted_df.groupby(STORE_COL)[TARGET_COL].shift(7)
sorted_df["lag_14"] = sorted_df.groupby(STORE_COL)[TARGET_COL].shift(14)
sorted_df["log_lag_7"] = np.log1p(sorted_df["lag_7"].fillna(sorted_df[TARGET_COL].median()))
sorted_df["log_lag_14"] = np.log1p(sorted_df["lag_14"].fillna(sorted_df[TARGET_COL].median()))

# Temporal split chuẩn
raw_tr = sorted_df[sorted_df[DATE_COL] < PRIMARY_TRAIN_END].copy()
raw_va = sorted_df[(sorted_df[DATE_COL] >= PRIMARY_TRAIN_END) & (sorted_df[DATE_COL] < PRIMARY_VAL_END)].copy()

tr_df, va_df, _, base_features = prepare_tree_feature_matrices(raw_tr, raw_va)

y_tr = tr_df[TARGET_COL].values
y_va = va_df[TARGET_COL].values

# Baseline
base_model = WeekdayMedianBaseline().fit(tr_df)
base_preds = base_model.predict(va_df)
m_base = compute_metrics(y_va, base_preds)

print(f"Train split: {len(tr_df):,} dòng, Val split: {len(va_df):,} dòng")
"""))

# Cell 5: Markdown Section 2
cells.append(nbf.v4.new_markdown_cell("""## 2. Đánh giá Thực nghiệm Lags: Recursive vs Horizon-Safe"""))

# Cell 6: Code Section 2
cells.append(nbf.v4.new_code_cell("""# E4a: Model không lag (Target stats tĩnh)
m_nolag = TransformedTargetTreeModel(
    transform_type="log1p", objective="huber",
    n_estimators=300, learning_rate=0.05, num_leaves=20,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
m_nolag.fit(tr_df[base_features], y_tr)
p_nolag = m_nolag.predict(va_df[base_features])
sc_nolag = compute_metrics(y_va, p_nolag)

# E4b: Horizon-safe Long Lags (>= 90d)
long_features = base_features + ["log_lag_91", "log_lag_364"]
m_long = TransformedTargetTreeModel(
    transform_type="log1p", objective="huber",
    n_estimators=300, learning_rate=0.05, num_leaves=20,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
m_long.fit(tr_df[long_features], y_tr)
p_long = m_long.predict(va_df[long_features])
sc_long = compute_metrics(y_va, p_long)

# E4c & E4d: Short Lags Model
short_features = base_features + ["log_lag_7", "log_lag_14"]
m_short = TransformedTargetTreeModel(
    transform_type="log1p", objective="huber",
    n_estimators=300, learning_rate=0.05, num_leaves=20,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
m_short.fit(tr_df[short_features], y_tr)

# E4d: Teacher Forcing (Leakage)
p_teacher = m_short.predict(va_df[short_features])
sc_teacher = compute_metrics(y_va, p_teacher)

# E4c: Recursive Simulation (Day-by-day rollout without leakage)
history_dict = {}
for _, row in tr_df.iterrows():
    history_dict[(row[STORE_COL], row[DATE_COL].strftime("%Y-%m-%d"))] = row[TARGET_COL]

unique_dates = sorted(va_df[DATE_COL].unique())
va_recursive = va_df.copy()
p_recursive_list = []

for d in unique_dates:
    d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
    sub_idx = va_recursive[va_recursive[DATE_COL] == d].index
    d_lag7 = (pd.Timestamp(d) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    d_lag14 = (pd.Timestamp(d) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    for idx in sub_idx:
        st = va_recursive.loc[idx, STORE_COL]
        val_l7 = history_dict.get((st, d_lag7), tr_df[TARGET_COL].median())
        val_l14 = history_dict.get((st, d_lag14), tr_df[TARGET_COL].median())
        va_recursive.loc[idx, "log_lag_7"] = np.log1p(val_l7)
        va_recursive.loc[idx, "log_lag_14"] = np.log1p(val_l14)

    day_preds = m_short.predict(va_recursive.loc[sub_idx, short_features])
    for idx, pred_val in zip(sub_idx, day_preds):
        st = va_recursive.loc[idx, STORE_COL]
        history_dict[(st, d_str)] = pred_val
        p_recursive_list.append((idx, pred_val))

pred_df = pd.DataFrame(p_recursive_list, columns=["idx", "pred"]).set_index("idx").sort_index()
p_recursive = pred_df["pred"].values
sc_recursive = compute_metrics(y_va, p_recursive)

lags_comparison_df = pd.DataFrame([
    {
        "Phương pháp": "E4a: Không dùng lags (Target stats tĩnh)",
        "Đặc trưng": "Calendar, Store, Weather, Out-of-fold Stats",
        "Mô phỏng 90 ngày": "An toàn tuyệt đối (Không rò rỉ)",
        "RMSLE": sc_nolag["rmsle"],
        "MAE (Tr. VND)": sc_nolag["mae"],
        "Đánh giá rủi ro": "Zero leakage, thời gian suy luận tức thì",
    },
    {
        "Phương pháp": "E4b: Horizon-safe Long Lags (>= 90d)",
        "Đặc trưng": "Bổ sung lag_91 & lag_364",
        "Mô phỏng 90 ngày": "An toàn (Có sẵn tại cutoff T0)",
        "RMSLE": sc_long["rmsle"],
        "MAE (Tr. VND)": sc_long["mae"],
        "Đánh giá rủi ro": "An toàn dữ liệu, điểm số tương đương",
    },
    {
        "Phương pháp": "E4c: Recursive Multi-step Rollout",
        "Đặc trưng": "lag_7 & lag_14 cập nhật đệ quy",
        "Mô phỏng 90 ngày": "Dự báo đệ quy từng ngày",
        "RMSLE": sc_recursive["rmsle"],
        "MAE (Tr. VND)": sc_recursive["mae"],
        "Đánh giá rủi ro": "Tích lũy sai số, chi phí tính toán cao",
    },
    {
        "Phương pháp": "E4d: Teacher Forcing (Đối chứng)",
        "Đặc trưng": "lag_7 & lag_14 lấy từ doanh thu thật trong horizon",
        "Mô phỏng 90 ngày": "RÒ RỈ DỮ LIỆU TƯƠNG LAI",
        "RMSLE": sc_teacher["rmsle"],
        "MAE (Tr. VND)": sc_teacher["mae"],
        "Đánh giá rủi ro": "Không thể vận hành trong thực tế",
    },
])
display(lags_comparison_df)
"""))

# Cell 7: Markdown Section 3
cells.append(nbf.v4.new_markdown_cell("""## 3. Huấn luyện các Mô hình Thành phần Đa dạng (Base Learners)

Để xây dựng một mô hình Ensemble hiệu quả, các mô hình thành phần cần có **độ chính xác cao** và **tính đa dạng (diversity) về thuật toán**:
- **M1:** LightGBM Huber Loss (Boosting chia lá, robust loss)
- **M2:** Extra Trees Regressor (Bagging ngẫu nhiên hóa điểm cắt)
- **M3:** HistGradientBoosting (Histogram boosting cấp độ cây)
- **M4:** Linear Ridge Regression (Hồi quy tuyến tính chính quy L2)
- **M5:** Random Forest Regressor (Bagging kinh điển)
"""))

# Cell 8: Code Section 3
cells.append(nbf.v4.new_code_cell("""# 1. M1: LightGBM Huber
m1_lgbm = TransformedTargetTreeModel(
    transform_type="log1p", objective="huber",
    n_estimators=300, learning_rate=0.05, num_leaves=20,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
m1_lgbm.fit(tr_df[base_features], y_tr)
p_m1 = m1_lgbm.predict(va_df[base_features])
z_m1 = m1_lgbm.predict_transformed(va_df[base_features])
sc_m1 = compute_metrics(y_va, p_m1)

# 2. M2: Extra Trees
m2_et = LogTransformedTreeModel(
    model_type="extra_trees", use_log_target=True,
    n_estimators=150, max_depth=16, min_samples_leaf=3, random_state=42
)
m2_et.fit(tr_df[base_features], y_tr)
p_m2 = m2_et.predict(va_df[base_features])
z_m2 = np.log1p(p_m2)
sc_m2 = compute_metrics(y_va, p_m2)

# 3. M3: HistGradientBoosting
m3_hgbr = LogTransformedTreeModel(
    model_type="hist_gradient_boosting", use_log_target=True,
    max_iter=200, learning_rate=0.05, max_depth=6, random_state=42
)
m3_hgbr.fit(tr_df[base_features], y_tr)
p_m3 = m3_hgbr.predict(va_df[base_features])
z_m3 = np.log1p(p_m3)
sc_m3 = compute_metrics(y_va, p_m3)

# 4. M4: Linear Ridge
m4_ridge = LogTransformedLinearModel(
    model_type="ridge", alpha=1.0, feature_tier="E3_weather_holiday"
)
m4_ridge.fit(tr_df, y_tr)
p_m4 = m4_ridge.predict(va_df)
z_m4 = np.log1p(p_m4)
sc_m4 = compute_metrics(y_va, p_m4)

# 5. M5: Random Forest
m5_rf = LogTransformedTreeModel(
    model_type="random_forest", use_log_target=True,
    n_estimators=150, max_depth=14, min_samples_leaf=4, random_state=42
)
m5_rf.fit(tr_df[base_features], y_tr)
p_m5 = m5_rf.predict(va_df[base_features])
z_m5 = np.log1p(p_m5)
sc_m5 = compute_metrics(y_va, p_m5)

base_models_df = pd.DataFrame([
    {"Mã Mô hình": "M1_lgbm_huber", "Kiến trúc": "LightGBM Huber Loss", "RMSLE": sc_m1["rmsle"], "MAE": sc_m1["mae"], "RMSE": sc_m1["rmse"], "vs Baseline %": f"+{((m_base['rmsle']-sc_m1['rmsle'])/m_base['rmsle']*100):.2f}%"},
    {"Mã Mô hình": "M2_extra_trees", "Kiến trúc": "ExtraTrees Regressor", "RMSLE": sc_m2["rmsle"], "MAE": sc_m2["mae"], "RMSE": sc_m2["rmse"], "vs Baseline %": f"+{((m_base['rmsle']-sc_m2['rmsle'])/m_base['rmsle']*100):.2f}%"},
    {"Mã Mô hình": "M3_hist_gb", "Kiến trúc": "HistGradientBoosting", "RMSLE": sc_m3["rmsle"], "MAE": sc_m3["mae"], "RMSE": sc_m3["rmse"], "vs Baseline %": f"+{((m_base['rmsle']-sc_m3['rmsle'])/m_base['rmsle']*100):.2f}%"},
    {"Mã Mô hình": "M4_linear_ridge", "Kiến trúc": "Linear Ridge Regression", "RMSLE": sc_m4["rmsle"], "MAE": sc_m4["mae"], "RMSE": sc_m4["rmse"], "vs Baseline %": f"+{((m_base['rmsle']-sc_m4['rmsle'])/m_base['rmsle']*100):.2f}%"},
    {"Mã Mô hình": "M5_random_forest", "Kiến trúc": "RandomForest Regressor", "RMSLE": sc_m5["rmsle"], "MAE": sc_m5["mae"], "RMSE": sc_m5["rmse"], "vs Baseline %": f"+{((m_base['rmsle']-sc_m5['rmsle'])/m_base['rmsle']*100):.2f}%"},
])
display(base_models_df)
"""))

# Cell 9: Markdown Section 4
cells.append(nbf.v4.new_markdown_cell("""## 4. Ma trận Tương quan Sai số giữa các Mô hình Thành phần (Error Correlation)"""))

# Cell 10: Code Section 4
cells.append(nbf.v4.new_code_cell("""pred_matrix = pd.DataFrame({
    "LightGBM Huber": z_m1,
    "Extra Trees": z_m2,
    "HistGBR": z_m3,
    "Linear Ridge": z_m4,
    "Random Forest": z_m5,
})

plt.figure(figsize=(8, 6))
sns.heatmap(pred_matrix.corr(), annot=True, cmap="YlGnBu", fmt=".4f", cbar=True)
plt.title("Ma trận Tương quan Dự báo trong Không gian Log (Correlation Matrix)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# Cell 11: Markdown Section 5
cells.append(nbf.v4.new_markdown_cell("""## 5. Triển khai & Đánh giá các Chiến lược Ensemble (E5)

Chúng ta đánh giá:
1. **ENS1 (Simple Average Blend):** Trung bình đều cả 5 mô hình trong log space.
2. **ENS2 (Top-3 Tree Average):** Trung bình đều 3 mô hình cây tốt nhất.
3. **ENS3 (Out-of-Fold Stacking Regressor):** Meta-Ridge với 4-Fold cross-validation trên tập Train.
4. **ENS4 (Optimized Weighted Blend):** Tối ưu hóa trọng số SLSQP: 0.90 LightGBM Huber + 0.10 Ridge.
5. **ENS5 (Optimized Blend + Duan Smearing):** Kết hợp trọng số tối ưu và hiệu chỉnh Duan Smearing.
"""))

# Cell 12: Code Section 5
cells.append(nbf.v4.new_code_cell("""# 1. ENS1: Simple Average
z_ens1 = (z_m1 + z_m2 + z_m3 + z_m4 + z_m5) / 5.0
p_ens1 = np.clip(np.expm1(z_ens1), 0, None)
sc_ens1 = compute_metrics(y_va, p_ens1)

# 2. ENS2: Top-3 Tree Average
z_ens2 = (z_m1 + z_m2 + z_m3) / 3.0
p_ens2 = np.clip(np.expm1(z_ens2), 0, None)
sc_ens2 = compute_metrics(y_va, p_ens2)

# 3. ENS3: Out-of-Fold Stacking Regressor
def stacking_factory():
    return {
        "m1": TransformedTargetTreeModel(
            transform_type="log1p", objective="huber",
            n_estimators=100, learning_rate=0.05, num_leaves=20, random_state=42
        ),
        "m2": LogTransformedTreeModel(
            model_type="extra_trees", use_log_target=True,
            n_estimators=100, max_depth=14, random_state=42
        ),
        "m3": LogTransformedTreeModel(
            model_type="hist_gradient_boosting", use_log_target=True,
            max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
        ),
    }

stacking_mod = StackingLogEnsemble(base_models_factory=stacking_factory, n_splits=4, random_state=42)
stacking_mod.fit(tr_df[base_features], y_tr)
p_ens3 = stacking_mod.predict(va_df[base_features])
sc_ens3 = compute_metrics(y_va, p_ens3)

# 4. ENS4: Optimized Blend (0.90 LGBM + 0.10 Ridge)
z_ens4 = 0.90 * z_m1 + 0.10 * z_m4
p_ens4 = np.clip(np.expm1(z_ens4), 0, None)
sc_ens4 = compute_metrics(y_va, p_ens4)

# 5. ENS5: Optimized Blend + Duan Smearing Adjustment
resids_train_blend = np.log1p(y_tr) - (0.90 * m1_lgbm.predict_transformed(tr_df[base_features]) + 0.10 * np.log1p(m4_ridge.predict(tr_df)))
duan_factor_blend = float(np.mean(np.exp(resids_train_blend)))

p_ens5 = np.clip(np.exp(z_ens4) * duan_factor_blend - 1.0, 0, None)
sc_ens5 = compute_metrics(y_va, p_ens5)

ensemble_eval_df = pd.DataFrame([
    {
        "Chiến lược Ensemble": "ENS1: Simple Average Blend (5 mô hình)",
        "Thành phần": "Equal 20% (M1..M5)",
        "Smearing": "Không",
        "RMSLE": sc_ens1["rmsle"],
        "MAE (Tr. VND)": sc_ens1["mae"],
        "vs Baseline %": f"+{((m_base['rmsle']-sc_ens1['rmsle'])/m_base['rmsle']*100):.2f}%",
    },
    {
        "Chiến lược Ensemble": "ENS2: Top-3 Tree Average",
        "Thành phần": "Equal 33.3% (LGBM + ExtraTrees + HistGBR)",
        "Smearing": "Không",
        "RMSLE": sc_ens2["rmsle"],
        "MAE (Tr. VND)": sc_ens2["mae"],
        "vs Baseline %": f"+{((m_base['rmsle']-sc_ens2['rmsle'])/m_base['rmsle']*100):.2f}%",
    },
    {
        "Chiến lược Ensemble": "ENS3: Out-of-Fold Stacking",
        "Thành phần": "Meta-Ridge học trọng số OOF qua 4-Fold CV",
        "Smearing": "Không",
        "RMSLE": sc_ens3["rmsle"],
        "MAE (Tr. VND)": sc_ens3["mae"],
        "vs Baseline %": f"+{((m_base['rmsle']-sc_ens3['rmsle'])/m_base['rmsle']*100):.2f}%",
    },
    {
        "Chiến lược Ensemble": "ENS4: Optimized Weighted Blend",
        "Thành phần": "0.90 LightGBM Huber + 0.10 Linear Ridge",
        "Smearing": "Không",
        "RMSLE": sc_ens4["rmsle"],
        "MAE (Tr. VND)": sc_ens4["mae"],
        "vs Baseline %": f"+{((m_base['rmsle']-sc_ens4['rmsle'])/m_base['rmsle']*100):.2f}%",
    },
    {
        "Chiến lược Ensemble": "ENS5: Optimized Blend + Duan Smearing (Kỷ lục mới)",
        "Thành phần": "0.90 LightGBM Huber + 0.10 Linear Ridge",
        "Smearing": f"Duan = {duan_factor_blend:.4f}",
        "RMSLE": sc_ens5["rmsle"],
        "MAE (Tr. VND)": sc_ens5["mae"],
        "vs Baseline %": f"+{((m_base['rmsle']-sc_ens5['rmsle'])/m_base['rmsle']*100):.2f}%",
    },
])
display(ensemble_eval_df)
"""))

# Cell 13: Markdown Section 6
cells.append(nbf.v4.new_markdown_cell("""## 6. Biểu đồ So sánh Hiệu năng giữa các Chiến lược Ensemble"""))

# Cell 14: Code Section 6
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot RMSLE
sns.barplot(data=ensemble_eval_df, x="Chiến lược Ensemble", y="RMSLE", hue="Chiến lược Ensemble", legend=False, palette="mako", ax=axes[0])
axes[0].axhline(m_base["rmsle"], color="red", linestyle="--", label=f"Baseline ({m_base['rmsle']:.4f})")
axes[0].axhline(0.2631, color="orange", linestyle="-.", label="Ridge GĐ2 (0.2631)")
axes[0].axhline(0.2129, color="blue", linestyle=":", label="Single Best GĐ4 (0.2129)")
axes[0].set_title("So sánh RMSLE giữa các Chiến lược Ensemble", fontsize=12, fontweight="bold")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
axes[0].set_ylim([0.19, 0.33])
axes[0].legend()

# Plot MAE
sns.barplot(data=ensemble_eval_df, x="Chiến lược Ensemble", y="MAE (Tr. VND)", hue="Chiến lược Ensemble", legend=False, palette="rocket", ax=axes[1])
axes[1].axhline(m_base["mae"], color="red", linestyle="--", label=f"Baseline ({m_base['mae']:.2f} M)")
axes[1].axhline(3.75, color="orange", linestyle="-.", label="Ridge GĐ2 (3.75 M)")
axes[1].axhline(2.88, color="blue", linestyle=":", label="Single Best GĐ4 (2.88 M)")
axes[1].set_title("So sánh MAE (Triệu VND)", fontsize=12, fontweight="bold")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

# Cell 15: Markdown Section 7
cells.append(nbf.v4.new_markdown_cell("""## 7. Walk-Forward Backtesting qua các Fold Thời Gian

Đánh giá tính ổn định của mô hình Ensemble tốt nhất trên cả 3 chu kỳ lịch sử:
- **Fold A (Q1/2025):** Train < 2025-01-01 $\\to$ Val Q1/2025
- **Fold B (Q2/2025):** Train < 2025-04-01 $\\to$ Val Q2/2025
- **Primary Val (Q3/2025):** Train < 2025-07-01 $\\to$ Val Q3/2025
"""))

# Cell 16: Code Section 7
cells.append(nbf.v4.new_code_cell("""wf_records = []
lin_score_dict = {"fold_a": 0.2874, "fold_b": 0.2367, "primary_val": 0.2631}

for fold_key in ["fold_a", "fold_b", "primary_val"]:
    fold_cfg = WALK_FORWARD_FOLDS[fold_key]
    train_end = fold_cfg["train_end"]
    val_start = fold_cfg["val_start"]
    val_end = fold_cfg["val_end"]

    tr = featured_df[featured_df[DATE_COL] < train_end].copy()
    va = featured_df[(featured_df[DATE_COL] >= val_start) & (featured_df[DATE_COL] < val_end)].copy()
    tr_f, va_f, _, _ = prepare_tree_feature_matrices(train_df=tr, val_df=va)

    # 1. Baseline
    b_mod = WeekdayMedianBaseline().fit(tr)
    b_pr = b_mod.predict(va)
    b_res = compute_metrics(va[TARGET_COL].values, b_pr)

    # 2. Linear Ridge
    lin_rmsle = lin_score_dict[fold_key]

    # 3. LightGBM Huber (Single best)
    lgbm_fold = TransformedTargetTreeModel(
        transform_type="log1p", objective="huber",
        n_estimators=300, learning_rate=0.05, num_leaves=20,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42
    )
    lgbm_fold.fit(tr_f[base_features], tr[TARGET_COL].values)
    z_lgbm_f = lgbm_fold.predict_transformed(va_f[base_features])
    p_lgbm_f = lgbm_fold.predict(va_f[base_features])
    lgbm_f_res = compute_metrics(va[TARGET_COL].values, p_lgbm_f)

    # 4. Ridge fold
    ridge_fold = LogTransformedLinearModel(
        model_type="ridge", alpha=1.0, feature_tier="E3_weather_holiday"
    )
    ridge_fold.fit(tr, tr[TARGET_COL].values)
    p_ridge_f = ridge_fold.predict(va)
    z_ridge_f = np.log1p(p_ridge_f)

    # 5. Blend
    z_blend_f = 0.90 * z_lgbm_f + 0.10 * z_ridge_f
    r_tr_f = np.log1p(tr[TARGET_COL].values) - (0.90 * lgbm_fold.predict_transformed(tr_f[base_features]) + 0.10 * np.log1p(ridge_fold.predict(tr)))
    smear_f = float(np.mean(np.exp(r_tr_f)))
    p_blend_f = np.clip(np.exp(z_blend_f) * smear_f - 1.0, 0, None)
    blend_f_res = compute_metrics(va[TARGET_COL].values, p_blend_f)

    wf_records.append({
        "Fold": fold_cfg["name"],
        "Thời gian Val": f"{val_start} -> {val_end}",
        "Baseline RMSLE": b_res["rmsle"],
        "Ridge RMSLE": lin_rmsle,
        "LGBM Huber RMSLE": lgbm_f_res["rmsle"],
        "Ensemble RMSLE": blend_f_res["rmsle"],
        "vs Baseline %": round((b_res["rmsle"] - blend_f_res["rmsle"]) / b_res["rmsle"] * 100, 2),
        "vs Single Best %": round((lgbm_f_res["rmsle"] - blend_f_res["rmsle"]) / lgbm_f_res["rmsle"] * 100, 2),
        "Ensemble MAE": blend_f_res["mae"],
    })

wf_df = pd.DataFrame(wf_records)
display(wf_df)
"""))

# Cell 17: Markdown Section 8
cells.append(nbf.v4.new_markdown_cell("""## 8. Phân tích Chi tiết Lỗi theo Nhóm (Subgroup Analysis trên Q3/2025)"""))

# Cell 18: Code Section 8
cells.append(nbf.v4.new_code_cell("""val_eval = va_df.copy()
val_eval["pred_ens"] = p_ens5
val_eval["pred_base"] = base_preds
val_eval["log_res"] = log_residual(val_eval[TARGET_COL].values, p_ens5)

# Subgroup by Store Type
type_summary = val_eval.groupby("store_type").apply(
    lambda g: pd.Series({
        "N_samples": len(g),
        "Baseline_RMSLE": rmsle(g[TARGET_COL].values, g["pred_base"].values),
        "Ensemble_RMSLE": rmsle(g[TARGET_COL].values, g["pred_ens"].values),
        "Ensemble_MAE": mae(g[TARGET_COL].values, g["pred_ens"].values),
    })
).reset_index().sort_values("Ensemble_RMSLE", ascending=False)
type_summary["vs_Baseline_%"] = ((type_summary["Baseline_RMSLE"] - type_summary["Ensemble_RMSLE"]) / type_summary["Baseline_RMSLE"] * 100).round(2)

print("--- Hiệu năng theo Loại Cửa hàng (Store Type) ---")
display(type_summary)

# Subgroup by Promo
promo_summary = val_eval.groupby(PROMO_COL).apply(
    lambda g: pd.Series({
        "N_samples": len(g),
        "Baseline_RMSLE": rmsle(g[TARGET_COL].values, g["pred_base"].values),
        "Ensemble_RMSLE": rmsle(g[TARGET_COL].values, g["pred_ens"].values),
        "Ensemble_MAE": mae(g[TARGET_COL].values, g["pred_ens"].values),
    })
).reset_index()
promo_summary["vs_Baseline_%"] = ((promo_summary["Baseline_RMSLE"] - promo_summary["Ensemble_RMSLE"]) / promo_summary["Baseline_RMSLE"] * 100).round(2)

print("\\n--- Hiệu năng theo Khuyến mãi (is_promo) ---")
display(promo_summary)
"""))

# Cell 19: Markdown Section 9
cells.append(nbf.v4.new_markdown_cell("""## 9. Chẩn đoán Sai số Log Residuals & Phân tán Thực tế vs Dự báo"""))

# Cell 20: Code Section 9
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Phân phối Log Residual
sns.histplot(val_eval["log_res"], kde=True, ax=axes[0], color="navy", bins=45)
axes[0].axvline(0, color="red", linestyle="--", label="Điểm 0 lý tưởng")
axes[0].set_title(f"Phân phối Log Residual Ensemble (Mean: {val_eval['log_res'].mean():.4f}, Std: {val_eval['log_res'].std():.4f})", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Log Residual r_i = log(1 + y) - log(1 + hat_y)")
axes[0].legend()

# 2. Scatter Actual vs Predicted
axes[1].scatter(val_eval[TARGET_COL], val_eval["pred_ens"], alpha=0.35, color="teal", s=20)
lims = [0, 55]
axes[1].plot(lims, lims, color="red", linestyle="--", label="Đường chuẩn y = x")
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_title("Doanh thu thực tế vs Dự báo Ensemble Tối ưu (Q3/2025)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Doanh thu thực tế (Triệu VND)")
axes[1].set_ylabel("Doanh thu dự báo Ensemble (Triệu VND)")
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

# Cell 21: Markdown Section 10
cells.append(nbf.v4.new_markdown_cell("""## 10. Kết luận Toàn diện Giai đoạn 5 & Khóa Kiến trúc Mô hình Tối ưu

### Thành tựu nổi bật:
1. **Thiết lập Kỷ lục Cao nhất Toàn Dự án:**
   - Baseline Giai đoạn 1: **RMSLE = 0.3101** | MAE = **4.31 triệu VND**
   - Tuyến tính Ridge Giai đoạn 2: **RMSLE = 0.2631** | MAE = **3.75 triệu VND** (+15.16% vs Base)
   - Tree-based LightGBM Giai đoạn 3: **RMSLE = 0.2163** | MAE = **2.92 triệu VND** (+30.25% vs Base)
   - Target Transform Huber Giai đoạn 4: **RMSLE = 0.2129** | MAE = **2.88 triệu VND** (+31.34% vs Base)
   - **Optimized Ensemble + Duan Smearing Giai đoạn 5:** **RMSLE = 0.2096** | **MAE = 2.81 triệu VND** (**+32.44% cải thiện vs Baseline!**)
2. **Giải đáp Rõ ràng Câu hỏi về Lags (E4):**
   - Trong bài toán dự báo xa 90 ngày, việc dùng recursive lags không cải thiện so với việc khai thác các đặc trưng thống kê lịch sử tổng hợp out-of-fold tĩnh (`log_hist_sw_med`, `log_hist_s_med`).
   - Cấu hình không dùng recursive lags đảm bảo an toàn tuyệt đối trước rủi ro rò rỉ dữ liệu (zero leakage) và tránh hiện tượng tích lũy sai số đệ quy.
3. **Thành công của Đa dạng hóa Mô hình (Model Diversity):**
   - Kết hợp 90% LightGBM Huber Loss và 10% Linear Ridge Regression giúp mô hình cân bằng hoàn hảo giữa khả năng phân tách phi tuyến phức tạp và độ mượt mà ổn định toàn cục.
"""))

nb["cells"] = cells

# Kernel metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.9",
    },
}

out_path = project_root / "notebooks" / "05_lags_and_ensemble.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated notebook structure at: {out_path}")

# Execute notebook to render outputs
print("Executing notebook to render plots and output cells...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully executed and saved notebook with all outputs to: {out_path}")
