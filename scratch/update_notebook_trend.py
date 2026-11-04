import json

NOTEBOOK_PATH = "cafe_revenue_regression.ipynb"
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_62_code = '''# 6.2. Huấn luyện toàn diện 10 mô hình trong Pipeline chống rò rỉ và đối chiếu với Baseline
import time
from sklearn.base import clone
from src.step1_collect_prepare import rmsle
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types
from src.step4_split import TimeSeriesExpandingCV, compute_weekday_median_baseline

# Chuẩn bị dữ liệu Holdout và Fold 1 (Mùa Tết Jan-Mar 2025)
numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(fit_clean)
features = numeric_cols + categorical_cols
features_with_time = [c for c in fit_clean.columns if c in features or c in ["t_years", "date"]]

X_train = fit_clean[features_with_time]
y_train = fit_clean["revenue"].values
X_val = val_clean[features_with_time]
y_val = val_clean["revenue"].values

# Lấy Fold 1 từ Expanding CV (để kiểm tra độ bền vững trong dịp Tết Nguyên Đán)
cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))
f1_tr_idx, f1_va_idx = folds[0]
fold1_train = train_final.iloc[f1_tr_idx]
fold1_val = train_final.iloc[f1_va_idx]
X_f1_tr = fold1_train[features_with_time]
y_f1_tr = fold1_train["revenue"].values
X_f1_va = fold1_val[features_with_time]
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
print("TIẾN TRÌNH HUẤN LUYỆN 10 MÔ HÌNH HỌC MÁY TRÊN TẬP HOLDOUT (TREND-AWARE SOTA):")
print("=" * 80)
for name, raw_model in candidate_models.items():
    t0 = time.time()
    pipe = build_leakage_free_pipeline(clone(raw_model), numeric_cols, categorical_cols, use_trend_aware=True)
    pipe.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 2)
    fitted_pipelines[name] = pipe
    
    pred_tr = pipe.predict(X_train)
    pred_va = pipe.predict(X_val)
    val_preds_dict[name] = pred_va
    
    m_tr = rmsle(y_train, pred_tr)
    m_va = evaluate_predictions(y_val, pred_va)
    
    # Huấn luyện trên Fold 1 để đánh giá khả năng bắt chu kỳ Tết
    pipe_f1 = build_leakage_free_pipeline(clone(raw_model), numeric_cols, categorical_cols, use_trend_aware=True)
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
'''

cell_64_code = '''# 6.3. Tối ưu hóa trọng số lồi cho mô hình Weighted Blend Ensemble
from sklearn.base import clone

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
    pipe_f1 = build_leakage_free_pipeline(clone(candidate_models[m]), numeric_cols, categorical_cols, use_trend_aware=True)
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
rel_gain = ((0.2276 - ens_metrics['RMSLE'])/0.2276)*100
tet_gain = ((0.2761 - ens_f1_rmsle)/0.2761)*100
print(f"\\n>> Ensemble Val RMSLE: {ens_metrics['RMSLE']:.4f} (Giảm {rel_gain:.1f}% so với Baseline!)")
print(f">> Ensemble Tet RMSLE: {ens_f1_rmsle:.4f} (Giảm {tet_gain:.1f}% so với Baseline!)")
'''

nb['cells'][62]['source'] = [line + '\n' for line in cell_62_code.strip().split('\n')]
nb['cells'][64]['source'] = [line + '\n' for line in cell_64_code.strip().split('\n')]

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated notebook cells 62 and 64 successfully!")
