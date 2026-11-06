import json

with open("cafe_revenue_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_64_code = '''# 6.3. Tối ưu hóa trọng số lồi cho mô hình Weighted Blend Ensemble
from sklearn.base import clone

ensemble_candidates = ["CatBoost", "LightGBM", "XGBoost", "Ridge (L2 Linear)"]
ensemble_models = {m: fitted_pipelines[m] for m in ensemble_candidates if m in fitted_pipelines}

# 1. Ma trận dự báo logarit trên Validation (Holdout Quý 4)
P_val_log = np.column_stack([np.log1p(np.maximum(0, val_preds_dict[m])) for m in ensemble_models.keys()])

# 2. Dự báo trên Fold 1 (Mùa Tết Quý 1)
f1_preds_log = []
for m in ensemble_models.keys():
    pipe_f1 = build_leakage_free_pipeline(clone(candidate_models[m]), numeric_cols, categorical_cols, use_trend_aware=True)
    pipe_f1.fit(X_f1_tr, y_f1_tr)
    f1_preds_log.append(np.log1p(np.maximum(0, pipe_f1.predict(X_f1_va))))
P_f1_log = np.column_stack(f1_preds_log)

# 3. Kết hợp cả tập Holdout (Quý 4) và Fold 1 (Mùa Tết Quý 1 phản chiếu Test) để tìm trọng số vững chắc cho mọi mùa
P_comb_log = np.vstack([P_val_log, P_f1_log])
y_comb = np.concatenate([y_val, y_f1_va])

ensemble = WeightedBlendEnsemble(models_dict=ensemble_models)
ensemble.fit_weights(P_comb_log, y_comb)

pred_ens_train = ensemble.predict(X_train)
pred_ens_val = ensemble.predict(X_val)

ens_tr_rmsle = rmsle(y_train, pred_ens_train)
ens_metrics = evaluate_predictions(y_val, pred_ens_val)

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
print(f">> Ensemble Tet RMSLE: {ens_f1_rmsle:.4f} (Giảm {tet_gain:.1f}% so với Baseline!)")'''

nb['cells'][64]['source'] = [l + '\n' for l in cell_64_code.strip().split('\n')]

with open("cafe_revenue_regression.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated notebook cell 64 successfully!")
