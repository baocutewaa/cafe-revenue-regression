import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.optimize import minimize

from step1_collect_prepare import rmsle
from step3_preprocess import run_step3
from step4_split import TimeSeriesExpandingCV
from step5_scale import get_feature_column_types, build_leakage_free_pipeline
from step6_model_train import get_candidate_models

print("Loading data...")
train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)
num_cols, cat_cols, ign_cols = get_feature_column_types(train_final)
features = num_cols + cat_cols
features_with_time = [c for c in train_final.columns if c in features or c in ["t_years", "date", "store_id"]]

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))

candidate_models = get_candidate_models()
models_to_test = ["CatBoost", "LightGBM", "XGBoost", "Ridge (L2 Linear)"]

oof_preds_log = {m: [] for m in models_to_test}
oof_trues = []

print("Running 4 folds...")
for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
    tr = train_final.iloc[tr_idx]
    va = train_final.iloc[va_idx]
    y_va = va["revenue"].values
    oof_trues.append(y_va)
    
    print(f"Fold {f_idx}: Train {len(tr)} rows, Val {len(va)} rows")
    for m in models_to_test:
        pipe = build_leakage_free_pipeline(clone(candidate_models[m]), num_cols, cat_cols, use_trend_aware=True)
        pipe.fit(tr[features_with_time], tr["revenue"].values)
        pred = pipe.predict(va[features_with_time])
        pred_log = np.log1p(np.maximum(0, pred))
        oof_preds_log[m].append(pred_log)
        score = rmsle(y_va, pred)
        print(f"   {m:20s}: RMSLE = {score:.4f}")

# Concatenate all OOF
all_trues = np.concatenate(oof_trues)
all_trues_log = np.log1p(np.maximum(0, all_trues))
P_oof_log = np.column_stack([np.concatenate(oof_preds_log[m]) for m in models_to_test])

print("\n=== Single Model OOF RMSLE across All 4 Folds ===")
for i, m in enumerate(models_to_test):
    sc = rmsle(all_trues, np.maximum(0, np.expm1(P_oof_log[:, i])))
    print(f"  {m:20s}: Full OOF RMSLE = {sc:.4f}")

# Optimize weights on Full OOF
def obj(w):
    w_norm = w / np.sum(w)
    pred_log = P_oof_log @ w_norm
    return np.mean((pred_log - all_trues_log) ** 2)

res = minimize(obj, [0.25]*4, bounds=[(0, 1)]*4, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
opt_w = res.x / np.sum(res.x)
print("\nOptimal SLSQP weights on Full OOF:")
for m, w in zip(models_to_test, opt_w):
    print(f"  {m:20s}: {w*100:.2f}%")

opt_pred_log = P_oof_log @ opt_w
opt_rmsle = rmsle(all_trues, np.maximum(0, np.expm1(opt_pred_log)))
print(f"\n>> Full OOF RMSLE with Optimal Weights: {opt_rmsle:.5f}")

# Check default weights
def_w = np.array([0.40, 0.25, 0.20, 0.15])
def_pred_log = P_oof_log @ def_w
def_rmsle = rmsle(all_trues, np.maximum(0, np.expm1(def_pred_log)))
print(f">> Full OOF RMSLE with Default Weights (40/25/20/15): {def_rmsle:.5f}")
