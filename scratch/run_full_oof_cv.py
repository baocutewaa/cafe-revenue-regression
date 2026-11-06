import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone

from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import TimeSeriesExpandingCV
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types
from src.step6_model_train import get_candidate_models, evaluate_predictions

train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

# Clean fixes for outliers
train_fixed = train_final.copy()
train_fixed.loc[(train_fixed["store_id"] == "S17") & (train_fixed["date"] == "2024-02-10"), "revenue"] = 8.97
train_fixed.loc[(train_fixed["store_id"] == "S29") & (train_fixed["date"] == "2024-09-02"), "revenue"] = 35.90

tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29"), 2026: pd.Timestamp("2026-02-17")}
train_fixed["days_from_tet_exact"] = train_fixed.apply(
    lambda r: (r["date"] - tet_map[r["date"].year]).days if r["date"].year in tet_map else 999, axis=1
)

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_fixed))

models_to_run = ["LightGBM", "CatBoost", "XGBoost", "Ridge (L2 Linear)"]
candidate_models = get_candidate_models()

# Store out-of-fold predictions
oof_preds = {m: [] for m in models_to_run}
oof_trues = []
fold_results = {m: [] for m in models_to_run}

print("Running 4-Fold CV with Enhanced Tet Dynamics...")

for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
    tr = train_fixed.iloc[tr_idx].copy()
    va = train_fixed.iloc[va_idx].copy()
    
    # Compute normal baseline excluding Tet and promos
    tr_normal = tr[(tr["days_from_tet_exact"] < -14) | (tr["days_from_tet_exact"] > 14)]
    tr_normal = tr_normal[(tr_normal["is_promo"] == 0) & (tr_normal["is_heavy_rain"] == 0)]
    tr_base = tr_normal.groupby(["store_id", "dow"])["revenue"].median()
    
    tr_norm_col = tr.set_index(["store_id", "dow"]).index.map(tr_base).fillna(tr["revenue"].median())
    tr["temp_ratio"] = tr["revenue"] / tr_norm_col
    
    tet_mask = tr["days_from_tet_exact"].between(-18, 18)
    tet_curve = tr[tet_mask].groupby(["store_type", "days_from_tet_exact"])["temp_ratio"].median().to_dict()
    tr = tr.drop(columns=["temp_ratio"])
    
    def get_tet_log_factor(df):
        return np.array([np.log(np.maximum(0.2, tet_curve.get((st, d), 1.0))) if -18 <= d <= 18 else 0.0 
                         for st, d in zip(df["store_type"], df["days_from_tet_exact"])])
        
    tr["feat_tet_log_factor"] = get_tet_log_factor(tr)
    va["feat_tet_log_factor"] = get_tet_log_factor(va)
    
    numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(tr)
    features = numeric_cols + categorical_cols
    features_with_time = [c for c in tr.columns if c in features or c in ["t_years", "date"]]
    
    y_va = va["revenue"].values
    oof_trues.append(y_va)
    
    for m in models_to_run:
        pipe = build_leakage_free_pipeline(clone(candidate_models[m]), numeric_cols, categorical_cols, use_trend_aware=True)
        pipe.fit(tr[features_with_time], tr["revenue"].values)
        pred = pipe.predict(va[features_with_time])
        score = rmsle(y_va, pred)
        fold_results[m].append(score)
        oof_preds[m].append(pred)
        
    print(f"Fold {f_idx}: " + " | ".join([f"{m}: {fold_results[m][-1]:.4f}" for m in models_to_run]))

# Print Summary per model
print("\n=== Model Benchmark across 4 Folds ===")
for m in models_to_run:
    scores = fold_results[m]
    print(f"{m:20s}: F1={scores[0]:.4f}, F2={scores[1]:.4f}, F3={scores[2]:.4f}, F4={scores[3]:.4f} | Mean={np.mean(scores):.4f}")

# Optimize ensemble weights across all out-of-fold predictions
all_trues = np.concatenate(oof_trues)
all_trues_log = np.log1p(np.maximum(0, all_trues))
all_preds_log = np.column_stack([np.log1p(np.maximum(0, np.concatenate(oof_preds[m]))) for m in models_to_run])

def objective(w):
    w_norm = w / np.sum(w)
    pred_blend_log = all_preds_log @ w_norm
    return np.mean((pred_blend_log - all_trues_log) ** 2)

init_w = np.ones(len(models_to_run)) / len(models_to_run)
bounds = [(0.0, 1.0) for _ in range(len(models_to_run))]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
opt_w = res.x / np.sum(res.x)

print("\n=== Optimal Ensemble Weights across All 4 Folds ===")
for m, w in zip(models_to_run, opt_w):
    print(f"  * {m:20s}: {w*100:5.2f}%")

# Compute ensemble performance per fold and overall
ens_fold_scores = []
for f_idx in range(4):
    y_f = oof_trues[f_idx]
    p_f_log = np.column_stack([np.log1p(np.maximum(0, oof_preds[m][f_idx])) for m in models_to_run])
    blend_log = p_f_log @ opt_w
    pred_blend = np.maximum(0, np.expm1(blend_log))
    score = rmsle(y_f, pred_blend)
    ens_fold_scores.append(score)
    print(f"Fold {f_idx+1} Ensemble RMSLE: {score:.4f}")

print(f"\nFinal Out-of-Fold Ensemble Mean RMSLE: {np.mean(ens_fold_scores):.4f}")
print(f"Overall Concatenated OOF RMSLE:       {rmsle(all_trues, np.maximum(0, np.expm1(all_preds_log @ opt_w))):.4f}")
