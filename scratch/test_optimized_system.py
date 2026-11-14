import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from scipy.optimize import minimize

from step1_collect_prepare import rmsle
from step3_preprocess import run_step3
from step4_split import TimeSeriesExpandingCV
from step5_scale import get_feature_column_types, build_leakage_free_pipeline

print("--- 1. Loading data & creating enhanced features ---")
train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

def add_advanced_features(df):
    d = df.copy()
    # 1. Store type indicators
    is_uni = (d["store_type"] == "university").astype(int)
    is_mall = (d["store_type"] == "mall").astype(int)
    is_street = (d["store_type"] == "street").astype(int)
    is_office = (d["store_type"] == "office").astype(int)
    is_res = (d["store_type"] == "residential").astype(int)
    
    # 2. Promotion interactions (University students have 38% lift, Mall has 29% lift)
    d["promo_x_uni"] = d["is_promo"] * is_uni
    d["promo_x_mall"] = d["is_promo"] * is_mall
    d["promo_x_street"] = d["is_promo"] * is_street
    d["promo_x_dow"] = d["is_promo"] * d["dow"]
    
    # 3. Tet Shopping Peak interaction (Day -7 Ong Tao and Days -6 to -3 for Mall and Street)
    # days_from_tet_clipped is between -21 and 21
    d_tet = d["days_from_tet_clipped"]
    d["is_pre_tet_mall_rush"] = (is_mall & (d_tet.between(-7, -3))).astype(int)
    d["is_pre_tet_street_rush"] = (is_street & (d_tet.between(-7, -3))).astype(int)
    
    # 4. Tet Core slump (-2 to 2) for Street and Residential
    d["is_tet_core_slump"] = ((is_street | is_res) & (d_tet.between(-2, 2))).astype(int)
    
    # 5. Post-Tet recovery (+3 to +9) for Office (slow return) vs Mall (immediate return)
    d["is_post_tet_office_ramp"] = (is_office & (d_tet.between(3, 9))).astype(int)
    d["is_post_tet_mall_bounce"] = (is_mall & (d_tet.between(2, 6))).astype(int)
    
    # 6. Smooth distance from Tet (quadratic/gaussian decay centered at Tet)
    d["tet_dist_smooth"] = np.exp(-0.5 * (d_tet / 5.0) ** 2)
    
    # 7. Rain and outdoor store interaction
    d["rain_heavy_outdoor"] = d["is_heavy_rain"] * (is_street | is_res | is_uni).astype(int)
    
    return d

train_enh = add_advanced_features(train_final)
test_enh = add_advanced_features(test_final)

num_cols, cat_cols, ign_cols = get_feature_column_types(train_enh)
cat_cols_with_sid = cat_cols + ["store_id"]
num_cols_no_sid = [c for c in num_cols if c != "store_id"]

features_with_time = [c for c in train_enh.columns if c in (num_cols_no_sid + cat_cols_with_sid) or c in ["t_years", "date", "store_id"]]

print(f"Total features: {len(num_cols_no_sid)} numeric + {len(cat_cols_with_sid)} categorical")

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_enh))

# Tuned models
models = {
    "CatBoost_Tuned": cb.CatBoostRegressor(iterations=750, learning_rate=0.03, depth=6, l2_leaf_reg=4.5, random_state=42, verbose=0),
    "LightGBM_Tuned": lgb.LGBMRegressor(n_estimators=550, learning_rate=0.028, num_leaves=28, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.5, random_state=42, verbose=-1),
    "XGBoost_Tuned": xgb.XGBRegressor(n_estimators=550, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.0, tree_method="hist", random_state=42),
    "Ridge": Ridge(alpha=3.0, random_state=42)
}

fold_scores = {m: [] for m in models}
oof_preds_log = {m: [] for m in models}
oof_trues = []

print("\n--- 2. Running 4-Fold CV Benchmark with Enhanced Features ---")
for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
    tr = train_enh.iloc[tr_idx]
    va = train_enh.iloc[va_idx]
    y_va = va["revenue"].values
    oof_trues.append(y_va)
    
    f_str = []
    for name, raw_m in models.items():
        # For tree models, include store_id in categorical_cols. For Ridge, use cat_cols without store_id to avoid sparsity overfitting
        use_cats = cat_cols_with_sid if "Ridge" not in name else cat_cols
        use_nums = num_cols_no_sid if "Ridge" not in name else [c for c in num_cols if c != "store_id"]
        
        pipe = build_leakage_free_pipeline(clone(raw_m), use_nums, use_cats, use_trend_aware=True)
        pipe.fit(tr[features_with_time], tr["revenue"].values)
        pred = pipe.predict(va[features_with_time])
        score = rmsle(y_va, pred)
        fold_scores[name].append(score)
        oof_preds_log[name].append(np.log1p(np.maximum(0, pred)))
        f_str.append(f"{name}: {score:.4f}")
    print(f"Fold {f_idx}: " + " | ".join(f_str))

print("\n=== Model Benchmark across 4 Folds ===")
for name in models:
    sc = fold_scores[name]
    print(f"  {name:20s}: F1={sc[0]:.4f}, F2={sc[1]:.4f}, F3={sc[2]:.4f}, F4={sc[3]:.4f} | Mean={np.mean(sc):.4f}")

# Full OOF Optimization
all_trues = np.concatenate(oof_trues)
all_trues_log = np.log1p(np.maximum(0, all_trues))
model_names = list(models.keys())
P_oof_log = np.column_stack([np.concatenate(oof_preds_log[m]) for m in model_names])

def obj(w):
    w_norm = w / np.sum(w)
    pred_log = P_oof_log @ w_norm
    return np.mean((pred_log - all_trues_log) ** 2)

res = minimize(obj, [0.25]*4, bounds=[(0, 1)]*4, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
opt_w = res.x / np.sum(res.x)

print("\n=== Optimal Ensemble Weights on Full OOF ===")
for name, w in zip(model_names, opt_w):
    print(f"  * {name:20s}: {w*100:.2f}%")

ens_pred_full = np.maximum(0, np.expm1(P_oof_log @ opt_w))
full_oof_rmsle = rmsle(all_trues, ens_pred_full)
print(f"\n>> FULL OOF RMSLE (Enhanced Ensemble): {full_oof_rmsle:.5f}")

# Compare fold by fold ensemble
ens_fold_scores = []
for f_idx in range(4):
    y_f = oof_trues[f_idx]
    P_f_log = np.column_stack([oof_preds_log[m][f_idx] for m in model_names])
    pred_f = np.maximum(0, np.expm1(P_f_log @ opt_w))
    sc = rmsle(y_f, pred_f)
    ens_fold_scores.append(sc)
    print(f"  Fold {f_idx+1} Ensemble RMSLE: {sc:.4f}")

print(f">> MEAN 4-FOLD ENSEMBLE RMSLE: {np.mean(ens_fold_scores):.5f}")
