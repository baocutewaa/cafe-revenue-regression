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

train_final, test_final, _, _, _, _ = run_step3("data", verbose=False)

def add_advanced_features(df):
    d = df.copy()
    is_uni = (d["store_type"] == "university").astype(int)
    is_mall = (d["store_type"] == "mall").astype(int)
    is_street = (d["store_type"] == "street").astype(int)
    is_office = (d["store_type"] == "office").astype(int)
    is_res = (d["store_type"] == "residential").astype(int)
    
    d["promo_x_uni"] = d["is_promo"] * is_uni
    d["promo_x_mall"] = d["is_promo"] * is_mall
    d["promo_x_street"] = d["is_promo"] * is_street
    d["promo_x_dow"] = d["is_promo"] * d["dow"]
    
    d_tet = d["days_from_tet_clipped"]
    d["is_pre_tet_mall_rush"] = (is_mall & (d_tet.between(-7, -3))).astype(int)
    d["is_pre_tet_street_rush"] = (is_street & (d_tet.between(-7, -3))).astype(int)
    d["is_tet_core_slump"] = ((is_street | is_res) & (d_tet.between(-2, 2))).astype(int)
    d["is_post_tet_office_ramp"] = (is_office & (d_tet.between(3, 9))).astype(int)
    d["is_post_tet_mall_bounce"] = (is_mall & (d_tet.between(2, 6))).astype(int)
    d["tet_dist_smooth"] = np.exp(-0.5 * (d_tet / 5.0) ** 2)
    d["rain_heavy_outdoor"] = d["is_heavy_rain"] * (is_street | is_res | is_uni).astype(int)
    return d

train_enh = add_advanced_features(train_final)
num_cols, cat_cols, ign_cols = get_feature_column_types(train_enh)
cat_cols_with_sid = cat_cols + ["store_id"]
num_cols_no_sid = [c for c in num_cols if c != "store_id"]
features_with_time = [c for c in train_enh.columns if c in (num_cols_no_sid + cat_cols_with_sid) or c in ["t_years", "date", "store_id"]]

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_enh))

models = {
    "CatBoost": cb.CatBoostRegressor(iterations=750, learning_rate=0.03, depth=6, l2_leaf_reg=4.5, random_state=42, verbose=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=550, learning_rate=0.028, num_leaves=28, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.5, random_state=42, verbose=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=550, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.0, tree_method="hist", random_state=42),
    "Ridge": Ridge(alpha=3.0, random_state=42)
}

oof_preds_log = {m: [] for m in models}
oof_trues = []

for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
    tr = train_enh.iloc[tr_idx]
    va = train_enh.iloc[va_idx]
    y_va = va["revenue"].values
    oof_trues.append(y_va)
    for name, raw_m in models.items():
        use_cats = cat_cols_with_sid if name != "Ridge" else cat_cols
        use_nums = num_cols_no_sid if name != "Ridge" else [c for c in num_cols if c != "store_id"]
        pipe = build_leakage_free_pipeline(clone(raw_m), use_nums, use_cats, use_trend_aware=True)
        pipe.fit(tr[features_with_time], tr["revenue"].values)
        pred = pipe.predict(va[features_with_time])
        oof_preds_log[name].append(np.log1p(np.maximum(0, pred)))

all_trues = np.concatenate(oof_trues)
P_oof_log = np.column_stack([np.concatenate(oof_preds_log[m]) for m in models.keys()])

# Grid search weights over step 0.05
best_w = None
best_score = 999.0
weights_grid = []
for w1 in np.linspace(0.2, 0.6, 9):
    for w2 in np.linspace(0.1, 0.4, 7):
        for w3 in np.linspace(0.1, 0.4, 7):
            w4 = 1.0 - w1 - w2 - w3
            if 0.05 <= w4 <= 0.25:
                w = np.array([w1, w2, w3, w4])
                pred_log = P_oof_log @ w
                pred = np.maximum(0, np.expm1(pred_log))
                sc = rmsle(all_trues, pred)
                if sc < best_score:
                    best_score = sc
                    best_w = w

print("Grid search best weights (CatBoost, LightGBM, XGBoost, Ridge):", best_w)
print(f"Best Full OOF RMSLE: {best_score:.5f}")

# Calculate fold-by-fold score with best_w
f_scores = []
for f_idx in range(4):
    y_f = oof_trues[f_idx]
    P_f_log = np.column_stack([oof_preds_log[m][f_idx] for m in models.keys()])
    pred_f = np.maximum(0, np.expm1(P_f_log @ best_w))
    sc = rmsle(y_f, pred_f)
    f_scores.append(sc)
    print(f"Fold {f_idx+1} RMSLE: {sc:.4f}")
print(f">> MEAN CV RMSLE with best weights: {np.mean(f_scores):.5f}")
