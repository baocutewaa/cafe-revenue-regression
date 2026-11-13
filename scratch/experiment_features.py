import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from step1_collect_prepare import rmsle
from step3_preprocess import run_step3, TET_DATES
from step4_split import TimeSeriesExpandingCV, get_expanding_window_folds
from step5_scale import get_feature_column_types, build_leakage_free_pipeline

print("--- 1. Loading baseline clean data ---")
train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))

def evaluate_model_cv(model, tr_data, feature_cols, cat_cols, model_name="Model"):
    scores = []
    features_with_time = [c for c in tr_data.columns if c in feature_cols or c in ["t_years", "date", "store_id"]]
    for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        tr = tr_data.iloc[tr_idx]
        va = tr_data.iloc[va_idx]
        y_va = va["revenue"].values
        
        num_c = [c for c in feature_cols if c not in cat_cols]
        pipe = build_leakage_free_pipeline(clone(model), num_c, cat_cols, use_trend_aware=True)
        pipe.fit(tr[features_with_time], tr["revenue"].values)
        pred = pipe.predict(va[features_with_time])
        score = rmsle(y_va, pred)
        scores.append(score)
    print(f"  {model_name:25s}: F1={scores[0]:.4f}, F2={scores[1]:.4f}, F3={scores[2]:.4f}, F4={scores[3]:.4f} | Mean={np.mean(scores):.4f}")
    return scores

# Baseline test
num_cols, cat_cols, ign_cols = get_feature_column_types(train_final)
base_cb = cb.CatBoostRegressor(iterations=650, learning_rate=0.035, depth=6, l2_leaf_reg=4.0, random_state=42, verbose=0)
base_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=26, subsample=0.8, colsample_bytree=0.75, min_child_samples=20, reg_alpha=1.2, reg_lambda=1.5, random_state=42, verbose=-1)

print("\n--- Current Baseline Features (store_id ignored) ---")
evaluate_model_cv(base_cb, train_final, num_cols + cat_cols, cat_cols, "CatBoost (Baseline)")
evaluate_model_cv(base_lgb, train_final, num_cols + cat_cols, cat_cols, "LightGBM (Baseline)")

print("\n--- Test 1: Include store_id in categorical_cols ---")
cat_cols_with_sid = cat_cols + ["store_id"]
num_cols_no_sid = [c for c in num_cols if c != "store_id"]
evaluate_model_cv(base_cb, train_final, num_cols_no_sid + cat_cols_with_sid, cat_cols_with_sid, "CatBoost (+store_id)")
evaluate_model_cv(base_lgb, train_final, num_cols_no_sid + cat_cols_with_sid, cat_cols_with_sid, "LightGBM (+store_id)")
