import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, HuberRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from step1_collect_prepare import rmsle
from step3_preprocess import run_step3
from step4_split import TimeSeriesExpandingCV
from step5_scale import get_feature_column_types, build_leakage_free_pipeline

train_final, test_final, _, _, _, _ = run_step3("data", verbose=False)
num_cols, cat_cols, ign_cols = get_feature_column_types(train_final)
features = num_cols + cat_cols
features_with_time = [c for c in train_final.columns if c in features or c in ["t_years", "date", "store_id"]]

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))

models = {
    "HistGB": HistGradientBoostingRegressor(max_iter=350, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.5, random_state=42),
    "Huber": HuberRegressor(max_iter=300, alpha=1.0),
    "CatBoost_Deep": cb.CatBoostRegressor(iterations=750, learning_rate=0.03, depth=6, l2_leaf_reg=4.5, random_state=42, verbose=0),
    "LightGBM_Tuned": lgb.LGBMRegressor(n_estimators=550, learning_rate=0.028, num_leaves=28, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.5, random_state=42, verbose=-1),
    "XGBoost_Tuned": xgb.XGBRegressor(n_estimators=550, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.0, tree_method="hist", random_state=42)
}

print("=== Evaluating Additional Models across 4 Folds ===")
for name, m in models.items():
    scores = []
    for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        tr = train_final.iloc[tr_idx]
        va = train_final.iloc[va_idx]
        pipe = build_leakage_free_pipeline(clone(m), num_cols, cat_cols, use_trend_aware=True)
        pipe.fit(tr[features_with_time], tr["revenue"].values)
        pred = pipe.predict(va[features_with_time])
        scores.append(rmsle(va["revenue"].values, pred))
    print(f"  {name:20s}: F1={scores[0]:.4f}, F2={scores[1]:.4f}, F3={scores[2]:.4f}, F4={scores[3]:.4f} | Mean={np.mean(scores):.4f}")
