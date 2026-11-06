import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import TimeSeriesExpandingCV
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types
from src.step6_model_train import get_candidate_models, WeightedBlendEnsemble

train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)
numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(train_final)
features = numeric_cols + categorical_cols
features_with_time = [c for c in train_final.columns if c in features or c in ["t_years", "date"]]

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))

candidate_models = get_candidate_models()
ensemble_members = ["LightGBM", "CatBoost", "XGBoost", "Ridge (L2 Linear)"]

# Evaluate default weights: LightGBM: 0.43, CatBoost: 0.281, XGBoost: 0.289, Ridge: 0.0
weights = [0.43, 0.281, 0.289, 0.0]

print("=== Evaluating Current Default Ensemble across 4 Folds ===")
fold_scores = []
for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
    tr_df = train_final.iloc[tr_idx]
    va_df = train_final.iloc[va_idx]
    y_va = va_df["revenue"].values
    
    preds_log = []
    for m in ensemble_members:
        pipe = build_leakage_free_pipeline(clone(candidate_models[m]), numeric_cols, categorical_cols, use_trend_aware=True)
        pipe.fit(tr_df[features_with_time], tr_df["revenue"].values)
        pred = pipe.predict(va_df[features_with_time])
        preds_log.append(np.log1p(np.maximum(0, pred)))
        
    P_log = np.column_stack(preds_log)
    blend_log = P_log @ np.array(weights)
    pred_blend = np.maximum(0, np.expm1(blend_log))
    score = rmsle(y_va, pred_blend)
    fold_scores.append(score)
    print(f"Fold {f_idx} RMSLE: {score:.4f}")

print(f"Ensemble Mean CV: {np.mean(fold_scores):.4f}")
