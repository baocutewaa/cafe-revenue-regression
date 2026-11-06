import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import TimeSeriesExpandingCV, get_starter_holdout_split, compute_weekday_median_baseline
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types
from src.step6_model_train import get_candidate_models, evaluate_predictions, WeightedBlendEnsemble

train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)
numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(train_final)
features = numeric_cols + categorical_cols
features_with_time = [c for c in train_final.columns if c in features or c in ["t_years", "date"]]

print(f"Total features: {len(features)}")
print(f"Numeric features ({len(numeric_cols)}):", numeric_cols)
print(f"Categorical features ({len(categorical_cols)}):", categorical_cols)

# Run Expanding CV evaluation for current models
cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))

candidate_models = get_candidate_models()
models_to_test = {
    "Baseline (Weekday Median)": None,
    "LightGBM": candidate_models["LightGBM"],
    "CatBoost": candidate_models["CatBoost"],
    "XGBoost": candidate_models["XGBoost"],
    "Ridge": candidate_models["Ridge (L2 Linear)"]
}

for name, model in models_to_test.items():
    print(f"\n--- Testing {name} ---")
    fold_scores = []
    for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        tr_df = train_final.iloc[tr_idx]
        va_df = train_final.iloc[va_idx]
        y_va = va_df["revenue"].values
        
        if model is None:
            pred_va = compute_weekday_median_baseline(tr_df, va_df)
        else:
            pipe = build_leakage_free_pipeline(clone(model), numeric_cols, categorical_cols, use_trend_aware=True)
            pipe.fit(tr_df[features_with_time], tr_df["revenue"].values)
            pred_va = pipe.predict(va_df[features_with_time])
            
        score = rmsle(y_va, pred_va)
        fold_scores.append(score)
        print(f"  Fold {f_idx} RMSLE: {score:.4f}")
    print(f"  Mean CV: {np.mean(fold_scores):.4f} | Fold 1 (Tet): {fold_scores[0]:.4f} | Fold 4 (Q4): {fold_scores[3]:.4f}")
