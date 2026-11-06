import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import TimeSeriesExpandingCV
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types
from src.step6_model_train import get_candidate_models

# Load data
train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

# Check S17 on 2024-02-10 in train_final
s17_row = train_final[(train_final["store_id"] == "S17") & (train_final["date"] == "2024-02-10")]
print("Current S17 on 2024-02-10 revenue:", s17_row["revenue"].values)

# Let's fix S17 and S29
train_fixed = train_final.copy()
train_fixed.loc[(train_fixed["store_id"] == "S17") & (train_fixed["date"] == "2024-02-10"), "revenue"] = 8.97
train_fixed.loc[(train_fixed["store_id"] == "S29") & (train_fixed["date"] == "2024-09-02"), "revenue"] = 35.90

numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(train_fixed)
features = numeric_cols + categorical_cols
features_with_time = [c for c in train_fixed.columns if c in features or c in ["t_years", "date"]]

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_fixed))
candidate_models = get_candidate_models()

for name in ["LightGBM", "CatBoost", "XGBoost", "Ridge (L2 Linear)"]:
    fold_scores = []
    for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        tr_df = train_fixed.iloc[tr_idx]
        va_df = train_fixed.iloc[va_idx]
        y_va = va_df["revenue"].values
        
        pipe = build_leakage_free_pipeline(clone(candidate_models[name]), numeric_cols, categorical_cols, use_trend_aware=True)
        pipe.fit(tr_df[features_with_time], tr_df["revenue"].values)
        pred_va = pipe.predict(va_df[features_with_time])
        fold_scores.append(rmsle(y_va, pred_va))
    print(f"{name:20s} | Fold 1 (Tet): {fold_scores[0]:.4f} | F2: {fold_scores[1]:.4f} | F3: {fold_scores[2]:.4f} | F4: {fold_scores[3]:.4f} | Mean: {np.mean(fold_scores):.4f}")
