import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import get_starter_holdout_split, compute_weekday_median_baseline
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types
import lightgbm as lgb
import shap

print("Loading step 3 data...")
train_final, _, _, _, _, _ = run_step3("data", verbose=False)
fit_clean, val_clean = get_starter_holdout_split(train_final, split_date="2025-10-01")

num_cols, cat_cols, ignore_cols = get_feature_column_types(fit_clean)
features = num_cols + cat_cols

print(f"Features: {len(num_cols)} num + {len(cat_cols)} cat = {len(features)}")

# Train LightGBM model pipeline
lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
pipeline = build_leakage_free_pipeline(lgb_model, num_cols, cat_cols)
pipeline.fit(fit_clean[features], fit_clean["revenue"].values)

# Get feature names from preprocessor
preprocessor = pipeline.regressor_.named_steps["preprocessor"]
feature_names = [n.replace("num__", "").replace("cat__", "") for n in preprocessor.get_feature_names_out()]
print("Transformed feature count:", len(feature_names))

# Feature importances from LightGBM model
fitted_lgb = pipeline.regressor_.named_steps["model"]
importances = fitted_lgb.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
print("\nTop 10 Features by LightGBM Importance:")
print(fi_df.head(10))

# Test SHAP TreeExplainer on a sample of 200 rows
X_val_trans = preprocessor.transform(val_clean[features])
explainer = shap.TreeExplainer(fitted_lgb)
shap_values = explainer.shap_values(X_val_trans[:200])
print("\nSHAP values shape:", shap_values.shape)
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
print("\nTop 10 Features by Mean |SHAP|:")
print(shap_df.head(10))
