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

train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)
numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(train_final)
features = numeric_cols + categorical_cols
features_with_time = [c for c in train_final.columns if c in features or c in ["t_years", "date"]]

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))
tr_idx, va_idx = folds[0]
tr_df = train_final.iloc[tr_idx]
va_df = train_final.iloc[va_idx].copy()

candidate_models = get_candidate_models()
lgb_pipe = build_leakage_free_pipeline(clone(candidate_models["LightGBM"]), numeric_cols, categorical_cols, use_trend_aware=True)
lgb_pipe.fit(tr_df[features_with_time], tr_df["revenue"].values)
va_df["pred_lgb"] = lgb_pipe.predict(va_df[features_with_time])

cb_pipe = build_leakage_free_pipeline(clone(candidate_models["CatBoost"]), numeric_cols, categorical_cols, use_trend_aware=True)
cb_pipe.fit(tr_df[features_with_time], tr_df["revenue"].values)
va_df["pred_cb"] = cb_pipe.predict(va_df[features_with_time])

va_df["log_true"] = np.log1p(va_df["revenue"])
va_df["log_pred_lgb"] = np.log1p(va_df["pred_lgb"])
va_df["log_error_lgb"] = va_df["log_pred_lgb"] - va_df["log_true"]
va_df["abs_log_error_lgb"] = np.abs(va_df["log_error_lgb"])

print("Overall Fold 1 RMSLE LGB:", rmsle(va_df["revenue"], va_df["pred_lgb"]))
print("Overall Fold 1 RMSLE CB:", rmsle(va_df["revenue"], va_df["pred_cb"]))

# Let's check RMSLE by time periods in Q1 2025:
# 1. Normal Jan (days_from_tet < -14)
# 2. Pre-Tet (-14 <= days_from_tet < -2)
# 3. Tet core (-2 <= days_from_tet <= 2)
# 4. Post-Tet recovery (3 <= days_from_tet <= 9)
# 5. Rest of Q1 (days_from_tet > 9)
tet_2025 = pd.Timestamp("2025-01-29")
va_df["days_from_tet"] = (va_df["date"] - tet_2025).dt.days

conditions = [
    va_df["days_from_tet"] < -14,
    va_df["days_from_tet"].between(-14, -3),
    va_df["days_from_tet"].between(-2, 2),
    va_df["days_from_tet"].between(3, 9),
    va_df["days_from_tet"] > 9
]
labels = ["1. Normal Jan (early)", "2. Pre-Tet Shopping", "3. Tet Core (-2 to 2)", "4. Post-Tet Recovery (3 to 9)", "5. Post-Tet Normal (>9)"]
va_df["period"] = np.select(conditions, labels, default="Other")

print("\n--- RMSLE by Period in Fold 1 (Q1 2025) ---")
for p in labels:
    sub = va_df[va_df["period"] == p]
    sc_lgb = rmsle(sub["revenue"], sub["pred_lgb"])
    sc_cb = rmsle(sub["revenue"], sub["pred_cb"])
    print(f"Period: {p:32s} | Rows: {len(sub):4d} | LGB RMSLE: {sc_lgb:.4f} | CB RMSLE: {sc_cb:.4f} | Mean True: {sub['revenue'].mean():.2f} | Mean LGB Pred: {sub['pred_lgb'].mean():.2f}")

# Check the biggest errors in Tet Core:
tet_core = va_df[va_df["period"] == "3. Tet Core (-2 to 2)"]
print("\nTet Core Errors by store_type:")
print(tet_core.groupby("store_type").apply(lambda g: pd.Series({
    "count": len(g),
    "true_mean": g["revenue"].mean(),
    "lgb_pred_mean": g["pred_lgb"].mean(),
    "lgb_rmsle": rmsle(g["revenue"], g["pred_lgb"]),
    "cb_rmsle": rmsle(g["revenue"], g["pred_cb"])
})))
