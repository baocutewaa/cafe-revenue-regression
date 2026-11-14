import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from sklearn.base import clone

from step1_collect_prepare import rmsle
from step3_preprocess import run_step3
from step4_split import TimeSeriesExpandingCV
from step5_scale import get_feature_column_types, build_leakage_free_pipeline
from step6_model_train import get_candidate_models

train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)
num_cols, cat_cols, ign_cols = get_feature_column_types(train_final)
features = num_cols + cat_cols
features_with_time = [c for c in train_final.columns if c in features or c in ["t_years", "date", "store_id"]]

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_final))

# Run on Fold 1 (Jan-Mar 2025, exact mirror of Test Q1 2026)
f1_tr_idx, f1_va_idx = folds[0]
tr = train_final.iloc[f1_tr_idx]
va = train_final.iloc[f1_va_idx].copy()

cand = get_candidate_models()
cb_pipe = build_leakage_free_pipeline(clone(cand["CatBoost"]), num_cols, cat_cols, use_trend_aware=True)
lgb_pipe = build_leakage_free_pipeline(clone(cand["LightGBM"]), num_cols, cat_cols, use_trend_aware=True)

cb_pipe.fit(tr[features_with_time], tr["revenue"].values)
lgb_pipe.fit(tr[features_with_time], tr["revenue"].values)

va["cb_pred"] = cb_pipe.predict(va[features_with_time])
va["lgb_pred"] = lgb_pipe.predict(va[features_with_time])
va["ens_pred"] = np.expm1(0.55 * np.log1p(va["cb_pred"]) + 0.45 * np.log1p(va["lgb_pred"]))

va["log_true"] = np.log1p(va["revenue"])
va["log_pred"] = np.log1p(va["ens_pred"])
va["log_err"] = va["log_pred"] - va["log_true"]
va["sq_log_err"] = va["log_err"] ** 2

print(f"Overall Fold 1 RMSLE: {np.sqrt(va['sq_log_err'].mean()):.4f}")
print(f"Overall Mean Error (Bias): {va['log_err'].mean():.4f} (positive = over-predict, negative = under-predict)")

print("\n--- RMSLE and Bias by Store Type ---")
st_diag = va.groupby("store_type").apply(lambda g: pd.Series({
    "count": len(g),
    "rmsle": np.sqrt(g["sq_log_err"].mean()),
    "mean_bias": g["log_err"].mean(),
    "true_mean": g["revenue"].mean(),
    "pred_mean": g["ens_pred"].mean()
}))
print(st_diag.round(4))

print("\n--- Top 10 Stores with Highest RMSLE in Fold 1 ---")
store_diag = va.groupby("store_id").apply(lambda g: pd.Series({
    "store_type": g["store_type"].iloc[0],
    "city": g["city"].iloc[0],
    "rmsle": np.sqrt(g["sq_log_err"].mean()),
    "bias": g["log_err"].mean(),
    "true_median": g["revenue"].median(),
    "pred_median": g["ens_pred"].median()
})).sort_values("rmsle", ascending=False)
print(store_diag.head(10).round(4))

print("\n--- Top 10 Dates with Highest Mean Squared Error in Fold 1 ---")
date_diag = va.groupby("date").apply(lambda g: pd.Series({
    "rmsle": np.sqrt(g["sq_log_err"].mean()),
    "bias": g["log_err"].mean(),
    "holiday": g["holiday_name_clean"].iloc[0],
    "days_from_tet": g["days_from_tet_clipped"].iloc[0]
})).sort_values("rmsle", ascending=False)
print(date_diag.head(10).round(4))
