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

# Clean fixes for the 2 missed outliers
train_fixed = train_final.copy()
train_fixed.loc[(train_fixed["store_id"] == "S17") & (train_fixed["date"] == "2024-02-10"), "revenue"] = 8.97
train_fixed.loc[(train_fixed["store_id"] == "S29") & (train_fixed["date"] == "2024-09-02"), "revenue"] = 35.90

tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29"), 2026: pd.Timestamp("2026-02-17")}
train_fixed["days_from_tet_exact"] = train_fixed.apply(
    lambda r: (r["date"] - tet_map[r["date"].year]).days if r["date"].year in tet_map else 999, axis=1
)

# Let's inspect city x store_type vs store_type alone for Tet
cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_fixed))

for grouping in ["type_only", "city_type"]:
    print(f"\n=== Testing Tet grouping: {grouping} ===")
    f1_scores = []
    for f_idx, (tr_idx, va_idx) in enumerate(folds[:1], 1): # Fold 1
        tr = train_fixed.iloc[tr_idx].copy()
        va = train_fixed.iloc[va_idx].copy()
        
        tr_normal = tr[(tr["days_from_tet_exact"] < -14) | (tr["days_from_tet_exact"] > 14)]
        tr_normal = tr_normal[(tr_normal["is_promo"] == 0) & (tr_normal["is_heavy_rain"] == 0)]
        tr_base = tr_normal.groupby(["store_id", "dow"])["revenue"].median()
        
        tr_norm_col = tr.set_index(["store_id", "dow"]).index.map(tr_base).fillna(tr["revenue"].median())
        tr["temp_ratio"] = tr["revenue"] / tr_norm_col
        
        tet_mask = tr["days_from_tet_exact"].between(-18, 18)
        if grouping == "type_only":
            curve = tr[tet_mask].groupby(["store_type", "days_from_tet_exact"])["temp_ratio"].median().to_dict()
            def get_factor(df):
                return np.array([np.log(np.maximum(0.2, curve.get((st, d), 1.0))) if -18 <= d <= 18 else 0.0 for st, d in zip(df["store_type"], df["days_from_tet_exact"])])
        else:
            curve = tr[tet_mask].groupby(["city", "store_type", "days_from_tet_exact"])["temp_ratio"].median().to_dict()
            type_curve = tr[tet_mask].groupby(["store_type", "days_from_tet_exact"])["temp_ratio"].median().to_dict()
            def get_factor(df):
                return np.array([np.log(np.maximum(0.2, curve.get((c, st, d), type_curve.get((st, d), 1.0)))) if -18 <= d <= 18 else 0.0 for c, st, d in zip(df["city"], df["store_type"], df["days_from_tet_exact"])])
                
        tr["feat_tet_log_factor"] = get_factor(tr)
        va["feat_tet_log_factor"] = get_factor(va)
        tr = tr.drop(columns=["temp_ratio"])
        
        numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(tr)
        features = numeric_cols + categorical_cols
        features_with_time = [c for c in tr.columns if c in features or c in ["t_years", "date"]]
        
        candidate_models = get_candidate_models()
        pipe_cb = build_leakage_free_pipeline(clone(candidate_models["CatBoost"]), numeric_cols, categorical_cols, use_trend_aware=True)
        pipe_cb.fit(tr[features_with_time], tr["revenue"].values)
        pred_cb = pipe_cb.predict(va[features_with_time])
        print(f"  CatBoost Fold 1 RMSLE: {rmsle(va['revenue'], pred_cb):.4f}")
