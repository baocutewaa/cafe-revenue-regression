import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import TimeSeriesExpandingCV
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types

train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

train_fixed = train_final.copy()
train_fixed.loc[(train_fixed["store_id"] == "S17") & (train_fixed["date"] == "2024-02-10"), "revenue"] = 8.97
train_fixed.loc[(train_fixed["store_id"] == "S29") & (train_fixed["date"] == "2024-09-02"), "revenue"] = 35.90

tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29"), 2026: pd.Timestamp("2026-02-17")}
train_fixed["days_from_tet_exact"] = train_fixed.apply(
    lambda r: (r["date"] - tet_map[r["date"].year]).days if r["date"].year in tet_map else 999, axis=1
)

# Add holiday interaction flags
is_work = train_fixed["store_type"].isin(["office", "university"]).astype(int)
is_retail = train_fixed["store_type"].isin(["mall", "street", "residential"]).astype(int)
train_fixed["is_holiday_work_drop"] = train_fixed["is_national_holiday"] * is_work
train_fixed["is_holiday_retail_boost"] = train_fixed["is_national_holiday"] * is_retail

cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_fixed))

models = {
    "CatBoost": cb.CatBoostRegressor(iterations=550, learning_rate=0.04, depth=6, l2_leaf_reg=3.5, random_state=42, verbose=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=450, learning_rate=0.035, num_leaves=28, subsample=0.8, colsample_bytree=0.75, min_child_samples=20, reg_alpha=1.2, reg_lambda=1.2, random_state=42, verbose=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=450, learning_rate=0.035, max_depth=5, subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=2.0, tree_method="hist", random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42)
}

fold_scores = {m: [] for m in models}
oof_preds = {m: [] for m in models}
oof_trues = []

for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
    tr = train_fixed.iloc[tr_idx].copy()
    va = train_fixed.iloc[va_idx].copy()
    
    tr_normal = tr[(tr["days_from_tet_exact"] < -14) | (tr["days_from_tet_exact"] > 14)]
    tr_normal = tr_normal[(tr_normal["is_promo"] == 0) & (tr_normal["is_heavy_rain"] == 0)]
    tr_base = tr_normal.groupby(["store_id", "dow"])["revenue"].median()
    
    tr_norm_col = tr.set_index(["store_id", "dow"]).index.map(tr_base).fillna(tr["revenue"].median())
    tr["temp_ratio"] = tr["revenue"] / tr_norm_col
    
    tet_mask = tr["days_from_tet_exact"].between(-18, 18)
    tet_curve = tr[tet_mask].groupby(["store_type", "days_from_tet_exact"])["temp_ratio"].median().to_dict()
    tr = tr.drop(columns=["temp_ratio"])
    
    def get_tet_log_factor(df):
        return np.array([np.log(np.maximum(0.2, tet_curve.get((st, d), 1.0))) if -18 <= d <= 18 else 0.0 
                         for st, d in zip(df["store_type"], df["days_from_tet_exact"])])
        
    tr["feat_tet_log_factor"] = get_tet_log_factor(tr)
    va["feat_tet_log_factor"] = get_tet_log_factor(va)
    
    numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(tr)
    features = numeric_cols + categorical_cols
    features_with_time = [c for c in tr.columns if c in features or c in ["t_years", "date"]]
    
    y_va = va["revenue"].values
    oof_trues.append(y_va)
    
    for name, raw_model in models.items():
        pipe = build_leakage_free_pipeline(clone(raw_model), numeric_cols, categorical_cols, use_trend_aware=True)
        pipe.fit(tr[features_with_time], tr["revenue"].values)
        pred = pipe.predict(va[features_with_time])
        score = rmsle(y_va, pred)
        fold_scores[name].append(score)
        oof_preds[name].append(pred)
        
    print(f"Fold {f_idx}: " + " | ".join([f"{name}: {fold_scores[name][-1]:.4f}" for name in models]))

print("\n=== Model Benchmark across 4 Folds ===")
for name in models:
    sc = fold_scores[name]
    print(f"{name:15s}: F1={sc[0]:.4f}, F2={sc[1]:.4f}, F3={sc[2]:.4f}, F4={sc[3]:.4f} | Mean={np.mean(sc):.4f}")

# Weighted Ensemble test with robust weights
# CatBoost: 0.40, LightGBM: 0.25, XGBoost: 0.20, Ridge: 0.15
ens_w = np.array([0.40, 0.25, 0.20, 0.15])
model_names = list(models.keys())

print(f"\n=== Ensemble (CatBoost 40%, LightGBM 25%, XGBoost 20%, Ridge 15%) ===")
ens_f_scores = []
for f_idx in range(4):
    y_f = oof_trues[f_idx]
    P_f_log = np.column_stack([np.log1p(np.maximum(0, oof_preds[m][f_idx])) for m in model_names])
    b_log = P_f_log @ ens_w
    pred_ens = np.maximum(0, np.expm1(b_log))
    sc = rmsle(y_f, pred_ens)
    ens_f_scores.append(sc)
    print(f"  Fold {f_idx+1} Ensemble RMSLE: {sc:.4f}")

print(f"\n>> OVERALL ENSEMBLE MEAN CV RMSLE: {np.mean(ens_f_scores):.4f}")
