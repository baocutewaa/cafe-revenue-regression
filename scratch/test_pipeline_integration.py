import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import Ridge
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from src.step1_collect_prepare import rmsle
from src.step4_split import TimeSeriesExpandingCV
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types

TET_DATES = {
    2024: pd.Timestamp("2024-02-10"),
    2025: pd.Timestamp("2025-01-29"),
    2026: pd.Timestamp("2026-02-17")
}

from src.step3_preprocess import run_step3

# Run step3 to get base data
train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

# Apply outlier fixes
train_fixed = train_final.copy()
train_fixed.loc[(train_fixed["store_id"] == "S17") & (train_fixed["date"] == "2024-02-10"), "revenue"] = 8.97
train_fixed.loc[(train_fixed["store_id"] == "S29") & (train_fixed["date"] == "2024-09-02"), "revenue"] = 35.90

train_fixed["days_from_tet_exact"] = train_fixed.apply(
    lambda r: (r["date"] - TET_DATES[r["date"].year]).days if r["date"].year in TET_DATES else 999, axis=1
)

is_work = train_fixed["store_type"].isin(["office", "university"]).astype(int)
is_retail = train_fixed["store_type"].isin(["mall", "street", "residential"]).astype(int)
train_fixed["is_holiday_work_drop"] = train_fixed["is_national_holiday"] * is_work
train_fixed["is_holiday_retail_boost"] = train_fixed["is_national_holiday"] * is_retail

class LeakageFreeTetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tet_curve_ = {}
        
    def fit(self, X, y):
        df = X.copy()
        df["revenue"] = y
        # dow median excluding tet and promo
        norm_mask = (df["days_from_tet_exact"] < -14) | (df["days_from_tet_exact"] > 14)
        norm_mask = norm_mask & (df["is_promo"] == 0)
        base = df[norm_mask].groupby(["store_id", "dow"])["revenue"].median()
        
        df_base = df.set_index(["store_id", "dow"]).index.map(base).fillna(df["revenue"].median())
        df["ratio"] = df["revenue"] / df_base
        
        tet_mask = df["days_from_tet_exact"].between(-18, 18)
        self.tet_curve_ = df[tet_mask].groupby(["store_type", "days_from_tet_exact"])["ratio"].median().to_dict()
        return self
        
    def transform(self, X):
        res = X.copy()
        factors = [
            np.log(np.maximum(0.2, self.tet_curve_.get((st, d), 1.0))) if -18 <= d <= 18 else 0.0
            for st, d in zip(res["store_type"], res["days_from_tet_exact"])
        ]
        res["feat_tet_log_factor"] = np.array(factors, dtype=float)
        return res

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
    
    tet_trans = LeakageFreeTetTransformer()
    tet_trans.fit(tr, tr["revenue"].values)
    tr = tet_trans.transform(tr)
    va = tet_trans.transform(va)
    
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

ens_w = np.array([0.40, 0.25, 0.20, 0.15])
model_names = list(models.keys())
ens_f_scores = []
for f_idx in range(4):
    y_f = oof_trues[f_idx]
    P_f_log = np.column_stack([np.log1p(np.maximum(0, oof_preds[m][f_idx])) for m in model_names])
    b_log = P_f_log @ ens_w
    pred_ens = np.maximum(0, np.expm1(b_log))
    sc = rmsle(y_f, pred_ens)
    ens_f_scores.append(sc)
    print(f"Fold {f_idx+1} Ensemble RMSLE: {sc:.4f}")

print(f"\nFinal Ensemble Mean CV RMSLE: {np.mean(ens_f_scores):.4f}")
