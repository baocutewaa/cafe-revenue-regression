import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from step1_collect_prepare import rmsle
from step3_preprocess import run_step3
from step5_scale import get_feature_column_types, build_leakage_free_pipeline, MultiSeedBaggedPipeline
from step6_model_train import WeightedBlendEnsemble

print("Loading data...")
train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

def add_advanced_features(df):
    d = df.copy()
    is_uni = (d["store_type"] == "university").astype(int)
    is_mall = (d["store_type"] == "mall").astype(int)
    is_street = (d["store_type"] == "street").astype(int)
    is_office = (d["store_type"] == "office").astype(int)
    is_res = (d["store_type"] == "residential").astype(int)
    
    d["promo_x_uni"] = d["is_promo"] * is_uni
    d["promo_x_mall"] = d["is_promo"] * is_mall
    d["promo_x_street"] = d["is_promo"] * is_street
    d["promo_x_dow"] = d["is_promo"] * d["dow"]
    
    d_tet = d["days_from_tet_clipped"]
    d["is_pre_tet_mall_rush"] = (is_mall & (d_tet.between(-7, -3))).astype(int)
    d["is_pre_tet_street_rush"] = (is_street & (d_tet.between(-7, -3))).astype(int)
    d["is_tet_core_slump"] = ((is_street | is_res) & (d_tet.between(-2, 2))).astype(int)
    d["is_post_tet_office_ramp"] = (is_office & (d_tet.between(3, 9))).astype(int)
    d["is_post_tet_mall_bounce"] = (is_mall & (d_tet.between(2, 6))).astype(int)
    d["tet_dist_smooth"] = np.exp(-0.5 * (d_tet / 5.0) ** 2)
    d["rain_heavy_outdoor"] = d["is_heavy_rain"] * (is_street | is_res | is_uni).astype(int)
    return d

train_enh = add_advanced_features(train_final)
test_enh = add_advanced_features(test_final)

num_cols, cat_cols, ign_cols = get_feature_column_types(train_enh)
cat_cols_with_sid = cat_cols + ["store_id"]
num_cols_no_sid = [c for c in num_cols if c != "store_id"]

features_with_time = [c for c in train_enh.columns if c in (num_cols_no_sid + cat_cols_with_sid) or c in ["t_years", "date", "store_id"]]

X_full = train_enh[features_with_time]
y_full = train_enh["revenue"].values
X_test = test_enh[features_with_time]

# 10 random seeds for variance reduction & private leaderboard stability
seeds = [42, 100, 2024, 7, 999, 123, 456, 789, 2025, 2026]

base_models = {
    "CatBoost": cb.CatBoostRegressor(iterations=750, learning_rate=0.03, depth=6, l2_leaf_reg=4.5, verbose=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=550, learning_rate=0.028, num_leaves=28, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.5, verbose=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=550, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.0, tree_method="hist"),
    "Ridge": Ridge(alpha=3.0, random_state=42)
}

weights = {
    "CatBoost": 0.60,
    "LightGBM": 0.10,
    "XGBoost": 0.10,
    "Ridge": 0.20
}

fitted_models = {}
for name, raw_m in base_models.items():
    print(f"Training {name} on 100% data...")
    use_cats = cat_cols_with_sid if name != "Ridge" else cat_cols
    use_nums = num_cols_no_sid if name != "Ridge" else [c for c in num_cols if c != "store_id"]
    
    if name != "Ridge":
        seed_pipes = []
        for s in seeds:
            m_s = clone(raw_m)
            if hasattr(m_s, "random_state"):
                m_s.set_params(random_state=s)
            elif hasattr(m_s, "random_seed"):
                m_s.set_params(random_seed=s)
            pipe = build_leakage_free_pipeline(m_s, use_nums, use_cats, use_trend_aware=True)
            pipe.fit(X_full, y_full)
            seed_pipes.append(pipe)
        fitted_models[name] = MultiSeedBaggedPipeline(seed_pipes)
    else:
        pipe = build_leakage_free_pipeline(clone(raw_m), use_nums, use_cats, use_trend_aware=True)
        pipe.fit(X_full, y_full)
        fitted_models[name] = pipe

final_ensemble = WeightedBlendEnsemble(models_dict=fitted_models, weights=weights)

# Predict on test
test_preds = final_ensemble.predict(X_test)
test_preds_clean = np.maximum(0.5, test_preds)

sub_df = pd.DataFrame({
    "id": test_final["id"].astype(int),
    "revenue": np.round(test_preds_clean, 2)
})

print("\n--- Test Predictions Summary ---")
print(sub_df["revenue"].describe().round(2))

# Compare with previous submission.csv
old_sub = pd.read_csv("submission.csv")
print("\n--- Comparison with Old Submission ---")
print("Old sub mean:", old_sub["revenue"].mean(), "New sub mean:", sub_df["revenue"].mean())
print("Old sub median:", old_sub["revenue"].median(), "New sub median:", sub_df["revenue"].median())
print("Correlation between old and new sub:", old_sub["revenue"].corr(sub_df["revenue"]))
print("RMSLE difference between old and new:", rmsle(old_sub["revenue"].values, sub_df["revenue"].values))

# Sanity checks
assert len(sub_df) == 3869, f"Row count mismatch: {len(sub_df)}"
assert sub_df["revenue"].isnull().sum() == 0, "Found NaNs in predictions"
assert (sub_df["revenue"] <= 0).sum() == 0, "Found non-positive predictions"
assert list(sub_df["id"]) == list(test_final["id"]), "IDs do not match test"

print("\nAll Sanity Checks Passed Successfully!")
