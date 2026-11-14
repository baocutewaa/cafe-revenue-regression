import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from step3_preprocess import run_step3
from step4_split import TimeSeriesExpandingCV, get_expanding_window_folds
from step5_scale import get_feature_column_types, build_leakage_free_pipeline
from step6_model_train import get_candidate_models
from step1_collect_prepare import rmsle

train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3("data", verbose=False)

print("=== CHECKING S01, S19, S45 IN TRAIN AND CURRENT ENCODER ===")
from step3_preprocess import DeSeasonalizedStoreStats
encoder = DeSeasonalizedStoreStats(shrinkage_m=10.0, alpha_ridge=1.0)
encoder.fit(train_final, train_final["revenue"].values)

for sid in ["S01", "S19", "S45"]:
    base_val = encoder.store_base_log_.get(sid, None)
    st = stores_clean[stores_clean["store_id"]==sid]["store_type"].values[0]
    city = stores_clean[stores_clean["store_id"]==sid]["city"].values[0]
    print(f"Store {sid} ({st}, {city}): store_base_log_rev = {base_val:.4f} (exp: {np.expm1(base_val):.2f})")
    # Compare with average store_base_log_rev for that store_type and city
    sim_stores = stores_clean[(stores_clean["store_type"]==st) & (stores_clean["city"]==city) & (~stores_clean["store_id"].isin(["S01", "S19", "S45", "S02"]))]
    sim_bases = [encoder.store_base_log_.get(s, np.nan) for s in sim_stores["store_id"]]
    print(f"  Similar mature stores in {city} {st}: mean log base = {np.mean(sim_bases):.4f} (exp: {np.expm1(np.mean(sim_bases)):.2f})")
