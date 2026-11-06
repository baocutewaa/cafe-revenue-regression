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

# Clean fixes
train_fixed = train_final.copy()
train_fixed.loc[(train_fixed["store_id"] == "S17") & (train_fixed["date"] == "2024-02-10"), "revenue"] = 8.97
train_fixed.loc[(train_fixed["store_id"] == "S29") & (train_fixed["date"] == "2024-09-02"), "revenue"] = 35.90

tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29"), 2026: pd.Timestamp("2026-02-17")}
train_fixed["days_from_tet_exact"] = train_fixed.apply(
    lambda r: (r["date"] - tet_map[r["date"].year]).days if r["date"].year in tet_map else 999, axis=1
)

# Let's test computing Tet effect curve on Fold 1 train (2024) and see its prediction on Fold 1 val (2025)
cv = TimeSeriesExpandingCV()
folds = list(cv.split(train_fixed))
tr_idx, va_idx = folds[0]
tr = train_fixed.iloc[tr_idx].copy()
va = train_fixed.iloc[va_idx].copy()

# Compute normal median for each store and dow in train
tr_normal = tr[(tr["days_from_tet_exact"] < -21) | (tr["days_from_tet_exact"] > 21)]
tr_base = tr_normal.groupby(["store_id", "dow"])["revenue"].median()
tr["normal_rev"] = tr.set_index(["store_id", "dow"]).index.map(tr_base).fillna(tr["revenue"].median())
tr["ratio_to_normal"] = tr["revenue"] / tr["normal_rev"]

# Learn Tet curve by store_type and days_from_tet_exact
tet_train = tr[tr["days_from_tet_exact"].between(-21, 21)]
tet_curve = tet_train.groupby(["store_type", "days_from_tet_exact"])["ratio_to_normal"].median().to_dict()

# Print sample of learned tet_curve
print("Sample learned Tet curve (2024 train):")
for st in ["mall", "office", "residential", "street", "university"]:
    sample = {d: round(tet_curve.get((st, d), 1.0), 2) for d in range(-7, 8)}
    print(f"{st:12s}: {sample}")

# Apply tet_curve to va
va_base = va.set_index(["store_id", "dow"]).index.map(tr_base).fillna(tr["revenue"].median())
va_tet_ratio = [tet_curve.get((st, d), 1.0) for st, d in zip(va["store_type"], va["days_from_tet_exact"])]
va_pred_pure_tet = va_base * np.array(va_tet_ratio)

# What is the RMSLE of pure weekday_median with tet_curve on Fold 1?
pure_tet_rmsle = rmsle(va["revenue"], va_pred_pure_tet)
base_weekday_rmsle = rmsle(va["revenue"], va_base)
print(f"\nFold 1 Weekday Median:          {base_weekday_rmsle:.4f}")
print(f"Fold 1 Weekday Median + Tet Curve: {pure_tet_rmsle:.4f}")
