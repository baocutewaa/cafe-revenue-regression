import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

sub = pd.read_csv("submission.csv")
test = pd.read_csv("data/test.csv")
stores = pd.read_csv("data/stores.csv")
train = pd.read_csv("data/train.csv")

test_sub = test.merge(sub, on="id")
print("Total rows in test_sub:", len(test_sub))
print("Stores in test:", test_sub["store_id"].nunique())

# Let's inspect store S01, S45, S19 in test
stores_clean = pd.read_csv("data/stores.csv")
print("\nStores info for S01, S19, S45, S12:")
print(stores[stores["store_id"].isin(["S01", "S19", "S45", "S12"])][["store_id", "open_date", "store_type", "city", "seats"]])

# Check train dates for S01, S19, S45
for sid in ["S01", "S19", "S45"]:
    tr_s = train[train["store_id"] == sid]
    print(f"\nStore {sid} in train: {len(tr_s)} rows, date min: {tr_s['date'].min()}, max: {tr_s['date'].max()}, rev mean: {tr_s['revenue'].mean():.2f}")
    
# Let's check average predicted revenue by store_type in test vs train Q1 2025
test_sub = test_sub.merge(stores[["store_id", "store_type"]], on="store_id")
train["date"] = pd.to_datetime(train["date"])
train_q1_2025 = train[(train["date"] >= "2025-01-01") & (train["date"] <= "2025-03-31")].merge(stores[["store_id", "store_type"]], on="store_id")

print("\nMean revenue Q1 2025 (Train) vs Q1 2026 (Test Predictions) by store_type:")
q1_2025_agg = train_q1_2025.groupby("store_type")["revenue"].mean()
test_agg = test_sub.groupby("store_type")["revenue"].mean()
comp = pd.DataFrame({"Q1_2025_Actual": q1_2025_agg, "Q1_2026_Pred": test_agg})
comp["diff_pct"] = (comp["Q1_2026_Pred"] - comp["Q1_2025_Actual"]) / comp["Q1_2025_Actual"] * 100
print(comp)
