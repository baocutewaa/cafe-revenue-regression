import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv", parse_dates=["date"])
test = pd.read_csv("data/test.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
sub = pd.read_csv("submission.csv")

test_sub = test.merge(sub, on="id").merge(stores, on="store_id")

for sid in ["S01", "S19", "S45"]:
    sub_s = test_sub[test_sub["store_id"] == sid]
    tr_s = train[train["store_id"] == sid]
    print(f"\n--- Store {sid} ({sub_s['store_type'].iloc[0]}, {sub_s['city'].iloc[0]}, seats={sub_s['seats'].iloc[0]}, area={sub_s['area_m2'].iloc[0]}) ---")
    print(f"  Train count: {len(tr_s)}, Train min: {tr_s['date'].min() if len(tr_s)>0 else 'N/A'}, max: {tr_s['date'].max() if len(tr_s)>0 else 'N/A'}")
    print(f"  Train mean: {tr_s['revenue'].mean():.2f}, median: {tr_s['revenue'].median():.2f}")
    print(f"  Test count: {len(sub_s)}")
    print(f"  Test pred mean: {sub_s['revenue'].mean():.2f}, min: {sub_s['revenue'].min():.2f}, max: {sub_s['revenue'].max():.2f}")
    # Compare with similar stores in train
    sim = train.merge(stores, on="store_id")
    sim = sim[(sim["store_type"] == sub_s['store_type'].iloc[0]) & (sim["city"] == sub_s['city'].iloc[0])]
    print(f"  Similar mature stores in {sub_s['city'].iloc[0]} ({sub_s['store_type'].iloc[0]}): mean={sim['revenue'].mean():.2f}, median={sim['revenue'].median():.2f}")
