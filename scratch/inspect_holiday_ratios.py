import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
stores["store_type"] = stores["store_type"].str.strip().str.lower()
train = train.merge(stores[["store_id", "store_type"]], on="store_id")

# Clean 1000x and S17, S29
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0
train.loc[(train["store_id"] == "S17") & (train["date"] == "2024-02-10"), "revenue"] = 8.97
train.loc[(train["store_id"] == "S29") & (train["date"] == "2024-09-02"), "revenue"] = 35.90

train["dow"] = train["date"].dt.dayofweek
sw_med = train.groupby(["store_id", "dow"])["revenue"].transform("median")
train["sw_ratio"] = train["revenue"] / sw_med

events = {
    "New Year's Day (Jan 1)": ["2024-01-01", "2025-01-01"],
    "Valentine's Day (Feb 14)": ["2024-02-14", "2025-02-14"],
    "Women's Day (Mar 8)": ["2024-03-08", "2025-03-08"],
    "Ong Cong Ong Tao": ["2024-02-03", "2025-01-22"]
}

for ev_name, dates in events.items():
    print(f"\n=== {ev_name} ===")
    sub = train[train["date"].isin([pd.Timestamp(d) for d in dates])]
    piv = sub.groupby(["date", "store_type"])["sw_ratio"].mean().unstack()
    print(piv.round(2))
