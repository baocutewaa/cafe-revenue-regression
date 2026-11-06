import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
stores["store_type"] = stores["store_type"].str.strip().str.lower()
train = train.merge(stores[["store_id", "store_type"]], on="store_id")

# Clean 1000x S12
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0

# Calculate sw_med on non-Tet, non-opening days
train["dow"] = train["date"].dt.dayofweek
sw_med = train.groupby(["store_id", "dow"])["revenue"].transform("median")
train["sw_ratio"] = train["revenue"] / sw_med

# Let's inspect rows with sw_ratio between 2.5 and 4.0
print("Rows with sw_ratio between 2.8 and 4.0:")
suspicious_high = train[train["sw_ratio"].between(2.8, 4.0)].sort_values(by="sw_ratio", ascending=False)
print(f"Count: {len(suspicious_high)}")
print(suspicious_high[["date", "store_id", "store_type", "is_promo", "revenue", "sw_ratio"]].to_string())

# Let's also inspect rows with sw_ratio between 0.14 and 0.25
print("\nRows with sw_ratio between 0.14 and 0.24:")
suspicious_low = train[train["sw_ratio"].between(0.14, 0.24)].sort_values(by="sw_ratio")
print(f"Count: {len(suspicious_low)}")
print(suspicious_low[["date", "store_id", "store_type", "is_promo", "revenue", "sw_ratio"]].to_string())
