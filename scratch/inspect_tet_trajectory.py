import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
stores["store_type"] = stores["store_type"].str.strip().str.lower()
train = train.merge(stores[["store_id", "store_type"]], on="store_id")

# Clean 1000x and S17 89.7
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0
train.loc[(train["store_id"] == "S17") & (train["date"] == "2024-02-10"), "revenue"] /= 10.0
train.loc[(train["store_id"] == "S29") & (train["date"] == "2024-09-02"), "revenue"] *= 10.0

train["dow"] = train["date"].dt.dayofweek
sw_med = train.groupby(["store_id", "dow"])["revenue"].transform("median")
train["sw_ratio"] = train["revenue"] / sw_med

tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29")}
train["year"] = train["date"].dt.year
train["days_from_tet"] = train.apply(lambda r: (r["date"] - tet_map[r["year"]]).days if r["year"] in tet_map else 999, axis=1)

tet_df = train[train["days_from_tet"].between(-15, 15)]

print("=== Mean Ratio (revenue / normal_dow_median) by days_from_tet for each store_type (both years combined) ===")
pivot = tet_df.groupby(["days_from_tet", "store_type"])["sw_ratio"].mean().unstack()
print(pivot.round(3))
