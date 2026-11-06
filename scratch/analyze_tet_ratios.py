import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
# Clean stores store_type
stores["store_type"] = stores["store_type"].str.strip().str.lower()
train = train.merge(stores[["store_id", "store_type", "city"]], on="store_id")

# Clean 1000x
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0

# Clean 10x and 0.1x
train["dow"] = train["date"].dt.dayofweek
sw_med = train.groupby(["store_id", "dow"])["revenue"].transform("median")
sw_ratio = train["revenue"] / sw_med
train.loc[sw_ratio > 3.8, "revenue"] /= 10.0
sw_med = train.groupby(["store_id", "dow"])["revenue"].transform("median")
sw_ratio = train["revenue"] / sw_med
train.loc[sw_ratio < 0.16, "revenue"] *= 10.0

# Map Tet dates
tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29")}
train["year"] = train["date"].dt.year
train["days_from_tet"] = train.apply(lambda r: (r["date"] - tet_map[r["year"]]).days if r["year"] in tet_map else 999, axis=1)

# Check sw_ratio by days_from_tet and store_type for days in [-15, 15]
tet_data = train[train["days_from_tet"].between(-15, 15)].copy()
tet_data["sw_ratio"] = tet_data["revenue"] / tet_data.groupby(["store_id", "dow"])["revenue"].transform("median")

print("\n--- Average Ratio (revenue / normal_dow_median) around Tet by store_type (2024 vs 2025) ---")
for y in [2024, 2025]:
    y_tet = tet_data[tet_data["year"] == y]
    pivot = y_tet.groupby(["days_from_tet", "store_type"])["sw_ratio"].mean().unstack()
    print(f"\nYear {y}:")
    print(pivot.loc[-10:10].round(2))
