import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
stores["store_type"] = stores["store_type"].str.strip().str.lower()
train = train.merge(stores[["store_id", "store_type"]], on="store_id")

# Clean 1000x
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0

# Tet dates
tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29")}
train["year"] = train["date"].dt.year
train["days_from_tet"] = train.apply(lambda r: (r["date"] - tet_map[r["year"]]).days if r["year"] in tet_map else 999, axis=1)

# Inspect street stores on days -2, -1, 0, 1, 2 in 2024 vs 2025
st = train[(train["store_type"] == "street") & (train["days_from_tet"].between(-2, 2))]
print("Street stores on days_from_tet in [-2, 2] in 2024:")
print(st[st["year"] == 2024][["date", "store_id", "days_from_tet", "revenue"]].head(25))

print("\nCheck if S17 on Mung 1 Tet 2024 had 89.7 (the 10x error mentioned in Regression_Plan.md):")
print(train[(train["store_id"] == "S17") & (train["date"] == "2024-02-10")][["date", "store_id", "revenue"]])
