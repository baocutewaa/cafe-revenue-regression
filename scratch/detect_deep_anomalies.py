import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
stores["store_type"] = stores["store_type"].str.strip().str.lower()
train = train.merge(stores[["store_id", "store_type"]], on="store_id")

# Clean 1000x for S12
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0

# Compute store-dayofweek medians excluding Tet and opening days
tet_dates = [pd.Timestamp("2024-02-10"), pd.Timestamp("2025-01-29")]
train["days_to_tet_24"] = (train["date"] - tet_dates[0]).dt.days
train["days_to_tet_25"] = (train["date"] - tet_dates[1]).dt.days
train["is_near_tet"] = train["days_to_tet_24"].between(-15, 15) | train["days_to_tet_25"].between(-15, 15)

# Normal mask
stores_meta = pd.read_csv("data/stores.csv")
stores_meta["open_date_dt"] = pd.to_datetime(stores_meta["open_date"], format="mixed", dayfirst=True)
train = train.merge(stores_meta[["store_id", "open_date_dt"]], on="store_id")
train["days_since_open"] = (train["date"] - train["open_date_dt"]).dt.days
train["is_opening"] = train["days_since_open"] <= 7

train["dow"] = train["date"].dt.dayofweek
normal_rows = train[(~train["is_near_tet"]) & (~train["is_opening"]) & (train["is_promo"] == 0)]
base_med = normal_rows.groupby(["store_id", "dow"])["revenue"].median()

train["normal_base"] = train.set_index(["store_id", "dow"]).index.map(base_med)
train["ratio_to_base"] = train["revenue"] / train["normal_base"]

# Check all rows on Tet with anomalous ratio compared to store_type
print("=== Checking Tet Anomalies in 2024 and 2025 ===")
for y, t_date in [(2024, tet_dates[0]), (2025, tet_dates[1])]:
    tet_sub = train[train["date"].between(t_date - pd.Timedelta(days=5), t_date + pd.Timedelta(days=5))].copy()
    tet_sub["days_from_tet"] = (tet_sub["date"] - t_date).dt.days
    # Group by store_type and days_from_tet to get expected ratio
    type_tet_med = tet_sub.groupby(["days_from_tet", "store_type"])["ratio_to_base"].transform("median")
    tet_sub["dev"] = tet_sub["ratio_to_base"] / type_tet_med
    
    outliers = tet_sub[(tet_sub["dev"] > 2.5) | (tet_sub["dev"] < 0.4)]
    if len(outliers) > 0:
        print(f"\nYear {y} Tet Outliers:")
        print(outliers[["date", "store_id", "store_type", "days_from_tet", "revenue", "normal_base", "ratio_to_base", "dev"]].to_string())
