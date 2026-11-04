import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
train["date"] = pd.to_datetime(train["date"])
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month

# Clean 1000x for S12
mask_s12 = (train["store_id"] == "S12") & (train["revenue"] > 1000)
train.loc[mask_s12, "revenue"] /= 1000.0

# Clean 10x and 0.1x
# Let's check average revenue in 2024 vs 2025
y2024 = train[train["year"] == 2024]["revenue"].mean()
y2025 = train[train["year"] == 2025]["revenue"].mean()
print(f"2024 Mean Revenue: {y2024:.2f} tr.d")
print(f"2025 Mean Revenue: {y2025:.2f} tr.d")
print(f"YoY Growth 2024 -> 2025: {(y2025 - y2024) / y2024 * 100:.2f}%")

# Let's check by month: 2024 vs 2025 for mature stores
stores = pd.read_csv("data/stores.csv")
mature_stores = stores[pd.to_datetime(stores["open_date"], format="mixed", dayfirst=True) < "2024-01-01"]["store_id"].unique()
train_mature = train[train["store_id"].isin(mature_stores)]

for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    r24 = train_mature[(train_mature["year"] == 2024) & (train_mature["month"] == m)]["revenue"].mean()
    r25 = train_mature[(train_mature["year"] == 2025) & (train_mature["month"] == m)]["revenue"].mean()
    print(f"Month {m:2d}: 2024 = {r24:5.2f} | 2025 = {r25:5.2f} | Growth = {(r25 - r24)/r24*100:+5.2f}%")
