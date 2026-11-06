import pandas as pd
import numpy as np

test = pd.read_csv("data/test.csv", parse_dates=["date"])
train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")

print("Test date range:", test["date"].min(), "to", test["date"].max())
print("Test shape:", test.shape)
print("Unique stores in test:", test["store_id"].nunique())
print("Stores in stores.csv but NOT in test:", set(stores["store_id"]) - set(test["store_id"]))

# Tet 2026: Feb 17, 2026.
# Check dates around Tet 2026 in test:
tet_2026 = pd.Timestamp("2026-02-17")
test["days_from_tet"] = (test["date"] - tet_2026).dt.days

# Count test rows per day around Tet
test_around_tet = test[test["days_from_tet"].between(-7, 7)]
print("\nTest rows per day around Tet 2026:")
print(test_around_tet.groupby(["days_from_tet", "date"])["store_id"].count())

# Let's check which stores are closed on which days around Tet in test:
closed_stores = test_around_tet.merge(stores, on="store_id")
print("\nStore types present on Tet days in test:")
print(pd.crosstab(closed_stores["days_from_tet"], closed_stores["store_type"]))
