import pandas as pd
import numpy as np

new_sub = pd.read_csv("submission.csv")
test = pd.read_csv("data/test.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
stores["store_type"] = stores["store_type"].str.strip().str.lower()
test_sub = test.merge(new_sub, on="id").merge(stores, on="store_id")

print("New submission describe:")
print(new_sub["revenue"].describe().round(2))

# Tet dates in 2026: Feb 17
tet_2026 = pd.Timestamp("2026-02-17")
test_sub["days_from_tet"] = (test_sub["date"] - tet_2026).dt.days

print("\nAverage predicted revenue by period in 2026:")
p1 = test_sub[test_sub["days_from_tet"] < -14]["revenue"].mean()
p2 = test_sub[test_sub["days_from_tet"].between(-12, -3)]["revenue"].mean()
p3 = test_sub[test_sub["days_from_tet"].between(-2, 2)]["revenue"].mean()
p4 = test_sub[test_sub["days_from_tet"].between(3, 9)]["revenue"].mean()
p5 = test_sub[test_sub["days_from_tet"] > 9]["revenue"].mean()

print(f"  Period 1 (Jan normal):              {p1:.2f} tr.d")
print(f"  Period 2 (Pre-Tet shopping rush):   {p2:.2f} tr.d")
print(f"  Period 3 (Tet core 29-Mung 3):      {p3:.2f} tr.d")
print(f"  Period 4 (Post-Tet recovery M4-10): {p4:.2f} tr.d")
print(f"  Period 5 (Post-Tet normal Mar):     {p5:.2f} tr.d")

print("\nTet Core (-2 to 2) Predictions by store_type:")
print(test_sub[test_sub["days_from_tet"].between(-2, 2)].groupby("store_type")["revenue"].agg(["count", "mean", "min", "max"]).round(2))
