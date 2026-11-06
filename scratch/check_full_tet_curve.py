import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
stores["store_type"] = stores["store_type"].str.strip().str.lower()
train = train.merge(stores[["store_id", "store_type"]], on="store_id")

# Clean 1000x and S17, S29
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0
train.loc[(train["store_id"] == "S17") & (train["date"] == "2024-02-10"), "revenue"] = 8.97
train.loc[(train["store_id"] == "S29") & (train["date"] == "2024-09-02"), "revenue"] = 35.90

train["dow"] = train["date"].dt.dayofweek

tet_map = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29"), 2026: pd.Timestamp("2026-02-17")}
train["days_from_tet_exact"] = train.apply(
    lambda r: (r["date"] - tet_map[r["date"].year]).days if r["date"].year in tet_map else 999, axis=1
)

# Normal base on 100% data
tr_normal = train[(train["days_from_tet_exact"] < -14) | (train["days_from_tet_exact"] > 14)]
tr_normal = tr_normal[(tr_normal["is_promo"] == 0)]
tr_base = tr_normal.groupby(["store_id", "dow"])["revenue"].median()

train["normal_rev"] = train.set_index(["store_id", "dow"]).index.map(tr_base).fillna(train["revenue"].median())
train["ratio_to_normal"] = train["revenue"] / train["normal_rev"]

tet_mask = train["days_from_tet_exact"].between(-18, 18)
tet_curve_full = train[tet_mask].groupby(["store_type", "days_from_tet_exact"])["ratio_to_normal"].median()

print("=== Tet Multipliers on 100% Clean Data (2024 + 2025) ===")
piv = tet_curve_full.unstack()
print(piv.loc[:, -10:10].round(3))
