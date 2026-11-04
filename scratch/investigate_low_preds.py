import pandas as pd
import numpy as np

sub = pd.read_csv("submission.csv")
test = pd.read_csv("data/test.csv")
test_sub = test.merge(sub, on="id")
stores = pd.read_csv("data/stores.csv")
test_sub = test_sub.merge(stores, on="store_id")

print("Rows with lowest predicted revenue (< 6.0):")
low = test_sub[test_sub["revenue"] < 6.0].sort_values(by="revenue")
print(f"Total rows < 6.0: {len(low)}")
print(low[["id", "date", "store_id", "store_type", "city", "revenue", "is_promo"]].head(20))

print("\nStore-level mean in our submission vs starter:")
train = pd.read_csv("data/train.csv", parse_dates=["date"])
def weekday_median(history, target):
    h = history.assign(dow=pd.to_datetime(target["date"]).dt.dayofweek)
    m = h.groupby(["store_id", h["date"].dt.dayofweek])["revenue"].median()
    keys = list(zip(target["store_id"], pd.to_datetime(target["date"]).dt.dayofweek))
    p = pd.Series(keys, index=target.index).map(m)
    p = p.fillna(history["revenue"].median())
    return p

test_sub["starter_pred"] = weekday_median(train, test_sub)
store_comp = test_sub.groupby(["store_id", "store_type", "city"])[["revenue", "starter_pred"]].mean()
store_comp["diff"] = store_comp["revenue"] - store_comp["starter_pred"]
store_comp["diff_pct"] = store_comp["diff"] / store_comp["starter_pred"] * 100
print("\nTop 10 stores where our prediction is much LOWER than starter:")
print(store_comp.sort_values(by="diff").head(10))

print("\nTop 10 stores where our prediction is much HIGHER than starter:")
print(store_comp.sort_values(by="diff", ascending=False).head(10))
