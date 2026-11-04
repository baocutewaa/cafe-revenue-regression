import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
test = pd.read_csv("data/test.csv", parse_dates=["date"])

def weekday_median(history, target):
    h = history.assign(dow=history.date.dt.dayofweek)
    m = h.groupby(["store_id", "dow"]).revenue.median()
    keys = list(zip(target.store_id, target.date.dt.dayofweek))
    p = pd.Series(keys, index=target.index).map(m)
    p = p.fillna(history.revenue.median())
    return p

starter_preds = weekday_median(train, test)
print("Starter predictions describe:")
print(starter_preds.describe())

our_sub = pd.read_csv("submission.csv")
print("\nOur submission describe:")
print(our_sub["revenue"].describe())

# Compare difference between starter and our submission
diff = our_sub["revenue"] - starter_preds
print("\nDifference (Our - Starter) describe:")
print(diff.describe())
print(f"RMSLE between Our Sub and Starter Sub: {np.sqrt(np.mean((np.log1p(our_sub['revenue']) - np.log1p(starter_preds))**2)):.4f}")
