import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
# Clean 1000x S12
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0

train["time_index"] = (train["date"] - pd.Timestamp("2024-01-01")).dt.days

# Let's fit a simple log-linear trend for the whole chain: ln(revenue) = alpha + beta * time_index
train_clean = train[(train["revenue"] > 0) & (train["revenue"] < 100)].copy()
train_clean["log_rev"] = np.log1p(train_clean["revenue"])

# Overall regression on time_index
slope, intercept = np.polyfit(train_clean["time_index"], train_clean["log_rev"], 1)
daily_growth = slope
annual_growth = np.expm1(slope * 365) * 100

print(f"Overall daily log slope: {slope:.6f}")
print(f"Annualized revenue growth: {annual_growth:.2f}% per year")

# Check by store
store_slopes = []
for sid, group in train_clean.groupby("store_id"):
    if len(group) > 100:
        s_slope, s_intercept = np.polyfit(group["time_index"], group["log_rev"], 1)
        store_slopes.append({"store_id": sid, "slope": s_slope, "annual_growth": np.expm1(s_slope * 365) * 100})
df_slopes = pd.DataFrame(store_slopes)
print("\nStore-level annual growth summary:")
print(df_slopes["annual_growth"].describe())
