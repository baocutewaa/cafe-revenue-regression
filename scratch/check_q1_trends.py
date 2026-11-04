import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv", parse_dates=["date"])
# Clean S12 1000x
train.loc[(train["store_id"] == "S12") & (train["revenue"] > 1000), "revenue"] /= 1000.0

# Filter Q1 2024 (2024-01-01 to 2024-03-31) and Q1 2025 (2025-01-01 to 2025-03-31)
q1_24 = train[(train["date"] >= "2024-01-01") & (train["date"] <= "2024-03-31")]
q1_25 = train[(train["date"] >= "2025-01-01") & (train["date"] <= "2025-03-31")]

print(f"Q1 2024: {len(q1_24)} rows, mean: {q1_24['revenue'].mean():.2f}, median: {q1_24['revenue'].median():.2f}")
print(f"Q1 2025: {len(q1_25)} rows, mean: {q1_25['revenue'].mean():.2f}, median: {q1_25['revenue'].median():.2f}")
print(f"Growth Q1 2024 -> Q1 2025: {(q1_25['revenue'].mean() - q1_24['revenue'].mean()) / q1_24['revenue'].mean() * 100:.2f}%")

# Now let's check our test predictions for Q1 2026:
sub = pd.read_csv("submission.csv")
print(f"Q1 2026 (Our Preds): {len(sub)} rows, mean: {sub['revenue'].mean():.2f}, median: {sub['revenue'].median():.2f}")
print(f"Ratio Q1 2026 Preds / Q1 2025 Actual: {sub['revenue'].mean() / q1_25['revenue'].mean():.4f}")
print(f"Ratio Q1 2026 Preds / Q1 2024 Actual: {sub['revenue'].mean() / q1_24['revenue'].mean():.4f}")
