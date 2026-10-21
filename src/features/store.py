"""Store metadata and store lifecycle feature engineering.

Adheres strictly to Regression_Plan.md Section 6.2:
- store_age_days: difference between observation date and parsed store open_date.
- store_age_months: store age expressed in months.
- is_new_store: binary flag identifying stores opened within the last 180 days.
- seats_per_m2: physical density metric (seats / area_m2).
"""

import numpy as np
import pandas as pd
from src.config import DATE_COL


def add_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate store lifecycle and physical characteristics features."""
    out = df.copy()

    # Store age calculation
    if "open_date_parsed" in out.columns and out["open_date_parsed"].notna().any():
        age_delta = (out[DATE_COL] - out["open_date_parsed"]).dt.days
        out["store_age_days"] = np.maximum(age_delta.fillna(365), 0).astype(float)
        out["store_age_months"] = out["store_age_days"] / 30.4375
        out["is_new_store"] = (out["store_age_days"] < 180).astype(int)
    else:
        out["store_age_days"] = 365.0
        out["store_age_months"] = 12.0
        out["is_new_store"] = 0

    # Physical ratio: seats per m2 (safe division using imputed or minimum area)
    if "area_m2" in out.columns and "seats" in out.columns:
        safe_area = out["area_m2"].fillna(out["area_m2"].median())
        safe_area = np.maximum(safe_area, 10.0)
        out["seats_per_m2"] = out["seats"] / safe_area
    else:
        out["seats_per_m2"] = 0.4

    # Store-Weekday interaction feature (captures store-specific weekly cycles)
    if "store_id" in out.columns and DATE_COL in out.columns:
        d = out[DATE_COL] if pd.api.types.is_datetime64_any_dtype(out[DATE_COL]) else pd.to_datetime(out[DATE_COL])
        out["store_dow"] = out["store_id"].astype(str) + "_dow" + d.dt.dayofweek.astype(str)
    else:
        out["store_dow"] = "unknown"

    return out
