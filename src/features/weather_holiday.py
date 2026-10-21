"""Promotion, weather, and holiday feature engineering.

Adheres strictly to Regression_Plan.md Section 6.3:
- Promotion interactions: promo_x_weekend (is_promo * is_weekend).
- Rain & Temperature indicators: is_rainy, rain_missing, temp_missing.
- Holiday signals: is_holiday, holiday_count, is_national_holiday, is_event.
"""

import numpy as np
import pandas as pd
from src.config import PROMO_COL


def add_promo_weather_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate promotion, weather, and holiday interaction features."""
    out = df.copy()

    # Promotion interaction with weekend
    if PROMO_COL in out.columns:
        out[PROMO_COL] = out[PROMO_COL].fillna(0).astype(int)
        if "is_weekend" in out.columns:
            out["promo_x_weekend"] = (out[PROMO_COL] * out["is_weekend"]).astype(int)
        else:
            out["promo_x_weekend"] = 0
    else:
        out[PROMO_COL] = 0
        out["promo_x_weekend"] = 0

    # Ensure holiday columns exist and are integer
    for col in ["is_holiday", "holiday_count", "is_national_holiday", "is_event"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
        else:
            out[col] = 0

    # Ensure weather missing flags and binary rain indicator exist
    if "rain_mm" in out.columns:
        out["is_rainy"] = (out["rain_mm"].fillna(0) > 0).astype(int)
        out["rain_missing"] = out["rain_mm"].isna().astype(int)
    else:
        out["is_rainy"] = 0
        out["rain_missing"] = 1

    if "temp_c" in out.columns:
        out["temp_missing"] = out["temp_c"].isna().astype(int)
    else:
        out["temp_missing"] = 1

    return out
