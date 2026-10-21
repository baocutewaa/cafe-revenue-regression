"""Calendar and cyclical time feature engineering.

Adheres strictly to Regression_Plan.md Section 6.1:
- Discrete calendar indicators: year, quarter, month, day, day_of_week, day_of_year, week_of_year.
- Business flags: is_weekend, is_month_start, is_month_end, is_month_mid.
- Sin/Cos cyclical harmonic encodings for smooth periodic modeling in linear regressions:
  * Weekday periodicity: sin_dow, cos_dow (period 7 days)
  * Annual month periodicity: sin_month, cos_month (period 12 months)
  * Day of year periodicity: sin_doy, cos_doy (period 365.25 days)
"""

import numpy as np
import pandas as pd
from src.config import DATE_COL


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate calendar and cyclical trigonometric features from date column."""
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[DATE_COL]):
        out[DATE_COL] = pd.to_datetime(out[DATE_COL])

    dt = out[DATE_COL].dt

    # Calendar parts
    out["year"] = dt.year
    out["quarter"] = dt.quarter
    out["month"] = dt.month
    out["day"] = dt.day
    out["day_of_week"] = dt.dayofweek  # 0 = Monday, 6 = Sunday
    out["day_of_year"] = dt.dayofyear
    out["week_of_year"] = dt.isocalendar().week.astype(int)

    # Business and calendar flags
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_month_start"] = dt.is_month_start.astype(int)
    out["is_month_end"] = dt.is_month_end.astype(int)
    out["is_month_mid"] = out["day"].between(10, 20).astype(int)

    # Sin / Cos cyclical encodings (continuous representations for linear regression)
    # 1. Day of week cycle (period = 7)
    out["sin_dow"] = np.sin(2.0 * np.pi * out["day_of_week"] / 7.0)
    out["cos_dow"] = np.cos(2.0 * np.pi * out["day_of_week"] / 7.0)

    # 2. Month of year cycle (period = 12)
    out["sin_month"] = np.sin(2.0 * np.pi * (out["month"] - 1.0) / 12.0)
    out["cos_month"] = np.cos(2.0 * np.pi * (out["month"] - 1.0) / 12.0)

    # 3. Day of year cycle (period = 365.25)
    out["sin_doy"] = np.sin(2.0 * np.pi * (out["day_of_year"] - 1.0) / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * (out["day_of_year"] - 1.0) / 365.25)

    return out
