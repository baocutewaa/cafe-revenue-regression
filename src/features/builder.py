"""Feature builder and column definition registry for experiments.

Organizes features into hierarchical tiers matching the experimental matrix
in Regression_Plan.md Section 8:
- E1: Calendar + Store metadata (investigates store differentiation and periodic seasonality)
- E2: E1 + Promotion flags and weekend interaction (investigates promotional revenue lift)
- E3: E2 + Weather & Holiday environmental signals
"""

from typing import Dict, List, Tuple
import pandas as pd

from src.features.calendar import add_calendar_features
from src.features.store import add_store_features
from src.features.weather_holiday import add_promo_weather_holiday_features

# Feature registry for experimental tiers
FEATURE_CONFIGS: Dict[str, Dict[str, List[str]]] = {
    "E1_calendar_store": {
        "categorical": ["store_dow", "store_type", "city", "district"],
        "numeric": [
            "area_m2",
            "seats",
            "store_age_days",
            "seats_per_m2",
            "sin_month",
            "cos_month",
            "sin_doy",
            "cos_doy",
        ],
        "binary": [
            "has_parking",
            "area_m2_missing",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_month_mid",
            "is_new_store",
        ],
    },
    "E2_promo": {
        "categorical": ["store_dow", "store_type", "city", "district"],
        "numeric": [
            "area_m2",
            "seats",
            "store_age_days",
            "seats_per_m2",
            "sin_month",
            "cos_month",
            "sin_doy",
            "cos_doy",
        ],
        "binary": [
            "has_parking",
            "area_m2_missing",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_month_mid",
            "is_new_store",
            "is_promo",
            "promo_x_weekend",
        ],
    },
    "E3_weather_holiday": {
        "categorical": ["store_dow", "store_type", "city", "district"],
        "numeric": [
            "area_m2",
            "seats",
            "store_age_days",
            "seats_per_m2",
            "sin_month",
            "cos_month",
            "sin_doy",
            "cos_doy",
            "temp_c",
            "rain_mm",
        ],
        "binary": [
            "has_parking",
            "area_m2_missing",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_month_mid",
            "is_new_store",
            "is_promo",
            "promo_x_weekend",
            "is_holiday",
            "holiday_count",
            "is_national_holiday",
            "is_event",
            "is_rainy",
            "rain_missing",
            "temp_missing",
        ],
    },
}


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature extractors sequentially on a joined DataFrame."""
    out = add_calendar_features(df)
    out = add_store_features(out)
    out = add_promo_weather_holiday_features(out)
    return out


def get_feature_columns(tier: str = "E2_promo") -> Tuple[List[str], List[str], List[str]]:
    """Retrieve lists of (categorical_cols, numeric_cols, binary_cols) for a given tier."""
    if tier not in FEATURE_CONFIGS:
        raise ValueError(
            f"Unknown tier: {tier}. Available tiers: {list(FEATURE_CONFIGS.keys())}"
        )
    cfg = FEATURE_CONFIGS[tier]
    return cfg["categorical"], cfg["numeric"], cfg["binary"]
