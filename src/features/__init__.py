"""Feature engineering modules for cafe revenue regression."""

from src.features.calendar import add_calendar_features
from src.features.store import add_store_features
from src.features.weather_holiday import add_promo_weather_holiday_features
from src.features.builder import (
    FEATURE_CONFIGS,
    build_all_features,
    get_feature_columns,
)

from src.features.tree_builder import (
    ALL_TREE_FEATURES,
    TREE_CATEGORICAL_COLS,
    TREE_NUMERIC_COLS,
    TREE_BINARY_COLS,
    correct_revenue_outliers,
    compute_historical_target_stats,
    apply_historical_target_stats,
    prepare_tree_feature_matrices,
)

__all__ = [
    "add_calendar_features",
    "add_store_features",
    "add_promo_weather_holiday_features",
    "FEATURE_CONFIGS",
    "build_all_features",
    "get_feature_columns",
    "ALL_TREE_FEATURES",
    "TREE_CATEGORICAL_COLS",
    "TREE_NUMERIC_COLS",
    "TREE_BINARY_COLS",
    "correct_revenue_outliers",
    "compute_historical_target_stats",
    "apply_historical_target_stats",
    "prepare_tree_feature_matrices",
]
