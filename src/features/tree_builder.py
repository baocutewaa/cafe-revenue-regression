"""Feature engineering and target-derived statistics for Tree-based models.

Adheres strictly to Regression_Plan.md Section 6.4, Section 8, and Section 14:
- Target-derived statistics computed strictly from the training history fold to avoid lookahead leakage.
- Features tailored for tree-based algorithms (Random Forest, Extra Trees, HistGradientBoosting, LightGBM).
- S12 March 2024 unit scale audit and correction (thousand VND -> million VND).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import DATE_COL, STORE_COL, TARGET_COL, PROMO_COL
from src.features.builder import build_all_features

TREE_CATEGORICAL_COLS = ["store_id", "store_type", "city", "district", "day_of_week"]

TREE_NUMERIC_COLS = [
    "area_m2",
    "seats",
    "seats_per_m2",
    "store_age_days",
    "store_age_months",
    "month",
    "day",
    "day_of_year",
    "temp_c",
    "rain_mm",
    "sin_month",
    "cos_month",
    "sin_doy",
    "cos_doy",
    "log_hist_sw_med",
    "log_hist_s_med",
    "log_hist_stw_med",
]

TREE_BINARY_COLS = [
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
]

ALL_TREE_FEATURES = TREE_CATEGORICAL_COLS + TREE_NUMERIC_COLS + TREE_BINARY_COLS


def correct_revenue_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Correct scale error where revenue was recorded in thousand VND instead of million VND.

    Audited per Regression_Plan Section 4.1 & 14:
    - Exactly 31 records of store S12 in March 2024 were entered in thousands (e.g., 53,600 vs 53.6).
    - Dividing values > 1,000 by 1,000 restores them to the correct scale.
    """
    out = df.copy()
    if TARGET_COL in out.columns:
        huge_mask = out[TARGET_COL] > 1000.0
        n_corrected = int(huge_mask.sum())
        if n_corrected > 0:
            out.loc[huge_mask, TARGET_COL] = out.loc[huge_mask, TARGET_COL] / 1000.0
        return out, n_corrected
    return out, 0


def compute_historical_target_stats(
    train_df: pd.DataFrame,
) -> Dict[str, dict]:
    """Compute historical target statistics strictly on the training fold.

    Returns lookup dictionaries for:
    - (store_id, day_of_week) median
    - store_id median
    - (store_type, day_of_week) median
    - global median
    """
    sw_med = train_df.groupby([STORE_COL, "day_of_week"])[TARGET_COL].median().to_dict()
    s_med = train_df.groupby(STORE_COL)[TARGET_COL].median().to_dict()
    stw_med = train_df.groupby(["store_type", "day_of_week"])[TARGET_COL].median().to_dict()
    g_med = float(train_df[TARGET_COL].median())

    return {
        "sw_med": sw_med,
        "s_med": s_med,
        "stw_med": stw_med,
        "global_med": g_med,
    }


def apply_historical_target_stats(
    df: pd.DataFrame,
    stats: Dict[str, dict],
) -> pd.DataFrame:
    """Apply pre-computed training target statistics to any dataset (train, val, or test).

    Ensures zero lookahead leakage by using pre-fitted lookups with hierarchical fallback.
    """
    out = df.copy()
    sw_med = stats["sw_med"]
    s_med = stats["s_med"]
    stw_med = stats["stw_med"]
    g_med = stats["global_med"]

    sw_vals = []
    s_vals = []
    stw_vals = []

    for _, row in out.iterrows():
        s = row[STORE_COL]
        dow = row["day_of_week"]
        st = row.get("store_type", "unknown")

        sw_vals.append(sw_med.get((s, dow), s_med.get(s, g_med)))
        s_vals.append(s_med.get(s, g_med))
        stw_vals.append(stw_med.get((st, dow), g_med))

    out["hist_sw_med"] = sw_vals
    out["hist_s_med"] = s_vals
    out["hist_stw_med"] = stw_vals

    out["log_hist_sw_med"] = np.log1p(out["hist_sw_med"])
    out["log_hist_s_med"] = np.log1p(out["hist_s_med"])
    out["log_hist_stw_med"] = np.log1p(out["hist_stw_med"])

    return out


def prepare_tree_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """Compute and attach tree features across train, val, and test splits with strict leakage prevention."""
    stats = compute_historical_target_stats(train_df)

    X_train = apply_historical_target_stats(train_df, stats)
    X_val = apply_historical_target_stats(val_df, stats)
    X_test = apply_historical_target_stats(test_df, stats) if test_df is not None else None

    return X_train, X_val, X_test, ALL_TREE_FEATURES
