"""Weekday-Store Median Baseline model with hierarchical fallback.

Adheres strictly to the starter_cafe.ipynb baseline logic and Regression_Plan.md:
1. Primary estimate: Median revenue of (store_id, weekday).
2. Fallback 1: Store-level median revenue.
3. Fallback 2: Global median revenue across the entire history.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import DATE_COL, STORE_COL, TARGET_COL


class WeekdayMedianBaseline:
    """Predicts daily cafe revenue using median by (store_id, weekday) with hierarchical fallbacks."""

    def __init__(self):
        self.store_weekday_medians: Dict[Tuple[str, int], float] = {}
        self.store_medians: Dict[str, float] = {}
        self.global_median: Optional[float] = None
        self.is_fitted: bool = False
        self.metadata: Dict[str, Any] = {}

    def fit(self, history: pd.DataFrame) -> "WeekdayMedianBaseline":
        """Fit baseline on historical observations.

        Args:
            history: DataFrame containing date, store_id, and revenue columns.
        """
        if not {DATE_COL, STORE_COL, TARGET_COL}.issubset(history.columns):
            missing = {DATE_COL, STORE_COL, TARGET_COL} - set(history.columns)
            raise ValueError(f"History DataFrame missing required columns: {missing}")

        h = history.copy()
        if not pd.api.types.is_datetime64_any_dtype(h[DATE_COL]):
            h[DATE_COL] = pd.to_datetime(h[DATE_COL])

        # Compute weekday index (0 = Monday, 6 = Sunday)
        h["dow"] = h[DATE_COL].dt.dayofweek

        # Level 1: (store_id, dow) median
        sw_series = h.groupby([STORE_COL, "dow"])[TARGET_COL].median()
        self.store_weekday_medians = {
            (str(k[0]), int(k[1])): float(v) for k, v in sw_series.items()
        }

        # Level 2: store median
        s_series = h.groupby(STORE_COL)[TARGET_COL].median()
        self.store_medians = {str(k): float(v) for k, v in s_series.items()}

        # Level 3: global median
        self.global_median = float(h[TARGET_COL].median())

        self.is_fitted = True
        self.metadata = {
            "n_observations": len(h),
            "n_stores": int(h[STORE_COL].nunique()),
            "n_store_weekday_pairs": len(self.store_weekday_medians),
            "min_date": str(h[DATE_COL].min().date()),
            "max_date": str(h[DATE_COL].max().date()),
            "global_median": self.global_median,
        }
        return self

    def predict(
        self,
        target: pd.DataFrame,
        return_fallback_info: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.Series]]:
        """Generate predictions for target observations.

        Args:
            target: DataFrame with date and store_id columns.
            return_fallback_info: If True, returns a tuple of (predictions, fallback_level_series).

        Returns:
            np.ndarray of clipped non-negative revenue predictions, or (preds, fallback_info).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if not {DATE_COL, STORE_COL}.issubset(target.columns):
            missing = {DATE_COL, STORE_COL} - set(target.columns)
            raise ValueError(f"Target DataFrame missing required columns: {missing}")

        t = target.copy()
        if not pd.api.types.is_datetime64_any_dtype(t[DATE_COL]):
            t[DATE_COL] = pd.to_datetime(t[DATE_COL])

        dows = t[DATE_COL].dt.dayofweek.values
        stores = t[STORE_COL].astype(str).values

        predictions = np.empty(len(t), dtype=float)
        fallback_levels = pd.Series(index=t.index, dtype="object")

        for idx, (s, dow) in enumerate(zip(stores, dows)):
            key = (s, int(dow))
            if key in self.store_weekday_medians:
                predictions[idx] = self.store_weekday_medians[key]
                fallback_levels.iloc[idx] = "store_weekday"
            elif s in self.store_medians:
                predictions[idx] = self.store_medians[s]
                fallback_levels.iloc[idx] = "store_fallback"
            else:
                predictions[idx] = self.global_median
                fallback_levels.iloc[idx] = "global_fallback"

        # Clip negative predictions to 0 as per competition rules
        predictions = np.clip(predictions, 0.0, None)

        if return_fallback_info:
            return predictions, fallback_levels
        return predictions

    def save(self, file_path: Union[str, Path]) -> None:
        """Serialize model state to JSON for transparent, portable reproducibility."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": self.metadata,
            "global_median": self.global_median,
            "store_medians": self.store_medians,
            "store_weekday_medians": {
                f"{k[0]}__{k[1]}": v for k, v in self.store_weekday_medians.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "WeekdayMedianBaseline":
        """Deserialize model state from JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        instance = cls()
        instance.metadata = payload.get("metadata", {})
        instance.global_median = payload["global_median"]
        instance.store_medians = payload["store_medians"]
        instance.store_weekday_medians = {
            (k.split("__")[0], int(k.split("__")[1])): v
            for k, v in payload["store_weekday_medians"].items()
        }
        instance.is_fitted = True
        return instance
