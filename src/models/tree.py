"""Tree-based regression modeling pipeline with optional log target transformation.

Adheres strictly to Regression_Plan.md Section 8 (Giai đoạn 3 — Tree-based models) & Section 8.4:
- Tree-based architectures: LightGBM, HistGradientBoosting, Random Forest, Extra Trees.
- Uses OrdinalEncoder for categorical variables to preserve tree split efficiency.
- Supports training on z = log1p(revenue) and inverse expm1(z) with non-negative clipping at 0.
- Extracts feature importances across all tree families.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor

from src.features.tree_builder import (
    ALL_TREE_FEATURES,
    TREE_CATEGORICAL_COLS,
    TREE_NUMERIC_COLS,
    TREE_BINARY_COLS,
)


class LogTransformedTreeModel:
    """Tree-based regression pipeline supporting LightGBM, HistGBR, RF, and ExtraTrees."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        use_log_target: bool = True,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None,
        random_state: int = 42,
        **model_params: Any,
    ):
        """Initialize tree model pipeline.

        Args:
            model_type: 'lightgbm', 'hist_gradient_boosting', 'random_forest', 'extra_trees'.
            use_log_target: If True, fits on log1p(y) and inverts via expm1(z).
            categorical_cols: Categorical columns to ordinal encode.
            numeric_cols: Numeric columns.
            binary_cols: Binary indicator columns.
            random_state: Random seed for reproducibility.
            **model_params: Hyperparameters passed to the underlying regressor.
        """
        self.model_type = model_type.lower()
        self.use_log_target = use_log_target
        self.random_state = random_state
        self.model_params = model_params

        self.cat_cols = categorical_cols or TREE_CATEGORICAL_COLS
        self.num_cols = numeric_cols or TREE_NUMERIC_COLS
        self.bin_cols = binary_cols or TREE_BINARY_COLS
        self.all_feature_cols = self.cat_cols + self.num_cols + self.bin_cols

        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.importances_df_: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def _create_regressor(self):
        """Instantiate underlying tree regressor with parameters."""
        params = self.model_params.copy()
        params.setdefault("random_state", self.random_state)

        if self.model_type == "lightgbm":
            params.setdefault("n_estimators", 300)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("num_leaves", 20)
            params.setdefault("min_child_samples", 20)
            params.setdefault("subsample", 0.8)
            params.setdefault("colsample_bytree", 0.8)
            params.setdefault("reg_alpha", 0.1)
            params.setdefault("reg_lambda", 1.0)
            params.setdefault("verbose", -1)
            params.pop("max_iter", None)
            return LGBMRegressor(**params)
        elif self.model_type == "hist_gradient_boosting":
            params.pop("n_estimators", None)
            params.setdefault("max_iter", 200)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("max_depth", 6)
            params.setdefault("min_samples_leaf", 20)
            return HistGradientBoostingRegressor(**params)
        elif self.model_type == "random_forest":
            params.pop("max_iter", None)
            params.setdefault("n_estimators", 150)
            params.setdefault("max_depth", 14)
            params.setdefault("min_samples_leaf", 5)
            params.setdefault("n_jobs", -1)
            return RandomForestRegressor(**params)
        elif self.model_type == "extra_trees":
            params.pop("max_iter", None)
            params.setdefault("n_estimators", 150)
            params.setdefault("max_depth", 16)
            params.setdefault("min_samples_leaf", 5)
            params.setdefault("n_jobs", -1)
            return ExtraTreesRegressor(**params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _create_preprocessor(self) -> ColumnTransformer:
        """Build column transformer with ordinal encoding for categorical features."""
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        bin_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ])

        transformers = [
            ("cat", cat_pipeline, self.cat_cols),
            ("num", num_pipeline, self.num_cols),
            ("bin", bin_pipeline, self.bin_cols),
        ]
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> "LogTransformedTreeModel":
        """Fit tree model pipeline."""
        y_arr = np.asarray(y, dtype=float)
        if np.any(y_arr < 0):
            raise ValueError("Target contains negative values; revenue must be non-negative.")

        # Transform target if requested
        y_fit = np.log1p(y_arr) if self.use_log_target else y_arr

        preprocessor = self._create_preprocessor()
        regressor = self._create_regressor()

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        self.pipeline.fit(X, y_fit)
        self.is_fitted = True

        # Extract feature names
        self.feature_names_ = self.cat_cols + self.num_cols + self.bin_cols

        # Extract feature importances
        reg = self.pipeline.named_steps["regressor"]
        importances = None
        if hasattr(reg, "feature_importances_"):
            importances = reg.feature_importances_
        elif hasattr(reg, "coef_"):
            importances = np.abs(reg.coef_)

        if importances is not None:
            imp_arr = np.asarray(importances, dtype=float)
            total = imp_arr.sum()
            norm_imp = imp_arr / total if total > 0 else imp_arr
            self.importances_df_ = pd.DataFrame({
                "feature": self.feature_names_[:len(norm_imp)],
                "importance": norm_imp,
                "raw_importance": imp_arr,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            self.importances_df_ = pd.DataFrame({
                "feature": self.feature_names_,
                "importance": [1.0 / len(self.feature_names_)] * len(self.feature_names_),
            })

        self.metadata = {
            "model_type": self.model_type,
            "use_log_target": self.use_log_target,
            "n_features_in": len(self.feature_names_),
            "n_samples": len(X),
            "model_params": self.model_params,
        }

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions on original revenue scale (Million VND)."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        raw_pred = self.pipeline.predict(X)
        if hasattr(raw_pred, "ravel"):
            raw_pred = raw_pred.ravel()

        if self.use_log_target:
            y_pred = np.expm1(raw_pred)
        else:
            y_pred = raw_pred

        # Strictly enforce non-negativity constraint
        y_pred_clipped = np.clip(y_pred, 0.0, None)
        return y_pred_clipped

    def get_feature_importances(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Retrieve sorted feature importance dataframe."""
        if not self.is_fitted or self.importances_df_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if top_n is not None:
            return self.importances_df_.head(top_n).copy()
        return self.importances_df_.copy()

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist fitted model pipeline and metadata using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "LogTransformedTreeModel":
        """Load fitted model pipeline from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        instance = joblib.load(path)
        return instance
