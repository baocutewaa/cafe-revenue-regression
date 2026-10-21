"""Target transformation and loss formulation module for revenue regression.

Adheres strictly to Regression_Plan.md Section 8 (Giai đoạn 4 — Biến đổi target):
- Mathematical transforms: raw (identity), log1p, sqrt, asinh, yeo_johnson.
- Scikit-learn compatible transformer with exact inverse transformations.
- Non-negative clipping: max(0, inverse(z)).
- Retransformation bias corrections: None, half_variance (exp(z + 0.5*s^2)-1), Duan's smearing.
- TransformedTargetRegressorPipeline wrapping LightGBM, Ridge, and other regressors.
- Zero-leakage parameter estimation strictly on training data.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler
from lightgbm import LGBMRegressor

from src.features.tree_builder import (
    ALL_TREE_FEATURES,
    TREE_CATEGORICAL_COLS,
    TREE_NUMERIC_COLS,
    TREE_BINARY_COLS,
)

TargetTransformType = Literal["raw", "log1p", "sqrt", "asinh", "yeo_johnson"]
BiasCorrectionType = Literal["none", "half_variance", "duan_smearing"]


class TargetTransformer(BaseEstimator, TransformerMixin):
    """Transformer for target variable y with exact inverse and bias adjustment."""

    def __init__(
        self,
        transform_type: TargetTransformType = "log1p",
        bias_correction: BiasCorrectionType = "none",
    ):
        """Initialize TargetTransformer.

        Args:
            transform_type: One of 'raw', 'log1p', 'sqrt', 'asinh', 'yeo_johnson'.
            bias_correction: One of 'none', 'half_variance', 'duan_smearing'.
        """
        self.transform_type = transform_type.lower()
        self.bias_correction = bias_correction.lower()
        self.is_fitted: bool = False

        # Parameters learned on training data
        self.yeo_johnson_transformer_: Optional[PowerTransformer] = None
        self.lambda_: Optional[float] = None
        self.residual_variance_: float = 0.0
        self.duan_factor_: float = 1.0

    def fit(self, y: np.ndarray, y_pred_train: Optional[np.ndarray] = None) -> "TargetTransformer":
        """Fit transformer parameters strictly on training target values.

        Args:
            y: 1D array of target values (revenue >= 0).
            y_pred_train: Optional predictions in transformed space for bias correction estimation.
        """
        y_arr = np.asarray(y, dtype=float).ravel()
        if np.any(y_arr < 0):
            # Clip negative inputs to 0 for safety
            y_arr = np.clip(y_arr, 0, None)

        if self.transform_type == "yeo_johnson":
            self.yeo_johnson_transformer_ = PowerTransformer(method="yeo-johnson", standardize=False)
            self.yeo_johnson_transformer_.fit(y_arr.reshape(-1, 1))
            self.lambda_ = float(self.yeo_johnson_transformer_.lambdas_[0])
        elif self.transform_type == "log1p":
            self.lambda_ = 0.0
        elif self.transform_type == "sqrt":
            self.lambda_ = 0.5
        elif self.transform_type == "raw":
            self.lambda_ = 1.0
        elif self.transform_type == "asinh":
            self.lambda_ = None
        else:
            raise ValueError(f"Unsupported transform_type: {self.transform_type}")

        # Compute bias correction parameters if training predictions are provided
        if y_pred_train is not None and self.transform_type == "log1p":
            z_true = np.log1p(y_arr)
            z_pred = np.asarray(y_pred_train, dtype=float).ravel()
            residuals = z_true - z_pred
            self.residual_variance_ = float(np.var(residuals))
            self.duan_factor_ = float(np.mean(np.exp(residuals)))

        self.is_fitted = True
        return self

    def update_bias_correction(self, y_true: np.ndarray, z_pred_train: np.ndarray) -> None:
        """Update residual variance and Duan smearing factor after model fit on train fold."""
        if self.transform_type != "log1p":
            return
        y_arr = np.asarray(y_true, dtype=float).ravel()
        z_true = np.log1p(y_arr)
        z_pred = np.asarray(z_pred_train, dtype=float).ravel()
        residuals = z_true - z_pred
        self.residual_variance_ = float(np.var(residuals))
        self.duan_factor_ = float(np.mean(np.exp(residuals)))

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Forward transform y -> z."""
        y_arr = np.asarray(y, dtype=float).ravel()
        y_arr = np.clip(y_arr, 0, None)

        if self.transform_type == "raw":
            return y_arr
        elif self.transform_type == "log1p":
            return np.log1p(y_arr)
        elif self.transform_type == "sqrt":
            return np.sqrt(y_arr)
        elif self.transform_type == "asinh":
            return np.arcsinh(y_arr)
        elif self.transform_type == "yeo_johnson":
            if self.yeo_johnson_transformer_ is None:
                raise RuntimeError("TargetTransformer must be fitted before transform.")
            return self.yeo_johnson_transformer_.transform(y_arr.reshape(-1, 1)).ravel()
        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

    def inverse_transform(
        self,
        z: np.ndarray,
        override_bias_correction: Optional[BiasCorrectionType] = None,
    ) -> np.ndarray:
        """Inverse transform z -> y_hat with non-negative guarantee and optional bias correction."""
        z_arr = np.asarray(z, dtype=float).ravel()
        correction = (override_bias_correction or self.bias_correction).lower()

        if self.transform_type == "raw":
            y_pred = z_arr
        elif self.transform_type == "log1p":
            if correction == "half_variance" and self.residual_variance_ > 0:
                y_pred = np.exp(z_arr + 0.5 * self.residual_variance_) - 1.0
            elif correction == "duan_smearing" and self.duan_factor_ > 0:
                y_pred = np.exp(z_arr) * self.duan_factor_ - 1.0
            else:
                y_pred = np.expm1(z_arr)
        elif self.transform_type == "sqrt":
            y_pred = np.clip(z_arr, 0, None) ** 2
        elif self.transform_type == "asinh":
            y_pred = np.sinh(z_arr)
        elif self.transform_type == "yeo_johnson":
            if self.yeo_johnson_transformer_ is None:
                raise RuntimeError("TargetTransformer must be fitted before inverse_transform.")
            y_pred = self.yeo_johnson_transformer_.inverse_transform(z_arr.reshape(-1, 1)).ravel()
        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        # Section 8.4: Clip any negative predictions to 0
        return np.clip(y_pred, 0, None)


class TransformedTargetTreeModel(BaseEstimator, RegressorMixin):
    """End-to-end Tree Model with explicit target transformation and objective formulation."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        transform_type: TargetTransformType = "log1p",
        bias_correction: BiasCorrectionType = "none",
        objective: Optional[str] = None,
        tweedie_variance_power: float = 1.5,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None,
        random_state: int = 42,
        **model_params: Any,
    ):
        """Initialize transformed target tree pipeline.

        Args:
            model_type: 'lightgbm', 'hist_gradient_boosting', 'random_forest'.
            transform_type: 'raw', 'log1p', 'sqrt', 'asinh', 'yeo_johnson'.
            bias_correction: 'none', 'half_variance', 'duan_smearing'.
            objective: Regressor objective (e.g. 'regression', 'regression_l1', 'huber', 'poisson', 'tweedie').
            tweedie_variance_power: Power parameter if objective == 'tweedie'.
            categorical_cols: List of categorical feature names.
            numeric_cols: List of numeric feature names.
            binary_cols: List of binary feature names.
            random_state: Random seed.
            **model_params: Regressor parameters.
        """
        self.model_type = model_type.lower()
        self.transform_type = transform_type.lower()
        self.bias_correction = bias_correction.lower()
        self.objective = objective
        self.tweedie_variance_power = tweedie_variance_power
        self.random_state = random_state
        self.model_params = model_params

        self.cat_cols = categorical_cols or TREE_CATEGORICAL_COLS
        self.num_cols = numeric_cols or TREE_NUMERIC_COLS
        self.bin_cols = binary_cols or TREE_BINARY_COLS
        self.all_feature_cols = self.cat_cols + self.num_cols + self.bin_cols

        self.target_transformer = TargetTransformer(
            transform_type=self.transform_type,
            bias_correction=self.bias_correction,
        )
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def _create_regressor(self):
        """Instantiate underlying tree regressor with parameters and specified objective."""
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

            if self.objective:
                params["objective"] = self.objective
                if self.objective == "tweedie":
                    params["tweedie_variance_power"] = self.tweedie_variance_power

            return LGBMRegressor(**params)
        else:
            raise ValueError(f"TransformedTargetTreeModel currently specializes in LightGBM, received {self.model_type}")

    def _create_preprocessor(
        self,
        cat_cols: List[str],
        num_cols: List[str],
        bin_cols: List[str],
    ) -> ColumnTransformer:
        """Create feature preprocessing pipeline for available columns."""
        transformers = []
        if cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
            transformers.append(("cat", cat_pipe, cat_cols))

        if num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ])
            transformers.append(("num", num_pipe, num_cols))

        if bin_cols:
            bin_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ])
            transformers.append(("bin", bin_pipe, bin_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "TransformedTargetTreeModel":
        """Fit model with target transform."""
        y_raw = np.asarray(y, dtype=float).ravel()
        self.target_transformer.fit(y_raw)
        z_target = self.target_transformer.transform(y_raw)

        avail_cat = [c for c in self.cat_cols if c in X.columns]
        avail_num = [c for c in self.num_cols if c in X.columns]
        avail_bin = [c for c in self.bin_cols if c in X.columns]
        available_cols = avail_cat + avail_num + avail_bin
        self.feature_names_ = available_cols

        preprocessor = self._create_preprocessor(avail_cat, avail_num, avail_bin)
        regressor = self._create_regressor()

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        X_subset = X[available_cols]
        self.pipeline.fit(X_subset, z_target)

        # Update bias correction parameters on training predictions
        if self.transform_type == "log1p":
            z_pred_train = self.pipeline.predict(X_subset)
            self.target_transformer.update_bias_correction(y_raw, z_pred_train)

        self.is_fitted = True
        self.metadata = {
            "model_type": self.model_type,
            "transform_type": self.transform_type,
            "bias_correction": self.bias_correction,
            "objective": self.objective or "regression (MSE)",
            "lambda": self.target_transformer.lambda_,
            "residual_variance": self.target_transformer.residual_variance_,
            "duan_factor": self.target_transformer.duan_factor_,
            "n_features": len(available_cols),
        }
        return self

    def predict(
        self,
        X: pd.DataFrame,
        override_bias_correction: Optional[BiasCorrectionType] = None,
    ) -> np.ndarray:
        """Predict revenue in raw currency units."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")

        X_subset = X[self.feature_names_]
        z_pred = self.pipeline.predict(X_subset)
        return self.target_transformer.inverse_transform(z_pred, override_bias_correction)

    def predict_transformed(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target directly in transformed space z."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        X_subset = X[self.feature_names_]
        return self.pipeline.predict(X_subset)

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize pipeline to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TransformedTargetTreeModel":
        """Deserialize pipeline from disk."""
        return joblib.load(filepath)
