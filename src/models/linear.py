"""Linear regression modeling pipeline with log1p target transformation.

Adheres strictly to Regression_Plan.md Section 8 (Giai đoạn 2 — Mô hình tuyến tính):
- Trains on z = log1p(revenue) to directly optimize MSE on the log scale (concordant with RMSLE).
- Inverts predictions via expm1(z) and clips strictly non-negative at 0.
- Implements ColumnTransformer with:
  * Median imputation and StandardScaler for numeric features.
  * Constant imputation and OneHotEncoder(handle_unknown='ignore') for categorical features.
  * Passthrough / constant imputation for binary indicator features.
- Supports Ridge, RidgeCV, ElasticNet, Lasso, HuberRegressor, and OLS.
- Provides coefficient extraction with business percent impacts (exp(beta) - 1).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.builder import get_feature_columns


class LogTransformedLinearModel:
    """End-to-end linear regression pipeline with log1p target transformation and robust feature processing."""

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        epsilon: float = 1.35,
        alphas: Optional[List[float]] = None,
        feature_tier: str = "E2_promo",
        random_state: int = 42,
    ):
        """Initialize linear regression model.

        Args:
            model_type: 'ridge', 'ridge_cv', 'elastic_net', 'lasso', 'huber', 'ols'.
            alpha: Regularization strength for Ridge/Lasso/ElasticNet/Huber.
            l1_ratio: Mixing parameter for ElasticNet (0 <= l1_ratio <= 1).
            epsilon: Robustness parameter for HuberRegressor.
            alphas: Search grid for RidgeCV/LassoCV.
            feature_tier: Key from FEATURE_CONFIGS ('E1_calendar_store', 'E2_promo', 'E3_weather_holiday').
            random_state: Reproducibility seed.
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        self.feature_tier = feature_tier
        self.random_state = random_state

        self.cat_cols, self.num_cols, self.bin_cols = get_feature_columns(feature_tier)
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.coefficients_df_: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def _create_regressor(self):
        """Instantiate underlying scikit-learn regressor."""
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.model_type == "ridge_cv":
            return RidgeCV(alphas=self.alphas)
        elif self.model_type == "elastic_net":
            return ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
                max_iter=500,
                tol=1e-3,
            )
        elif self.model_type == "lasso":
            return Lasso(
                alpha=self.alpha,
                random_state=self.random_state,
                max_iter=500,
                tol=1e-3,
            )
        elif self.model_type == "huber":
            return HuberRegressor(
                alpha=self.alpha,
                epsilon=self.epsilon,
                max_iter=150,
            )
        elif self.model_type == "ols":
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _create_preprocessor(self) -> ColumnTransformer:
        """Build column transformer strictly fit on training data."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        bin_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ])

        transformers = [
            ("num", num_pipeline, self.num_cols),
            ("cat", cat_pipeline, self.cat_cols),
            ("bin", bin_pipeline, self.bin_cols),
        ]
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> "LogTransformedLinearModel":
        """Fit linear regression pipeline on log1p-transformed target."""
        y_arr = np.asarray(y, dtype=float)
        if np.any(y_arr < 0):
            raise ValueError("Target contains negative values; revenue must be non-negative.")

        # Transform target to log scale: z = log(1 + y)
        z = np.log1p(y_arr)

        preprocessor = self._create_preprocessor()
        regressor = self._create_regressor()

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        self.pipeline.fit(X, z)
        self.is_fitted = True

        # Extract post-transformation feature names
        try:
            self.feature_names_ = list(
                self.pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
        except Exception:
            self.feature_names_ = [f"feat_{i}" for i in range(regressor.coef_.shape[0])]

        # Extract coefficients
        reg = self.pipeline.named_steps["regressor"]
        coefs = reg.coef_
        if hasattr(coefs, "ravel"):
            coefs = coefs.ravel()

        coef_df = pd.DataFrame({
            "feature": self.feature_names_,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            # In log-linear models, exp(beta) - 1 represents the relative percentage effect on original scale
            "percent_impact": np.expm1(coefs) * 100.0,
        }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

        self.coefficients_df_ = coef_df

        best_alpha = getattr(reg, "alpha_", self.alpha)
        self.metadata = {
            "model_type": self.model_type,
            "feature_tier": self.feature_tier,
            "selected_alpha": float(best_alpha) if best_alpha is not None else None,
            "intercept": float(getattr(reg, "intercept_", 0.0)),
            "n_features_in": len(self.feature_names_),
            "n_samples": len(X),
        }

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict revenue on original scale (Million VND).

        Transforms predictions from z = log1p(y) back via expm1(z)
        and clips negative values to 0.0 as required by competition rules.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Predict in log1p space
        z_pred = self.pipeline.predict(X)
        if hasattr(z_pred, "ravel"):
            z_pred = z_pred.ravel()

        # Invert to original revenue scale
        y_pred = np.expm1(z_pred)

        # Enforce non-negativity constraint
        y_pred_clipped = np.clip(y_pred, 0.0, None)
        return y_pred_clipped

    def get_coefficients(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Retrieve sorted feature coefficients and impact percentages."""
        if not self.is_fitted or self.coefficients_df_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if top_n is not None:
            return self.coefficients_df_.head(top_n).copy()
        return self.coefficients_df_.copy()

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist fitted model pipeline and metadata using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "LogTransformedLinearModel":
        """Load fitted model pipeline from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        instance = joblib.load(path)
        return instance
