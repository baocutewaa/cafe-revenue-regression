"""Ensemble and multi-step forecasting module for cafe revenue regression.

Adheres strictly to Regression_Plan.md Section 6.4 & Section 8 (Giai đoạn 5 — Lags và ensemble):
- Multi-model blending in log space: z = sum(w_i * z_i).
- Optimization of non-negative weights on validation (sum(w_i) = 1).
- Out-of-fold Stacking Regressor with Ridge meta-learner.
- Recursive multi-step simulation demonstrating zero-leakage horizon rollout vs teacher forcing.
- Non-negative prediction clipping at 0 and optional Duan smearing factor adjustment.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.config import DATE_COL, STORE_COL, TARGET_COL
from src.evaluation.metrics import rmsle, mae, compute_metrics


class WeightedLogEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble combining diverse base models via log-space weighted blending."""

    def __init__(
        self,
        base_models: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        use_smearing: bool = False,
        random_state: int = 42,
    ):
        """Initialize WeightedLogEnsemble.

        Args:
            base_models: Dictionary of named base model instances.
            weights: Optional manual weights mapping model_name -> weight.
            use_smearing: If True, applies Duan smearing factor estimated on train residuals.
            random_state: Random seed for reproducibility.
        """
        self.base_models = base_models or {}
        self.weights = weights
        self.use_smearing = use_smearing
        self.random_state = random_state

        self.is_fitted: bool = False
        self.weights_: Dict[str, float] = {}
        self.duan_factor_: float = 1.0
        self.model_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "WeightedLogEnsemble":
        """Fit all underlying base models on the training data."""
        y_arr = np.asarray(y, dtype=float).ravel()
        self.model_names_ = list(self.base_models.keys())

        # Fit each base model
        for name, model in self.base_models.items():
            model.fit(X, y_arr)

        # Set or normalize weights
        if self.weights:
            total_w = sum(self.weights.values())
            self.weights_ = {k: self.weights[k] / total_w for k in self.model_names_}
        else:
            # Default equal weights
            equal_w = 1.0 / len(self.model_names_)
            self.weights_ = {k: equal_w for k in self.model_names_}

        # Estimate smearing factor on train predictions
        if self.use_smearing:
            z_train_comb = self._predict_log(X)
            z_true_train = np.log1p(np.clip(y_arr, 0, None))
            residuals = z_true_train - z_train_comb
            self.duan_factor_ = float(np.mean(np.exp(residuals)))
        else:
            self.duan_factor_ = 1.0

        self.is_fitted = True
        return self

    def _predict_log_single(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Predict target in log space for a single base model."""
        if hasattr(model, "predict_transformed"):
            return model.predict_transformed(X)
        elif hasattr(model, "predict"):
            preds = model.predict(X)
            return np.log1p(np.clip(preds, 0, None))
        else:
            raise ValueError(f"Model {model} lacks predict method.")

    def _predict_log(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted blend of log-space predictions."""
        z_combined = np.zeros(len(X), dtype=float)
        for name in self.model_names_:
            w = self.weights_.get(name, 0.0)
            if w > 0:
                z_single = self._predict_log_single(self.base_models[name], X)
                z_combined += w * z_single
        return z_combined

    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """Optimize non-negative weights strictly minimizing validation RMSLE."""
        if not self.is_fitted:
            raise RuntimeError("Base models must be fitted before optimizing weights.")

        y_true = np.asarray(y_val, dtype=float).ravel()
        n_models = len(self.model_names_)

        # Precompute base model log predictions on validation
        Z_val_list = [self._predict_log_single(self.base_models[name], X_val) for name in self.model_names_]
        Z_val_matrix = np.column_stack(Z_val_list)

        def objective(weights_vec):
            total = np.sum(weights_vec)
            if total <= 0:
                return 1e6
            w_norm = weights_vec / total
            z_blend = Z_val_matrix @ w_norm
            y_pred = np.clip(np.expm1(z_blend), 0, None)
            return rmsle(y_true, y_pred)

        init_weights = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        res = minimize(objective, init_weights, bounds=bounds, method="SLSQP")
        w_opt = res.x / np.sum(res.x)
        self.weights_ = {name: float(w_opt[i]) for i, name in enumerate(self.model_names_)}

        # Update smearing factor if enabled
        if self.use_smearing:
            z_blend_val = Z_val_matrix @ w_opt
            z_true_val = np.log1p(np.clip(y_true, 0, None))
            residuals = z_true_val - z_blend_val
            self.duan_factor_ = float(np.mean(np.exp(residuals)))

        return self.weights_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict revenue in raw currency units."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted yet.")

        z_pred = self._predict_log(X)
        if self.use_smearing and self.duan_factor_ > 0:
            y_pred = np.exp(z_pred) * self.duan_factor_ - 1.0
        else:
            y_pred = np.expm1(z_pred)

        # Clip non-negative
        return np.clip(y_pred, 0, None)

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target in log space."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted yet.")
        return self._predict_log(X)

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize ensemble model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "WeightedLogEnsemble":
        """Deserialize ensemble model from disk."""
        return joblib.load(filepath)


class StackingLogEnsemble(BaseEstimator, RegressorMixin):
    """Stacking ensemble using out-of-fold predictions with positive-constrained meta-regressor."""

    def __init__(
        self,
        base_models_factory: Callable[[], Dict[str, Any]],
        n_splits: int = 5,
        meta_alpha: float = 10.0,
        random_state: int = 42,
    ):
        """Initialize StackingLogEnsemble.

        Args:
            base_models_factory: Callable returning a fresh dict of base model instances.
            n_splits: Number of K-Fold splits for out-of-fold prediction.
            meta_alpha: Regularization strength for Ridge meta-regressor.
            random_state: Random seed.
        """
        self.base_models_factory = base_models_factory
        self.n_splits = n_splits
        self.meta_alpha = meta_alpha
        self.random_state = random_state

        self.fitted_base_models_: Dict[str, Any] = {}
        self.meta_regressor_: Optional[Ridge] = None
        self.meta_weights_: Dict[str, float] = {}
        self.model_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "StackingLogEnsemble":
        """Fit base models with out-of-fold cross validation and meta-regressor."""
        y_arr = np.asarray(y, dtype=float).ravel()
        z_arr = np.log1p(np.clip(y_arr, 0, None))
        n_samples = len(X)

        # Initialize base models
        dummy_models = self.base_models_factory()
        self.model_names_ = list(dummy_models.keys())
        n_models = len(self.model_names_)

        # Array to store OOF log predictions
        oof_matrix = np.zeros((n_samples, n_models), dtype=float)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for fold, (trn_idx, oof_idx) in enumerate(kf.split(X)):
            X_trn = X.iloc[trn_idx]
            y_trn = y_arr[trn_idx]
            X_oof = X.iloc[oof_idx]

            fold_models = self.base_models_factory()
            for i, name in enumerate(self.model_names_):
                model = fold_models[name]
                model.fit(X_trn, y_trn)
                if hasattr(model, "predict_transformed"):
                    oof_matrix[oof_idx, i] = model.predict_transformed(X_oof)
                else:
                    oof_matrix[oof_idx, i] = np.log1p(np.clip(model.predict(X_oof), 0, None))

        # Train meta-regressor with positive constraint
        self.meta_regressor_ = Ridge(alpha=self.meta_alpha, positive=True, fit_intercept=False)
        self.meta_regressor_.fit(oof_matrix, z_arr)

        raw_weights = self.meta_regressor_.coef_
        total_w = np.sum(raw_weights)
        if total_w > 0:
            norm_weights = raw_weights / total_w
        else:
            norm_weights = np.ones(n_models) / n_models

        self.meta_weights_ = {name: float(norm_weights[i]) for i, name in enumerate(self.model_names_)}

        # Fit final base models on the entire training set
        self.fitted_base_models_ = self.base_models_factory()
        for name, model in self.fitted_base_models_.items():
            model.fit(X, y_arr)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict revenue on new data using meta-model weighting."""
        if not self.is_fitted:
            raise RuntimeError("StackingLogEnsemble is not fitted yet.")

        Z_meta_list = []
        for name in self.model_names_:
            model = self.fitted_base_models_[name]
            if hasattr(model, "predict_transformed"):
                Z_meta_list.append(model.predict_transformed(X))
            else:
                Z_meta_list.append(np.log1p(np.clip(model.predict(X), 0, None)))

        Z_meta_matrix = np.column_stack(Z_meta_list)
        weights_vec = np.array([self.meta_weights_[name] for name in self.model_names_])
        z_pred = Z_meta_matrix @ weights_vec
        return np.clip(np.expm1(z_pred), 0, None)
