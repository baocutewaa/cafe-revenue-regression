"""Metrics implementation for cafe revenue regression.

Metric definitions adhere strictly to Regression_Plan.md:
- Primary metric: Root Mean Squared Logarithmic Error (RMSLE)
- Diagnostic metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
- Log residual: r_i = log(1 + y_true) - log(1 + y_pred_clipped)
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def rmsle(y_true: Union[np.ndarray, pd.Series, list], y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """Compute Root Mean Squared Logarithmic Error (RMSLE).

    Formula:
        RMSLE = sqrt( mean( (log(1 + clip(y_pred, 0, None)) - log(1 + y_true))^2 ) )

    Args:
        y_true: Ground truth target values (non-negative).
        y_pred: Predicted target values. Automatically clipped at 0.

    Returns:
        RMSLE score as a float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if np.any(y_true < 0):
        raise ValueError("y_true contains negative values; revenue must be non-negative.")

    y_pred_clipped = np.clip(y_pred, 0, None)
    log_diff = np.log1p(y_pred_clipped) - np.log1p(y_true)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def mae(y_true: Union[np.ndarray, pd.Series, list], y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """Compute Mean Absolute Error on original scale (Million VND)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: Union[np.ndarray, pd.Series, list], y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """Compute Root Mean Squared Error on original scale (Million VND)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def log_residual(y_true: Union[np.ndarray, pd.Series, list], y_pred: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Compute log residuals defined in Regression_Plan.md section 9.2:
        r_i = log(1 + y_i) - log(1 + hat_y_i)

    Interpretation:
        r > 0: Model underpredicts actual revenue (y_pred < y_true)
        r < 0: Model overpredicts actual revenue (y_pred > y_true)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return np.log1p(y_true) - np.log1p(y_pred)


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list]
) -> Dict[str, float]:
    """Compute all evaluation metrics and return a formatted dictionary."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    r_score = rmsle(y_true_arr, y_pred_arr)
    m_score = mae(y_true_arr, y_pred_arr)
    rmse_score = rmse(y_true_arr, y_pred_arr)
    residuals = log_residual(y_true_arr, y_pred_arr)

    return {
        "rmsle": round(r_score, 4),
        "mae": round(m_score, 4),
        "rmse": round(rmse_score, 4),
        "mean_log_residual": round(float(np.mean(residuals)), 4),
        "std_log_residual": round(float(np.std(residuals)), 4),
        "sample_count": len(y_true_arr),
    }
