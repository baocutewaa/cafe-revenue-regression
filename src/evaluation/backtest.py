"""Walk-forward temporal backtesting and subgroup error analysis."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import DATE_COL, STORE_COL, TARGET_COL, PROMO_COL, WALK_FORWARD_FOLDS
from src.evaluation.metrics import compute_metrics, log_residual, rmsle, mae
from src.models.baseline import WeekdayMedianBaseline


def evaluate_walk_forward(
    df_clean: pd.DataFrame,
    df_raw: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Run walk-forward evaluation across temporal folds defined in Regression_Plan.md."""
    results = []

    # 1. Walk-forward folds on clean deduplicated data
    for fold_key, fold_cfg in WALK_FORWARD_FOLDS.items():
        train_end = fold_cfg["train_end"]
        val_start = fold_cfg["val_start"]
        val_end = fold_cfg["val_end"]

        fit_df = df_clean[df_clean[DATE_COL] < train_end]
        val_df = df_clean[
            (df_clean[DATE_COL] >= val_start) & (df_clean[DATE_COL] < val_end)
        ]

        model = WeekdayMedianBaseline().fit(fit_df)
        preds = model.predict(val_df)
        metrics = compute_metrics(val_df[TARGET_COL].values, preds)

        results.append({
            "fold_key": fold_key,
            "experiment": "E0_baseline",
            "fold_name": fold_cfg["name"],
            "data_version": "clean_dedup",
            "train_period": f"Start to {train_end}",
            "val_period": f"{val_start} to {val_end}",
            "train_rows": len(fit_df),
            "val_rows": len(val_df),
            "rmsle": metrics["rmsle"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mean_log_residual": metrics["mean_log_residual"],
            "std_log_residual": metrics["std_log_residual"],
            "purpose": fold_cfg["purpose"],
        })

    # 2. Historical notebook reproduction on raw data (exact 0.2387 benchmark)
    if df_raw is not None:
        fit_raw = df_raw[df_raw[DATE_COL] < "2025-10-01"]
        val_raw = df_raw[df_raw[DATE_COL] >= "2025-10-01"]
        model_raw = WeekdayMedianBaseline().fit(fit_raw)
        preds_raw = model_raw.predict(val_raw)
        metrics_raw = compute_metrics(val_raw[TARGET_COL].values, preds_raw)

        results.append({
            "fold_key": "notebook_raw_benchmark",
            "experiment": "E0_starter_raw",
            "fold_name": "Notebook Benchmark (Raw duplicates)",
            "data_version": "raw_with_132_dups",
            "train_period": "Start to 2025-10-01",
            "val_period": "2025-10-01 to 2025-12-31",
            "train_rows": len(fit_raw),
            "val_rows": len(val_raw),
            "rmsle": metrics_raw["rmsle"],
            "mae": metrics_raw["mae"],
            "rmse": metrics_raw["rmse"],
            "mean_log_residual": metrics_raw["mean_log_residual"],
            "std_log_residual": metrics_raw["std_log_residual"],
            "purpose": "Đối chiếu chính xác kết quả 0.2387 trong starter_cafe.ipynb",
        })

    return pd.DataFrame(results)


def evaluate_subgroups(
    val_df: pd.DataFrame,
    predictions: np.ndarray,
    stores_df: Optional[pd.DataFrame] = None,
    holidays_df: Optional[pd.DataFrame] = None,
    train_history: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Perform subgroup error analysis specified in Regression_Plan Section 7.3.

    Subgroups evaluated:
    - By store_id
    - By city and store_type (metadata joined)
    - By day of week and month
    - By promo status (is_promo)
    - By holiday status (is_holiday)
    - By store history length (short history < 100 observations vs long history)
    - By revenue tier: Low (< 15M), Medium (15-25M), High (> 25M)
    """
    df = val_df.copy()
    df["pred_revenue"] = np.clip(predictions, 0, None)
    df["log_residual"] = log_residual(df[TARGET_COL].values, df["pred_revenue"].values)
    df["dow"] = df[DATE_COL].dt.day_name()
    df["dow_num"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.strftime("%Y-%m")

    # Join store metadata if available
    if stores_df is not None:
        stores_clean = stores_df.copy()
        stores_clean["store_type_norm"] = stores_clean["store_type"].astype(str).str.strip().str.lower()
        stores_clean["city_norm"] = stores_clean["city"].astype(str).str.strip()
        df = df.merge(
            stores_clean[[STORE_COL, "city_norm", "store_type_norm"]],
            on=STORE_COL,
            how="left"
        )

    # Join holidays if available
    if holidays_df is not None:
        holiday_dates = set(pd.to_datetime(holidays_df[DATE_COL]).dt.date)
        df["is_holiday"] = df[DATE_COL].dt.date.isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    # Store history length indicator
    if train_history is not None:
        store_counts = train_history[STORE_COL].value_counts().to_dict()
        df["train_history_days"] = df[STORE_COL].map(store_counts).fillna(0)
        df["history_tier"] = np.where(
            df["train_history_days"] < 100,
            "Short History (<100 obs)",
            "Established (>=100 obs)"
        )
    else:
        df["history_tier"] = "Unknown"

    # Revenue tier
    df["revenue_tier"] = pd.cut(
        df[TARGET_COL],
        bins=[-np.inf, 15.0, 25.0, np.inf],
        labels=["Low (<15M)", "Medium (15-25M)", "High (>25M)"]
    )

    def group_summary(group_col: str, sort_by_col: Optional[str] = None) -> pd.DataFrame:
        records = []
        for name, grp in df.groupby(group_col, observed=False):
            r = rmsle(grp[TARGET_COL].values, grp["pred_revenue"].values)
            m = mae(grp[TARGET_COL].values, grp["pred_revenue"].values)
            records.append({
                group_col: name,
                "n_samples": len(grp),
                "actual_median": round(float(grp[TARGET_COL].median()), 2),
                "pred_median": round(float(grp["pred_revenue"].median()), 2),
                "rmsle": round(r, 4),
                "mae": round(m, 4),
                "mean_log_residual": round(float(grp["log_residual"].mean()), 4),
            })
        res = pd.DataFrame(records)
        if sort_by_col and sort_by_col in res.columns:
            res = res.sort_values(sort_by_col)
        else:
            res = res.sort_values("rmsle", ascending=False)
        return res

    subgroups = {
        "by_store": group_summary(STORE_COL).sort_values("rmsle", ascending=False),
        "by_dow": group_summary("dow").sort_values("rmsle", ascending=False),
        "by_month": group_summary("month").sort_values("month"),
        "by_promo": group_summary(PROMO_COL),
        "by_holiday": group_summary("is_holiday"),
        "by_revenue_tier": group_summary("revenue_tier"),
    }

    if "city_norm" in df.columns:
        subgroups["by_city"] = group_summary("city_norm")
    if "store_type_norm" in df.columns:
        subgroups["by_store_type"] = group_summary("store_type_norm")
    if "history_tier" in df.columns:
        subgroups["by_history_tier"] = group_summary("history_tier")

    return subgroups
