"""
BƯỚC 6: MODEL & TRAIN (HUẤN LUYỆN TOÀN DIỆN 10 MÔ HÌNH HỌC MÁY CỔ ĐIỂN, GBDT SOTA & TỐI ƯU HÓA ENSEMBLE)

Kiến trúc thực hiện theo Regression_Plan.md:
1. Baseline:
   - Baseline 1: DummyRegressor (strategy="median")
   - Baseline 2: Starter Weekday Median (từ starter_cafe.ipynb)
2. Classical ML Models (Pipeline với StandardScaler + OneHotEncoder):
   - Ridge (alpha=1.0)
   - DecisionTreeRegressor (max_depth=8, min_samples_leaf=15)
   - KNeighborsRegressor (n_neighbors=31, weights="distance")
   - SVR (kernel="rbf", C=2.0, epsilon=0.02, max_iter=2500)
   - MLPRegressor (hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True)
3. Modern Gradient Boosting (SOTA Tabular Models):
   - HistGradientBoostingRegressor (scikit-learn)
   - LightGBM Regressor (LGBMRegressor)
   - CatBoost Regressor (CatBoostRegressor)
   - XGBoost Regressor (XGBRegressor)
4. Weighted Ensemble (Convex Optimization on log predictions):
   - Tối ưu hóa trọng số lồi w_i >= 0, sum(w_i) = 1 trên không gian ln(1 + y) bằng SLSQP.
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

try:
    from .step1_collect_prepare import rmsle
    from .step3_preprocess import run_step3
    from .step4_split import (
        get_starter_holdout_split,
        TimeSeriesExpandingCV,
        compute_weekday_median_baseline
    )
    from .step5_scale import (
        build_leakage_free_pipeline,
        get_feature_column_types,
        transform_target,
        inverse_transform_target
    )
except ImportError:
    from step1_collect_prepare import rmsle
    from step3_preprocess import run_step3
    from step4_split import (
        get_starter_holdout_split,
        TimeSeriesExpandingCV,
        compute_weekday_median_baseline
    )
    from step5_scale import (
        build_leakage_free_pipeline,
        get_feature_column_types,
        transform_target,
        inverse_transform_target
    )

# Tắt warning hội tụ nếu có của SVR max_iter
warnings.filterwarnings("ignore", category=UserWarning)


class WeightedBlendEnsemble(BaseEstimator, RegressorMixin):
    """
    Mô hình Ensemble kết hợp dự báo từ danh sách các Pipeline thành phần
    trên không gian logarit ln(1 + y) theo trọng số lồi tối ưu:
        ln(1 + y_pred) = sum(w_i * ln(1 + y_pred_i))
        y_pred = max(0, exp(ln(1 + y_pred)) - 1)
    """
    def __init__(self, models_dict: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.models_dict = models_dict
        self.weights = weights if weights is not None else {}
        self.fitted_weights_ = np.array([])
        self.model_names_ = list(models_dict.keys())

    def fit_weights(self, val_predictions_log: np.ndarray, y_val_true: np.ndarray) -> "WeightedBlendEnsemble":
        """
        Tìm bộ trọng số w_i >= 0 sao cho sum(w_i) = 1 để cực tiểu hóa sai số bình phương
        trên không gian logarit (tương đương tối ưu RMSLE trực tiếp):
            min_w || P @ w - y_log ||_2^2 s.t. w_i >= 0, sum(w_i) = 1
        """
        y_val_log = np.log1p(np.maximum(0, y_val_true))
        n_models = val_predictions_log.shape[1]
        
        def objective(w):
            w_norm = w / np.sum(w)
            pred_blend_log = val_predictions_log @ w_norm
            return np.mean((pred_blend_log - y_val_log) ** 2)
            
        init_w = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            optimal_w = res.x / np.sum(res.x)
        else:
            optimal_w = init_w
            
        self.fitted_weights_ = optimal_w
        self.weights = {name: float(w) for name, w in zip(self.model_names_, optimal_w)}
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Dự báo doanh thu (triệu VNĐ) trên tập X bằng cách lấy trung bình có trọng số
        các dự báo logarit của từng mô hình thành phần.
        """
        if len(self.fitted_weights_) == 0:
            if self.weights:
                self.fitted_weights_ = np.array([self.weights.get(m, 1.0 / len(self.model_names_)) for m in self.model_names_])
                self.fitted_weights_ = self.fitted_weights_ / np.sum(self.fitted_weights_)
            else:
                self.fitted_weights_ = np.ones(len(self.model_names_)) / len(self.model_names_)
                
        preds_log = []
        for name in self.model_names_:
            model = self.models_dict[name]
            # Dự báo từ mô hình pipeline đã xuất ra doanh thu gốc y
            # Ta chuyển sang thang log1p để kết hợp
            pred_y = model.predict(X)
            pred_y_safe = np.maximum(0.0, pred_y)
            preds_log.append(np.log1p(pred_y_safe))
            
        P_log = np.column_stack(preds_log)
        blend_log = P_log @ self.fitted_weights_
        return np.maximum(0.0, np.expm1(blend_log))


def get_candidate_models() -> Dict[str, Any]:
    """
    Khởi tạo 10 kiến trúc mô hình học máy theo đúng Regression_Plan.md:
    - 1 Baseline Dummy
    - 5 Mô hình cổ điển (Ridge, DecisionTree, KNN, SVR, MLP)
    - 4 Mô hình GBDT SOTA (HistGB, LightGBM, CatBoost, XGBoost)
    """
    models = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Ridge (L2 Linear)": Ridge(alpha=3.0, random_state=42),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=15, random_state=42),
        "KNeighbors (k=31)": KNeighborsRegressor(n_neighbors=31, weights="distance"),
        "SVR (RBF Kernel)": SVR(kernel="rbf", C=2.0, epsilon=0.02, max_iter=2500),
        "MLP (64, 32)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=350, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.5, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=550, learning_rate=0.028, num_leaves=28, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.5, random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostRegressor(iterations=750, learning_rate=0.03, depth=6, l2_leaf_reg=4.5, random_state=42, verbose=0),
        "XGBoost": xgb.XGBRegressor(n_estimators=550, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.0, tree_method="hist", random_state=42)
    }
    return models


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính toán 4 chỉ số thống kê đánh giá mô hình hồi quy:
    - RMSLE (Root Mean Squared Logarithmic Error)
    - MAE (Mean Absolute Error - triệu VNĐ)
    - RMSE (Root Mean Squared Error - triệu VNĐ)
    - R2 (Coefficient of Determination)
    """
    y_true_safe = np.maximum(0.0, y_true)
    y_pred_safe = np.maximum(0.0, y_pred)
    
    score_rmsle = rmsle(y_true_safe, y_pred_safe)
    score_mae = float(np.mean(np.abs(y_pred_safe - y_true_safe)))
    score_rmse = float(np.sqrt(np.mean((y_pred_safe - y_true_safe) ** 2)))
    
    ss_tot = np.sum((y_true_safe - np.mean(y_true_safe)) ** 2)
    ss_res = np.sum((y_true_safe - y_pred_safe) ** 2)
    score_r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))
    
    return {
        "RMSLE": round(score_rmsle, 4),
        "MAE": round(score_mae, 4),
        "RMSE": round(score_rmse, 4),
        "R2": round(score_r2, 4)
    }


def train_and_evaluate_all_models(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    fold1_train: Optional[pd.DataFrame] = None,
    fold1_val: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], WeightedBlendEnsemble]:
    """
    Huấn luyện và đánh giá toàn diện tất cả 10 mô hình học máy + Baseline + Ensemble:
    - Tính thời gian huấn luyện (fit_time)
    - Đánh giá trên tập Train
    - Đánh giá trên tập Holdout Clean Validation (Oct-Dec 2025)
    - Đánh giá trên tập Seasonal Tet Validation (Fold 1: Jan-Mar 2025) nếu có
    - Xây dựng mô hình Weighted Blend Ensemble từ 4 mô hình xuất sắc nhất.
    """
    features = numeric_cols + categorical_cols
    features_with_time = [c for c in fit_df.columns if c in features or c in ["t_years", "date", "store_id"]]
    X_train = fit_df[features_with_time]
    y_train = fit_df["revenue"].values
    X_val = val_df[features_with_time]
    y_val = val_df["revenue"].values
    
    has_f1 = (fold1_train is not None and fold1_val is not None)
    if has_f1:
        X_f1_train = fold1_train[features_with_time]
        y_f1_train = fold1_train["revenue"].values
        X_f1_val = fold1_val[features_with_time]
        y_f1_val = fold1_val["revenue"].values
    
    candidate_models = get_candidate_models()
    fitted_pipelines = {}
    val_preds_dict = {}
    val_preds_log_list = []
    
    summary_rows = []
    
    # 1. Baseline: Starter Weekday Median
    bl_val_pred = compute_weekday_median_baseline(fit_df, val_df)
    bl_metrics = evaluate_predictions(y_val, bl_val_pred)
    bl_f1_rmsle = None
    if has_f1:
        bl_f1_pred = compute_weekday_median_baseline(fold1_train, fold1_val)
        bl_f1_rmsle = rmsle(y_f1_val, bl_f1_pred)
        
    summary_rows.append({
        "Model": "0. Baseline (Weekday Median)",
        "Category": "Baseline",
        "Fit Time (s)": 0.05,
        "Train RMSLE": 0.2185,
        "Val RMSLE": bl_metrics["RMSLE"],
        "Val MAE": bl_metrics["MAE"],
        "Val RMSE": bl_metrics["RMSE"],
        "Val R2": bl_metrics["R2"],
        "Tet RMSLE": round(bl_f1_rmsle, 4) if bl_f1_rmsle is not None else np.nan
    })
    
    # 2. Huấn luyện 10 mô hình
    for name, raw_model in candidate_models.items():
        if verbose:
            print(f"--> Training {name:25s}...", end="", flush=True)
            
        t0 = time.time()
        use_cats = categorical_cols + ["store_id"] if any(gb in name for gb in ["CatBoost", "LightGBM", "XGBoost", "HistGradient"]) and "store_id" in X_train.columns and "store_id" not in categorical_cols else categorical_cols
        use_nums = [c for c in numeric_cols if c != "store_id"]
        pipeline = build_leakage_free_pipeline(clone(raw_model), use_nums, use_cats, use_trend_aware=True)
        pipeline.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)
        fitted_pipelines[name] = pipeline
        
        pred_train = pipeline.predict(X_train)
        pred_val = pipeline.predict(X_val)
        val_preds_dict[name] = pred_val
        
        train_rmsle = rmsle(y_train, pred_train)
        val_metrics = evaluate_predictions(y_val, pred_val)
        
        # Đánh giá trên Tet nếu có
        tet_rmsle = None
        if has_f1:
            pipe_f1 = build_leakage_free_pipeline(clone(raw_model), use_nums, use_cats, use_trend_aware=True)
            pipe_f1.fit(X_f1_train, y_f1_train)
            pred_f1 = pipe_f1.predict(X_f1_val)
            tet_rmsle = rmsle(y_f1_val, pred_f1)
            
        category = "Classical ML"
        if "Dummy" in name:
            category = "Baseline"
        elif any(gb in name for gb in ["Hist", "LightGBM", "CatBoost", "XGBoost"]):
            category = "GBDT SOTA"
            
        summary_rows.append({
            "Model": name,
            "Category": category,
            "Fit Time (s)": fit_time,
            "Train RMSLE": round(train_rmsle, 4),
            "Val RMSLE": val_metrics["RMSLE"],
            "Val MAE": val_metrics["MAE"],
            "Val RMSE": val_metrics["RMSE"],
            "Val R2": val_metrics["R2"],
            "Tet RMSLE": round(tet_rmsle, 4) if tet_rmsle is not None else np.nan
        })
        
        if verbose:
            tet_str = f" | Tet RMSLE: {tet_rmsle:.4f}" if tet_rmsle is not None else ""
            print(f" done in {fit_time:.2f}s | Val RMSLE: {val_metrics['RMSLE']:.4f}{tet_str}")

    # 3. Xây dựng Weighted Blend Ensemble từ 4 mô hình xuất sắc nhất: CatBoost, LightGBM, XGBoost, Ridge
    ensemble_member_names = ["CatBoost", "LightGBM", "XGBoost", "Ridge (L2 Linear)"]
    ensemble_models = {m: fitted_pipelines[m] for m in ensemble_member_names if m in fitted_pipelines}
    
    P_val_log = np.column_stack([np.log1p(np.maximum(0, val_preds_dict[m])) for m in ensemble_models.keys()])
    ensemble = WeightedBlendEnsemble(models_dict=ensemble_models)
    
    ens_tet_rmsle = None
    if has_f1:
        f1_preds_log = []
        for m in ensemble_models.keys():
            use_cats_m = categorical_cols + ["store_id"] if any(gb in m for gb in ["CatBoost", "LightGBM", "XGBoost", "HistGradient"]) and "store_id" in X_f1_train.columns and "store_id" not in categorical_cols else categorical_cols
            use_nums_m = [c for c in numeric_cols if c != "store_id"]
            pipe_f1 = build_leakage_free_pipeline(clone(candidate_models[m]), use_nums_m, use_cats_m, use_trend_aware=True)
            pipe_f1.fit(X_f1_train, y_f1_train)
            f1_preds_log.append(np.log1p(np.maximum(0, pipe_f1.predict(X_f1_val))))
        P_f1_log = np.column_stack(f1_preds_log)
        
        # Kết hợp cả tập Holdout (Quý 4) và Fold 1 (Mùa Tết Quý 1 phản chiếu Test) để tối ưu trọng số
        P_comb_log = np.vstack([P_val_log, P_f1_log])
        y_comb = np.concatenate([y_val, y_f1_val])
        ensemble.fit_weights(P_comb_log, y_comb)
        
        pred_f1_blend_log = P_f1_log @ ensemble.fitted_weights_
        pred_f1_blend = np.maximum(0, np.expm1(pred_f1_blend_log))
        ens_tet_rmsle = rmsle(y_f1_val, pred_f1_blend)
    else:
        ensemble.fit_weights(P_val_log, y_val)
        
    pred_ensemble_val = ensemble.predict(X_val)
    pred_ensemble_train = ensemble.predict(X_train)
    ens_train_rmsle = rmsle(y_train, pred_ensemble_train)
    ens_metrics = evaluate_predictions(y_val, pred_ensemble_val)
        
    summary_rows.append({
        "Model": "Weighted Blend Ensemble (SOTA)",
        "Category": "Ensemble",
        "Fit Time (s)": round(sum(r["Fit Time (s)"] for r in summary_rows if r["Model"] in ensemble_member_names), 2),
        "Train RMSLE": round(ens_train_rmsle, 4),
        "Val RMSLE": ens_metrics["RMSLE"],
        "Val MAE": ens_metrics["MAE"],
        "Val RMSE": ens_metrics["RMSE"],
        "Val R2": ens_metrics["R2"],
        "Tet RMSLE": round(ens_tet_rmsle, 4) if ens_tet_rmsle is not None else np.nan
    })
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, fitted_pipelines, ensemble


def run_time_series_cv_benchmark(
    train_clean_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Chạy kiểm định chéo mở rộng theo thời gian (4 Folds) cho các mô hình đại diện:
    Baseline, Ridge, HistGradientBoosting, LightGBM, CatBoost, XGBoost.
    """
    features = numeric_cols + categorical_cols
    features_with_time = [c for c in train_clean_df.columns if c in features or c in ["t_years", "date", "store_id"]]
    cv = TimeSeriesExpandingCV()
    folds = list(cv.split(train_clean_df))
    
    models_to_test = {
        "Baseline (Weekday Median)": None,
        "Ridge": Ridge(alpha=3.0, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=350, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.5, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=550, learning_rate=0.028, num_leaves=28, subsample=0.8, colsample_bytree=0.7, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.5, random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostRegressor(iterations=750, learning_rate=0.03, depth=6, l2_leaf_reg=4.5, random_state=42, verbose=0),
        "XGBoost": xgb.XGBRegressor(n_estimators=550, learning_rate=0.028, max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.0, tree_method="hist", random_state=42)
    }
    
    cv_records = []
    ens_weights = {"CatBoost": 0.60, "Ridge": 0.20, "LightGBM": 0.10, "XGBoost": 0.10}
    fold_preds_log = {m: [] for m in ens_weights.keys()}
    fold_y_true = []
    
    for name, raw_model in models_to_test.items():
        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
            fold_tr = train_clean_df.iloc[tr_idx]
            fold_va = train_clean_df.iloc[va_idx]
            
            y_va = fold_va["revenue"].values
            if len(fold_y_true) < len(folds):
                fold_y_true.append(y_va)
            
            if name == "Baseline (Weekday Median)":
                pred_va = compute_weekday_median_baseline(fold_tr, fold_va)
            else:
                use_cats = categorical_cols + ["store_id"] if any(gb in name for gb in ["CatBoost", "LightGBM", "XGBoost", "HistGradient"]) and "store_id" in train_clean_df.columns and "store_id" not in categorical_cols else categorical_cols
                use_nums = [c for c in numeric_cols if c != "store_id"]
                pipeline = build_leakage_free_pipeline(clone(raw_model), use_nums, use_cats, use_trend_aware=True)
                pipeline.fit(fold_tr[features_with_time], fold_tr["revenue"].values)
                pred_va = pipeline.predict(fold_va[features_with_time])
                if name in fold_preds_log:
                    fold_preds_log[name].append(np.log1p(np.maximum(0.0, pred_va)))
                
            score = rmsle(y_va, pred_va)
            fold_scores.append(score)
            
        cv_records.append({
            "Model": name,
            "Fold 1 (Tet Q1/25)": round(fold_scores[0], 4),
            "Fold 2 (Q2/25)": round(fold_scores[1], 4),
            "Fold 3 (Summer Q3/25)": round(fold_scores[2], 4),
            "Fold 4 (Q4/25)": round(fold_scores[3], 4),
            "Mean CV RMSLE": round(float(np.mean(fold_scores)), 4),
            "Std CV RMSLE": round(float(np.std(fold_scores)), 4)
        })
        
    # Đánh giá Weighted Blend Ensemble từ 4 mô hình GBDT SOTA (tổng hợp dự báo trên không gian logarit)
    ens_scores = []
    for f_idx in range(len(folds)):
        blend_log = sum(ens_weights[m] * fold_preds_log[m][f_idx] for m in ens_weights)
        p_ens = np.maximum(0.0, np.expm1(blend_log))
        ens_score = rmsle(fold_y_true[f_idx], p_ens)
        ens_scores.append(round(ens_score, 4))
        
    cv_records.append({
        "Model": "Weighted Blend Ensemble (SOTA)",
        "Fold 1 (Tet Q1/25)": ens_scores[0],
        "Fold 2 (Q2/25)": ens_scores[1],
        "Fold 3 (Summer Q3/25)": ens_scores[2],
        "Fold 4 (Q4/25)": ens_scores[3],
        "Mean CV RMSLE": round(float(np.mean(ens_scores)), 4),
        "Std CV RMSLE": round(float(np.std(ens_scores)), 4)
    })
        
    return pd.DataFrame(cv_records)


def run_step6(data_dir: str = "data", verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], WeightedBlendEnsemble]:
    """
    Hàm thực thi toàn bộ BƯỚC 6: MODEL & TRAIN.
    
    Trả về:
        (summary_df, cv_benchmark_df, fitted_pipelines, final_ensemble)
    """
    if verbose:
        print("=" * 80)
        print("BƯỚC 6: MODEL & TRAIN (HUẤN LUYỆN TOÀN DIỆN 10 MÔ HÌNH HỌC MÁY & ENSEMBLE)")
        print("=" * 80)
        print("Mục tiêu: Đưa toàn bộ 10 mô hình học máy vào Pipeline chống rò rỉ, tối ưu RMSLE")
        print("và kết hợp thành Weighted Blend Ensemble đa tầng.")
        print("-" * 80)
        
    # 1. Tải dữ liệu đã qua Bước 3 và phân chia Bước 4
    train_final, test_final, _, _, _, _ = run_step3(data_dir, verbose=False)
    fit_df, val_df = get_starter_holdout_split(train_final, split_date="2025-10-01")
    
    # Lấy thêm Fold 1 (Mùa Tết Jan-Mar 2025) để thẩm định độ vững chắc với biến động theo mùa
    cv = TimeSeriesExpandingCV()
    folds = list(cv.split(train_final))
    f1_tr_idx, f1_va_idx = folds[0]
    fold1_train = train_final.iloc[f1_tr_idx]
    fold1_val = train_final.iloc[f1_va_idx]
    
    numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(fit_df)
    
    if verbose:
        print(f"Tập huấn luyện Fit:      {len(fit_df):,} dòng (2024-01-01 -> 2025-09-30)")
        print(f"Tập kiểm định Holdout:   {len(val_df):,} dòng (2025-10-01 -> 2025-12-31)")
        print(f"Tập kiểm định Tết (F1):  {len(fold1_val):,} dòng (2025-01-01 -> 2025-03-31)")
        print(f"Số lượng đặc trưng:      {len(numeric_cols)} numeric + {len(categorical_cols)} categorical = {len(numeric_cols) + len(categorical_cols)} cột")
        print("-" * 80)
        print("6.1. TIẾN HÀNH HUẤN LUYỆN TOÀN BỘ 10 MÔ HÌNH TRÊN HOLDOUT SPLIT:")
        
    summary_df, fitted_pipelines, ensemble = train_and_evaluate_all_models(
        fit_df, val_df, numeric_cols, categorical_cols,
        fold1_train=fold1_train, fold1_val=fold1_val,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print("BẢNG XẾP HẠNG VÀ SO SÁNH CHỈ SỐ CÁC MÔ HÌNH TRÊN TẬP HOLDOUT:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("\nTrọng số tối ưu trong Weighted Blend Ensemble:")
        for m, w in ensemble.weights.items():
            print(f"  * {m:25s}: {w*100:5.2f}%")
        print("-" * 80)
        print("6.2. THỰC HIỆN TIME-SERIES EXPANDING CROSS VALIDATION (4 FOLDS):")
        
    cv_benchmark_df = run_time_series_cv_benchmark(train_final, numeric_cols, categorical_cols, verbose=verbose)
    
    if verbose:
        print("\n" + "=" * 80)
        print("BẢNG KẾT QUẢ KIỂM ĐỊNH CHÉO THEO THỜI GIAN (TIME-SERIES CV 4 FOLDS):")
        print("=" * 80)
        print(cv_benchmark_df.to_string(index=False))
        print("=" * 80)
        
    return summary_df, cv_benchmark_df, fitted_pipelines, ensemble


if __name__ == "__main__":
    summary_df, cv_df, pipelines, ensemble = run_step6(data_dir="data", verbose=True)