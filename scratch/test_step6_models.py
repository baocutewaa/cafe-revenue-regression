import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import numpy as np
import pandas as pd
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
from scipy.optimize import minimize

from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import get_starter_holdout_split, TimeSeriesExpandingCV, compute_weekday_median_baseline
from src.step5_scale import build_leakage_free_pipeline, get_feature_column_types

def run():
    print("Loading preprocessed dataset (Step 3)...")
    train_final, test_final, _, _, _, _ = run_step3("data", verbose=False)
    
    fit_df, val_df = get_starter_holdout_split(train_final, split_date="2025-10-01")
    numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(fit_df)
    features = numeric_cols + categorical_cols
    
    X_train = fit_df[features]
    y_train = fit_df["revenue"].values
    X_val = val_df[features]
    y_val = val_df["revenue"].values
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    # Also evaluate on Fold 1 of Expanding CV (Tet season Jan-Mar 2025)
    cv = TimeSeriesExpandingCV()
    folds = list(cv.split(train_final))
    f1_train_idx, f1_val_idx = folds[0]
    fold1_train = train_final.iloc[f1_train_idx]
    fold1_val = train_final.iloc[f1_val_idx]
    X_f1_train = fold1_train[features]
    y_f1_train = fold1_train["revenue"].values
    X_f1_val = fold1_val[features]
    y_f1_val = fold1_val["revenue"].values
    
    models = {
        "1. Dummy (Median)": DummyRegressor(strategy="median"),
        "2. Ridge (alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "3. DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=15, random_state=42),
        "4. KNN (k=31)": KNeighborsRegressor(n_neighbors=31, weights="distance"),
        "5. SVR (RBF)": SVR(kernel="rbf", C=2.0, epsilon=0.02, max_iter=3000),
        "6. MLP (64, 32)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True),
        "7. HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42),
        "8. LightGBM": lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1),
        "9. CatBoost": cb.CatBoostRegressor(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=3.0, random_state=42, verbose=0),
        "10. XGBoost": xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42)
    }
    
    # Baseline 0: Starter Weekday Median
    bl_val_pred = compute_weekday_median_baseline(fit_df, val_df)
    bl_f1_pred = compute_weekday_median_baseline(fold1_train, fold1_val)
    print(f"Starter Weekday Median -> Val RMSLE: {rmsle(y_val, bl_val_pred):.4f}, Tet RMSLE: {rmsle(y_f1_val, bl_f1_pred):.4f}")
    
    results = []
    val_preds = {}
    f1_preds = {}
    
    for name, raw_model in models.items():
        t0 = time.time()
        pipeline = build_leakage_free_pipeline(raw_model, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        pred_train = pipeline.predict(X_train)
        pred_val = pipeline.predict(X_val)
        val_score = rmsle(y_val, pred_val)
        train_score = rmsle(y_train, pred_train)
        val_preds[name] = pred_val
        
        # Fit on Fold 1
        pipeline_f1 = build_leakage_free_pipeline(raw_model, numeric_cols, categorical_cols)
        pipeline_f1.fit(X_f1_train, y_f1_train)
        pred_f1 = pipeline_f1.predict(X_f1_val)
        f1_score = rmsle(y_f1_val, pred_f1)
        f1_preds[name] = pred_f1
        
        print(f"{name:20s} | Fit: {fit_time:5.2f}s | Train RMSLE: {train_score:.4f} | Val RMSLE: {val_score:.4f} | Tet RMSLE: {f1_score:.4f}")
        results.append({
            "model": name,
            "fit_time": fit_time,
            "train_rmsle": train_score,
            "val_rmsle": val_score,
            "tet_rmsle": f1_score
        })
        
    print("\n--- Optimizing Weighted Blend Ensemble ---")
    ensemble_candidates = ["2. Ridge (alpha=1.0)", "7. HistGB", "8. LightGBM", "9. CatBoost", "10. XGBoost"]
    P_val_log = np.column_stack([np.log1p(val_preds[m]) for m in ensemble_candidates])
    y_val_log = np.log1p(y_val)
    
    def loss_func(w):
        w = w / np.sum(w)
        pred_blend_log = P_val_log @ w
        return np.sqrt(np.mean((pred_blend_log - y_val_log) ** 2))
        
    init_w = np.ones(len(ensemble_candidates)) / len(ensemble_candidates)
    bnds = [(0, 1) for _ in ensemble_candidates]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    res = minimize(loss_func, init_w, method='SLSQP', bounds=bnds, constraints=cons)
    best_w = res.x / np.sum(res.x)
    
    for m, w in zip(ensemble_candidates, best_w):
        print(f"  Weight for {m:20s}: {w:.4f}")
        
    blend_val_log = P_val_log @ best_w
    blend_val_pred = np.clip(np.expm1(blend_val_log), 0, None)
    blend_val_rmsle = rmsle(y_val, blend_val_pred)
    
    P_f1_log = np.column_stack([np.log1p(f1_preds[m]) for m in ensemble_candidates])
    blend_f1_log = P_f1_log @ best_w
    blend_f1_pred = np.clip(np.expm1(blend_f1_log), 0, None)
    blend_f1_rmsle = rmsle(y_f1_val, blend_f1_pred)
    
    print(f"\nFinal Weighted Ensemble -> Val RMSLE: {blend_val_rmsle:.4f} | Tet RMSLE: {blend_f1_rmsle:.4f}")

if __name__ == "__main__":
    run()
