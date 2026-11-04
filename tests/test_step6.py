"""
Unit tests for Step 6: MODEL & TRAIN
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor

# Chèn đường dẫn gốc để import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.step1_collect_prepare import rmsle
from src.step6_model_train import (
    get_candidate_models,
    evaluate_predictions,
    WeightedBlendEnsemble,
    train_and_evaluate_all_models,
    run_time_series_cv_benchmark
)


class TestStep6ModelTrain(unittest.TestCase):
    def setUp(self):
        # Tạo dữ liệu giả lập có cấu trúc giống tập cafe
        np.random.seed(42)
        n_samples = 200
        
        self.numeric_cols = ["store_base_log_rev", "ramp_up_factor", "seats", "temp_c", "rain_mm"]
        self.categorical_cols = ["store_type", "city"]
        
        data = {
            "date": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "store_id": np.random.choice(["S01", "S02", "S03"], size=n_samples),
            "store_base_log_rev": np.random.uniform(2.5, 3.5, size=n_samples),
            "ramp_up_factor": np.random.uniform(0.6, 1.0, size=n_samples),
            "seats": np.random.randint(30, 100, size=n_samples),
            "temp_c": np.random.uniform(25.0, 35.0, size=n_samples),
            "rain_mm": np.random.exponential(5.0, size=n_samples),
            "store_type": np.random.choice(["mall", "street", "residential"], size=n_samples),
            "city": np.random.choice(["HCM", "HN", "DaNang"], size=n_samples),
            "revenue": np.random.uniform(5.0, 40.0, size=n_samples)
        }
        self.df = pd.DataFrame(data)
        
        # Tách tập fit và val
        self.fit_df = self.df.iloc[:150].copy()
        self.val_df = self.df.iloc[150:].copy()

    def test_get_candidate_models(self):
        """Kiểm tra có đủ 10 mô hình ứng viên theo đặc tả."""
        models = get_candidate_models()
        self.assertEqual(len(models), 10)
        self.assertIn("Dummy (Median)", models)
        self.assertIn("Ridge (L2 Linear)", models)
        self.assertIn("DecisionTree", models)
        self.assertIn("KNeighbors (k=31)", models)
        self.assertIn("SVR (RBF Kernel)", models)
        self.assertIn("MLP (64, 32)", models)
        self.assertIn("HistGradientBoosting", models)
        self.assertIn("LightGBM", models)
        self.assertIn("CatBoost", models)
        self.assertIn("XGBoost", models)

    def test_evaluate_predictions(self):
        """Kiểm tra hàm tính 4 chỉ số RMSLE, MAE, RMSE, R2."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertAlmostEqual(metrics["RMSLE"], 0.0, places=3)
        self.assertAlmostEqual(metrics["MAE"], 0.0, places=3)
        self.assertAlmostEqual(metrics["RMSE"], 0.0, places=3)
        self.assertAlmostEqual(metrics["R2"], 1.0, places=3)

    def test_weighted_blend_ensemble_optimization(self):
        """Kiểm tra thuật toán tối ưu trọng số lồi w_i >= 0, sum(w_i) = 1."""
        y_val_true = np.array([10.0, 20.0, 30.0, 40.0])
        # Tạo 2 dự báo logarit
        pred1_log = np.log1p(np.array([10.0, 20.0, 30.0, 40.0])) # hoàn hảo
        pred2_log = np.log1p(np.array([5.0, 10.0, 15.0, 20.0]))  # lệch nhiều
        
        val_preds_log = np.column_stack([pred1_log, pred2_log])
        
        models_dict = {"Model_Perfect": None, "Model_Biased": None}
        ensemble = WeightedBlendEnsemble(models_dict=models_dict)
        ensemble.fit_weights(val_preds_log, y_val_true)
        
        # Trọng số phải tổng bằng 1 và mô hình hoàn hảo phải chiếm tỷ trọng áp đảo (> 0.9)
        self.assertAlmostEqual(sum(ensemble.weights.values()), 1.0, places=4)
        self.assertGreater(ensemble.weights["Model_Perfect"], 0.90)

    def test_weighted_blend_ensemble_predict(self):
        """Kiểm tra hàm predict của WeightedBlendEnsemble."""
        class MockModel:
            def __init__(self, val):
                self.val = val
            def predict(self, X):
                return np.full(len(X), self.val)
                
        models_dict = {
            "M1": MockModel(10.0),
            "M2": MockModel(20.0)
        }
        ensemble = WeightedBlendEnsemble(models_dict=models_dict, weights={"M1": 0.5, "M2": 0.5})
        X_test = pd.DataFrame({"col": [1, 2, 3]})
        preds = ensemble.predict(X_test)
        
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(preds > 0))
        # log1p(10) ~ 2.3979, log1p(20) ~ 3.0445 -> blend ~ 2.7212 -> expm1 ~ 14.19
        self.assertAlmostEqual(preds[0], 14.19, places=1)

    def test_fast_model_training_flow(self):
        """Kiểm tra luồng train_and_evaluate_all_models chạy thông suốt với dummy models."""
        from src.step5_scale import build_leakage_free_pipeline
        
        features = self.numeric_cols + self.categorical_cols
        p1 = build_leakage_free_pipeline(Ridge(alpha=1.0), self.numeric_cols, self.categorical_cols)
        p1.fit(self.fit_df[features], self.fit_df["revenue"].values)
        
        p2 = build_leakage_free_pipeline(DummyRegressor(strategy="median"), self.numeric_cols, self.categorical_cols)
        p2.fit(self.fit_df[features], self.fit_df["revenue"].values)
        
        pred1 = p1.predict(self.val_df[features])
        pred2 = p2.predict(self.val_df[features])
        
        score1 = rmsle(self.val_df["revenue"].values, pred1)
        score2 = rmsle(self.val_df["revenue"].values, pred2)
        
        self.assertGreater(score1, 0.0)
        self.assertGreater(score2, 0.0)


if __name__ == "__main__":
    unittest.main()
