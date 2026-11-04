"""
Unit Tests cho BƯỚC 5: SCALE & LEAKAGE-FREE PIPELINE
Tệp: tests/test_step5.py
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from src.step1_collect_prepare import rmsle
from src.step3_preprocess import run_step3
from src.step4_split import get_starter_holdout_split
from src.step5_scale import (
    transform_target,
    inverse_transform_target,
    LogTargetRegressorWrapper,
    get_feature_column_types,
    build_feature_preprocessor,
    build_leakage_free_pipeline,
    run_step5
)

class TestStep5Scale(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.train_final, cls.test_final, _, _, _, _ = run_step3("data", verbose=False)
        cls.fit_df, cls.val_df = get_starter_holdout_split(cls.train_final, "2025-10-01")
        cls.num_cols, cls.cat_cols, cls.ignore_cols = get_feature_column_types(cls.fit_df)
        
    def test_target_transformation_exact(self):
        y = np.array([0.0, 10.0, 25.5, 100.0])
        y_log = transform_target(y)
        y_recov = inverse_transform_target(y_log)
        
        np.testing.assert_allclose(y, y_recov, rtol=1e-6, atol=1e-6)
        
        # Kiểm tra clipping số âm
        y_neg = np.array([-5.0, -1.0, 0.0])
        y_neg_log = transform_target(y_neg)
        np.testing.assert_array_equal(y_neg_log, np.zeros(3))
        
    def test_get_feature_column_types(self):
        self.assertEqual(set(self.cat_cols), {"city", "district", "store_type", "holiday_name_clean"})
        self.assertNotIn("revenue", self.num_cols)
        self.assertNotIn("date", self.num_cols)
        self.assertGreater(len(self.num_cols), 40)
        
    def test_build_feature_preprocessor_no_leakage(self):
        preprocessor = build_feature_preprocessor(self.num_cols, self.cat_cols)
        
        X_fit = self.fit_df.drop(columns=["revenue"])
        X_val = self.val_df.drop(columns=["revenue"])
        
        X_fit_trans = preprocessor.fit_transform(X_fit)
        X_val_trans = preprocessor.transform(X_val)
        
        self.assertEqual(X_fit_trans.shape[0], len(self.fit_df))
        self.assertEqual(X_val_trans.shape[0], len(self.val_df))
        self.assertEqual(X_fit_trans.shape[1], X_val_trans.shape[1])
        self.assertEqual(np.isnan(X_fit_trans).sum(), 0)
        self.assertEqual(np.isnan(X_val_trans).sum(), 0)
        
        # Xác nhận Scaler học tham số chỉ từ fit_df
        scaler_step = preprocessor.named_transformers_["num"].named_steps["scaler"]
        self.assertEqual(len(scaler_step.mean_), len(self.num_cols))
        
    def test_build_leakage_free_pipeline_with_ridge(self):
        pipeline = build_leakage_free_pipeline(
            regressor=Ridge(alpha=1.0),
            numeric_cols=self.num_cols,
            categorical_cols=self.cat_cols,
            scaler_type="standard",
            use_log_target=True
        )
        
        X_fit = self.fit_df.drop(columns=["revenue"])
        y_fit = self.fit_df["revenue"].values
        X_val = self.val_df.drop(columns=["revenue"])
        y_val = self.val_df["revenue"].values
        
        pipeline.fit(X_fit, y_fit)
        preds = pipeline.predict(X_val)
        
        self.assertEqual(len(preds), len(y_val))
        self.assertTrue((preds >= 0).all())
        self.assertEqual(np.isnan(preds).sum(), 0)
        
        score = rmsle(y_val, preds)
        # Điểm RMSLE phải vượt trội hơn Baseline 0.2276
        self.assertLess(score, 0.15)
        
    def test_log_target_regressor_wrapper(self):
        wrapper = LogTargetRegressorWrapper(Ridge(alpha=1.0))
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        self.assertEqual(len(preds), 4)
        self.assertTrue((preds > 0).all())
        
    def test_run_step5_end_to_end(self):
        preprocessor, pipeline, metrics = run_step5("data", verbose=False)
        self.assertIn("ridge_val_rmsle", metrics)
        self.assertIn("ridge_fold1_tet_rmsle", metrics)
        self.assertLess(metrics["ridge_val_rmsle"], 0.15)
        self.assertLess(metrics["ridge_fold1_tet_rmsle"], 0.18)

if __name__ == "__main__":
    unittest.main()
