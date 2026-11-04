"""
Unit tests for Step 7: EVALUATE & DIAGNOSTICS
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Chèn đường dẫn gốc để import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.step7_evaluate import (
    compute_ablation_study_table,
    compute_store_type_error_breakdown,
    compute_residual_diagnostics,
    compute_feature_importances,
    generate_evaluation_plots
)


class TestStep7Evaluate(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.val_df = pd.DataFrame({
            "store_type": np.random.choice(["mall", "street", "office"], size=n),
            "days_since_open": np.random.randint(5, 500, size=n)
        })
        self.y_true = np.random.uniform(10.0, 30.0, size=n)
        self.y_pred = self.y_true + np.random.normal(0, 1.0, size=n)
        self.y_baseline = self.y_true + np.random.normal(0, 3.0, size=n)

    def test_compute_ablation_study_table(self):
        """Kiểm tra cấu trúc và tính liên tục của bảng Ablation Study."""
        df_ablation = compute_ablation_study_table()
        self.assertEqual(len(df_ablation), 7)
        self.assertIn("Step", df_ablation.columns)
        self.assertIn("Holdout RMSLE", df_ablation.columns)
        self.assertIn("Seasonal Tet RMSLE", df_ablation.columns)
        # Điểm RMSLE phải giảm đơn điệu
        holdout_scores = df_ablation["Holdout RMSLE"].values
        self.assertTrue(all(x >= y for x, y in zip(holdout_scores, holdout_scores[1:])))

    def test_compute_store_type_error_breakdown(self):
        """Kiểm tra phân tích sai số theo store_type."""
        st_df = compute_store_type_error_breakdown(self.val_df, self.y_true, self.y_pred, self.y_baseline)
        self.assertGreaterEqual(len(st_df), 1)
        self.assertIn("Store Type", st_df.columns)
        self.assertIn("Ensemble RMSLE", st_df.columns)
        self.assertIn("Baseline RMSLE", st_df.columns)
        self.assertIn("Improvement (%)", st_df.columns)
        # Ensemble phải tốt hơn Baseline trên dữ liệu giả lập này
        self.assertTrue(all(st_df["Ensemble RMSLE"] < st_df["Baseline RMSLE"]))

    def test_compute_residual_diagnostics(self):
        """Kiểm tra chẩn đoán phần dư logarit."""
        diag = compute_residual_diagnostics(self.y_true, self.y_pred)
        self.assertIn("mean_residual", diag)
        self.assertIn("std_residual", diag)
        self.assertIn("skewness", diag)
        self.assertIn("p50_abs_log_err", diag)
        # Trung bình sai số phải gần 0
        self.assertAlmostEqual(diag["mean_residual"], 0.0, delta=0.2)

    def test_compute_feature_importances(self):
        """Kiểm tra hàm tính tầm quan trọng đặc trưng."""
        class MockTreeModel:
            def __init__(self, n_feat):
                self.feature_importances_ = np.random.uniform(1, 100, size=n_feat)
                
        class MockPipeline:
            def __init__(self, n_feat):
                class MockRegressor:
                    def __init__(self, n_feat):
                        self.named_steps = {"model": MockTreeModel(n_feat)}
                self.regressor_ = MockRegressor(n_feat)
                
        feature_names = [f"feat_{i}" for i in range(10)]
        mock_pipe = MockPipeline(10)
        fi_df = compute_feature_importances(mock_pipe, feature_names, top_n=5)
        self.assertEqual(len(fi_df), 5)
        self.assertIn("feature", fi_df.columns)
        self.assertIn("relative_importance", fi_df.columns)
        self.assertAlmostEqual(fi_df["relative_importance"].sum(), 100.0, delta=50.0)

    def test_generate_evaluation_plots(self):
        """Kiểm tra tạo 6 biểu đồ đánh giá ra thư mục plots."""
        output_dir = "plots_test"
        feature_names = [f"feat_{i}" for i in range(15)]
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": np.arange(15, 0, -1),
            "relative_importance": np.linspace(20, 1, 15)
        })
        ablation_df = compute_ablation_study_table()
        
        plots = generate_evaluation_plots(
            self.val_df, self.y_true, self.y_pred, self.y_baseline,
            fi_df, ablation_df, output_dir=output_dir
        )
        self.assertEqual(len(plots), 6)
        for name, path in plots.items():
            self.assertTrue(os.path.exists(path))
            os.remove(path)
        if os.path.exists(output_dir):
            os.rmdir(output_dir)


if __name__ == "__main__":
    unittest.main()
