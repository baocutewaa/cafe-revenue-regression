"""Unit tests for evaluation metrics."""

import unittest
import numpy as np
from src.evaluation.metrics import rmsle, mae, rmse, log_residual, compute_metrics


class TestMetrics(unittest.TestCase):

    def test_rmsle_identical_inputs(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        score = rmsle(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_rmsle_known_value(self):
        y_true = np.array([0.0, 3.0])
        # log1p(0) = 0, log1p(3) = log(4) = 1.386294
        # if pred is [0.0, 1.0], log1p(0)=0, log1p(1) = log(2) = 0.693147
        # diffs: [0 - 0, log(2) - log(4)] = [0, -log(2)]
        # squared mean: log(2)^2 / 2 = 0.480453 / 2 = 0.240226
        # sqrt: log(2) / sqrt(2) = 0.693147 / 1.414213 = 0.490129
        y_pred = np.array([0.0, 1.0])
        expected = np.log(2) / np.sqrt(2)
        score = rmsle(y_true, y_pred)
        self.assertAlmostEqual(score, expected, places=5)

    def test_rmsle_clips_negative_predictions(self):
        y_true = np.array([0.0, 5.0])
        y_pred_neg = np.array([-10.0, 5.0])
        y_pred_zero = np.array([0.0, 5.0])
        self.assertAlmostEqual(rmsle(y_true, y_pred_neg), rmsle(y_true, y_pred_zero), places=6)

    def test_rmsle_raises_on_negative_true_values(self):
        y_true = np.array([-1.0, 10.0])
        y_pred = np.array([10.0, 10.0])
        with self.assertRaises(ValueError):
            rmsle(y_true, y_pred)

    def test_mae_and_rmse(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 16.0, 35.0])
        self.assertAlmostEqual(mae(y_true, y_pred), (2 + 4 + 5) / 3, places=5)
        self.assertAlmostEqual(rmse(y_true, y_pred), np.sqrt((4 + 16 + 25) / 3), places=5)

    def test_log_residual_sign(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([5.0, 30.0])  # first underpredicted (r > 0), second overpredicted (r < 0)
        residuals = log_residual(y_true, y_pred)
        self.assertGreater(residuals[0], 0)
        self.assertLess(residuals[1], 0)

    def test_compute_metrics_dict(self):
        y_true = [10.0, 20.0]
        y_pred = [10.0, 20.0]
        res = compute_metrics(y_true, y_pred)
        self.assertEqual(res["rmsle"], 0.0)
        self.assertEqual(res["mae"], 0.0)
        self.assertEqual(res["rmse"], 0.0)
        self.assertEqual(res["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
