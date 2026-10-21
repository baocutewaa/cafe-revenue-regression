"""Unit tests for TargetTransformer and TransformedTargetTreeModel."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.models.target_transform import (
    TargetTransformer,
    TransformedTargetTreeModel,
)


class TestTargetTransformer(unittest.TestCase):
    """Test suite for TargetTransformer transformations, invertibility, and bias adjustments."""

    def setUp(self):
        np.random.seed(42)
        # Realistic non-negative revenue sample
        self.y_train = np.random.lognormal(mean=2.8, sigma=0.5, size=500)
        self.y_val = np.random.lognormal(mean=2.8, sigma=0.5, size=100)

    def test_invertibility_raw(self):
        tt = TargetTransformer(transform_type="raw")
        tt.fit(self.y_train)
        z = tt.transform(self.y_val)
        y_recon = tt.inverse_transform(z)
        np.testing.assert_allclose(self.y_val, y_recon, rtol=1e-5)

    def test_invertibility_log1p(self):
        tt = TargetTransformer(transform_type="log1p")
        tt.fit(self.y_train)
        z = tt.transform(self.y_val)
        y_recon = tt.inverse_transform(z)
        np.testing.assert_allclose(self.y_val, y_recon, rtol=1e-5)

    def test_invertibility_sqrt(self):
        tt = TargetTransformer(transform_type="sqrt")
        tt.fit(self.y_train)
        z = tt.transform(self.y_val)
        y_recon = tt.inverse_transform(z)
        np.testing.assert_allclose(self.y_val, y_recon, rtol=1e-5)

    def test_invertibility_asinh(self):
        tt = TargetTransformer(transform_type="asinh")
        tt.fit(self.y_train)
        z = tt.transform(self.y_val)
        y_recon = tt.inverse_transform(z)
        np.testing.assert_allclose(self.y_val, y_recon, rtol=1e-5)

    def test_invertibility_yeo_johnson(self):
        tt = TargetTransformer(transform_type="yeo_johnson")
        tt.fit(self.y_train)
        self.assertIsNotNone(tt.lambda_)
        z = tt.transform(self.y_val)
        y_recon = tt.inverse_transform(z)
        np.testing.assert_allclose(self.y_val, y_recon, rtol=1e-4)

    def test_non_negative_clipping(self):
        tt = TargetTransformer(transform_type="log1p")
        tt.fit(self.y_train)
        # Negative z in log space corresponds to negative expm1
        z_neg = np.array([-1.0, -0.5, -10.0])
        y_recon = tt.inverse_transform(z_neg)
        self.assertTrue(np.all(y_recon >= 0.0))
        self.assertEqual(list(y_recon), [0.0, 0.0, 0.0])

    def test_bias_correction_parameters(self):
        tt = TargetTransformer(transform_type="log1p", bias_correction="half_variance")
        # Ensure positive residual variance
        z_pred_dummy = np.log1p(self.y_train) + np.random.normal(0.05, 0.1, size=len(self.y_train))
        tt.fit(self.y_train, y_pred_train=z_pred_dummy)

        self.assertGreater(tt.residual_variance_, 0.0)
        self.assertGreater(tt.duan_factor_, 0.0)

        z_sample = np.array([2.5, 3.0, 3.5])
        y_unadj = tt.inverse_transform(z_sample, override_bias_correction="none")
        y_half_var = tt.inverse_transform(z_sample, override_bias_correction="half_variance")

        # Because variance > 0, half_variance adjusted predictions should be strictly greater than unadjusted
        self.assertTrue(np.all(y_half_var > y_unadj))


class TestTransformedTargetTreeModel(unittest.TestCase):
    """Test suite for TransformedTargetTreeModel integration."""

    def setUp(self):
        np.random.seed(42)
        n_samples = 300
        self.df = pd.DataFrame({
            "store_type": np.random.choice(["mall", "street", "drive_thru"], n_samples),
            "city": np.random.choice(["Hanoi", "HCM", "DaNang"], n_samples),
            "store_id": np.random.choice(["S01", "S02", "S03"], n_samples),
            "area_m2": np.random.uniform(50, 200, n_samples),
            "seats": np.random.uniform(20, 80, n_samples),
            "store_age_days": np.random.uniform(100, 1000, n_samples),
            "temp_c": np.random.uniform(20, 35, n_samples),
            "rain_mm": np.random.exponential(5, n_samples),
            "day_of_year": np.random.randint(1, 365, n_samples),
            "is_promo": np.random.choice([0, 1], n_samples),
            "is_weekend": np.random.choice([0, 1], n_samples),
            "is_holiday": np.random.choice([0, 1], n_samples),
            "is_rainy": np.random.choice([0, 1], n_samples),
            "revenue": np.random.lognormal(mean=2.8, sigma=0.4, size=n_samples),
        })
        self.feature_cols = [c for c in self.df.columns if c != "revenue"]

    def test_fit_predict_log_huber(self):
        model = TransformedTargetTreeModel(
            transform_type="log1p",
            objective="huber",
            n_estimators=30,
            learning_rate=0.1,
            num_leaves=15,
        )
        model.fit(self.df[self.feature_cols], self.df["revenue"])
        self.assertTrue(model.is_fitted)

        preds = model.predict(self.df[self.feature_cols])
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_fit_predict_tweedie(self):
        model = TransformedTargetTreeModel(
            transform_type="raw",
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=30,
            learning_rate=0.1,
            num_leaves=15,
        )
        model.fit(self.df[self.feature_cols], self.df["revenue"])
        preds = model.predict(self.df[self.feature_cols])
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(preds >= 0.0))

    def test_fit_predict_yeo_johnson(self):
        model = TransformedTargetTreeModel(
            transform_type="yeo_johnson",
            n_estimators=30,
            learning_rate=0.1,
            num_leaves=15,
        )
        model.fit(self.df[self.feature_cols], self.df["revenue"])
        preds = model.predict(self.df[self.feature_cols])
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(preds >= 0.0))

    def test_save_and_load(self):
        model = TransformedTargetTreeModel(
            transform_type="log1p",
            objective="huber",
            n_estimators=20,
            num_leaves=10,
        )
        model.fit(self.df[self.feature_cols], self.df["revenue"])
        preds_orig = model.predict(self.df[self.feature_cols])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_target_model.joblib"
            model.save(save_path)
            loaded_model = TransformedTargetTreeModel.load(save_path)

            preds_loaded = loaded_model.predict(self.df[self.feature_cols])
            np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
