"""Unit tests for LogTransformedLinearModel."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.models.linear import LogTransformedLinearModel
from src.features.builder import build_all_features
from src.data.join import join_all_data


class TestLinearModels(unittest.TestCase):

    def setUp(self):
        # Create synthetic data with enough samples to train linear models
        np.random.seed(42)
        n_samples = 120
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="D")
        stores = np.random.choice(["S01", "S02", "S03"], size=n_samples)
        promo = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        # Base revenue ~ 15, promo adds +8, store S01 has higher base
        base_rev = 15.0 + (stores == "S01") * 5.0 + promo * 8.0 + np.random.normal(0, 1.5, size=n_samples)
        revenue = np.clip(base_rev, 2.0, None)

        self.transactions = pd.DataFrame({
            "date": dates,
            "store_id": stores,
            "is_promo": promo,
            "revenue": revenue,
        })

        self.stores_raw = pd.DataFrame({
            "store_id": ["S01", "S02", "S03"],
            "city": ["Ha Noi", "Ha Noi", "Ho Chi Minh"],
            "district": ["Ba Dinh", "Dong Da", "District 1"],
            "store_type": ["Office", "street", "mall"],
            "area_m2": [120.0, 95.0, 85.0],
            "seats": [40, 30, 25],
            "has_parking": [1, 0, 1],
            "open_date": ["2020-05-15", "2021-12-25", "2023-01-10"],
        })

        self.holidays_raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-14"],
            "holiday_name": ["New Year", "Valentine"],
            "type": ["national_holiday", "event"],
        })

        self.weather_raw = pd.DataFrame({
            "date": list(dates) * 2,
            "city": ["Ha Noi"] * n_samples + ["Ho Chi Minh"] * n_samples,
            "temp_c": np.random.uniform(18.0, 32.0, size=2 * n_samples),
            "rain_mm": np.random.exponential(2.0, size=2 * n_samples),
        })

        joined = join_all_data(
            self.transactions, self.stores_raw, self.weather_raw, self.holidays_raw
        )
        self.features_df = build_all_features(joined)

    def test_fit_and_predict_ridge(self):
        model = LogTransformedLinearModel(model_type="ridge", feature_tier="E2_promo", alpha=1.0)
        model.fit(self.features_df, self.features_df["revenue"])
        self.assertTrue(model.is_fitted)

        preds = model.predict(self.features_df)
        self.assertEqual(len(preds), len(self.features_df))
        self.assertTrue(np.all(preds >= 0.0), "Predictions must be non-negative")
        # Check prediction magnitude is in reasonable revenue range
        self.assertGreater(float(np.mean(preds)), 5.0)
        self.assertLess(float(np.mean(preds)), 40.0)

    def test_promo_coefficient_positive(self):
        """Promo was constructed to add revenue; linear model should estimate positive coefficient for promo."""
        model = LogTransformedLinearModel(model_type="ridge", feature_tier="E2_promo", alpha=1.0)
        model.fit(self.features_df, self.features_df["revenue"])
        coefs = model.get_coefficients()
        promo_rows = coefs[coefs["feature"].str.contains("is_promo")]
        self.assertFalse(promo_rows.empty)
        promo_coef = promo_rows.iloc[0]["coefficient"]
        self.assertGreater(promo_coef, 0.0, "Promotion coefficient must be positive")

    def test_save_and_load_serialization(self):
        model = LogTransformedLinearModel(model_type="ridge", feature_tier="E2_promo", alpha=2.5)
        model.fit(self.features_df, self.features_df["revenue"])
        orig_preds = model.predict(self.features_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_model.joblib"
            model.save(tmp_path)
            self.assertTrue(tmp_path.exists())

            loaded = LogTransformedLinearModel.load(tmp_path)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.alpha, 2.5)
            loaded_preds = loaded.predict(self.features_df)
            np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)

    def test_multiple_model_types(self):
        for m_type in ["elastic_net", "huber", "lasso"]:
            model = LogTransformedLinearModel(model_type=m_type, feature_tier="E1_calendar_store", alpha=0.5)
            model.fit(self.features_df, self.features_df["revenue"])
            preds = model.predict(self.features_df)
            self.assertEqual(len(preds), len(self.features_df))
            self.assertTrue(np.all(preds >= 0.0))


if __name__ == "__main__":
    unittest.main()
