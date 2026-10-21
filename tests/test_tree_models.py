"""Unit tests for LogTransformedTreeModel and tree feature engineering."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.models.tree import LogTransformedTreeModel
from src.features.tree_builder import (
    ALL_TREE_FEATURES,
    TREE_CATEGORICAL_COLS,
    TREE_NUMERIC_COLS,
    TREE_BINARY_COLS,
    correct_revenue_outliers,
    compute_historical_target_stats,
    apply_historical_target_stats,
)


class TestTreeModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n_samples = 150
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="D")
        stores = np.random.choice(["S01", "S02", "S03"], size=n_samples)
        promo = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        rev = 15.0 + (stores == "S01") * 6.0 + promo * 10.0 + np.random.normal(0, 1.0, size=n_samples)
        rev = np.clip(rev, 2.0, None)

        self.df = pd.DataFrame({
            "date": dates,
            "store_id": stores,
            "store_type": np.random.choice(["office", "street", "mall"], size=n_samples),
            "city": np.random.choice(["Ha Noi", "Ho Chi Minh"], size=n_samples),
            "district": np.random.choice(["Dist 1", "Dist 2"], size=n_samples),
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "day": dates.day,
            "day_of_year": dates.dayofyear,
            "area_m2": np.random.uniform(60, 200, size=n_samples),
            "seats": np.random.randint(20, 80, size=n_samples),
            "seats_per_m2": np.random.uniform(0.2, 0.6, size=n_samples),
            "store_age_days": np.random.uniform(100, 1000, size=n_samples),
            "store_age_months": np.random.uniform(3, 33, size=n_samples),
            "temp_c": np.random.uniform(18, 35, size=n_samples),
            "rain_mm": np.random.exponential(1.5, size=n_samples),
            "sin_month": np.sin(2 * np.pi * dates.month / 12),
            "cos_month": np.cos(2 * np.pi * dates.month / 12),
            "sin_doy": np.sin(2 * np.pi * dates.dayofyear / 365.25),
            "cos_doy": np.cos(2 * np.pi * dates.dayofyear / 365.25),
            "has_parking": np.random.choice([0, 1], size=n_samples),
            "area_m2_missing": [0] * n_samples,
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "is_month_start": (dates.day == 1).astype(int),
            "is_month_end": (dates.is_month_end).astype(int),
            "is_month_mid": ((dates.day >= 10) & (dates.day <= 20)).astype(int),
            "is_new_store": [0] * n_samples,
            "is_promo": promo,
            "promo_x_weekend": (promo & (dates.dayofweek >= 5)).astype(int),
            "is_holiday": [0] * n_samples,
            "holiday_count": [0] * n_samples,
            "is_national_holiday": [0] * n_samples,
            "is_event": [0] * n_samples,
            "is_rainy": (np.random.uniform(0, 10, size=n_samples) > 0).astype(int),
            "rain_missing": [0] * n_samples,
            "temp_missing": [0] * n_samples,
            "revenue": rev,
        })

        # Add target statistics
        stats = compute_historical_target_stats(self.df)
        self.df = apply_historical_target_stats(self.df, stats)

    def test_outlier_correction(self):
        df_outlier = self.df.copy()
        df_outlier.loc[0, "revenue"] = 29490.0  # Thousand VND
        df_outlier.loc[1, "revenue"] = 53600.0  # Thousand VND

        cleaned, n_corr = correct_revenue_outliers(df_outlier)
        self.assertEqual(n_corr, 2)
        self.assertAlmostEqual(cleaned.loc[0, "revenue"], 29.49)
        self.assertAlmostEqual(cleaned.loc[1, "revenue"], 53.60)

    def test_fit_and_predict_lightgbm(self):
        model = LogTransformedTreeModel(model_type="lightgbm", n_estimators=50, random_state=42)
        model.fit(self.df, self.df["revenue"])
        self.assertTrue(model.is_fitted)

        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(preds >= 0.0), "Predictions must be non-negative")
        self.assertGreater(float(np.mean(preds)), 5.0)

        # Feature importances
        imp = model.get_feature_importances()
        self.assertFalse(imp.empty)
        self.assertIn("feature", imp.columns)
        self.assertIn("importance", imp.columns)

    def test_multiple_tree_families(self):
        for m_type in ["hist_gradient_boosting", "random_forest", "extra_trees"]:
            model = LogTransformedTreeModel(
                model_type=m_type,
                n_estimators=30 if "estimators" in m_type or "forest" in m_type or "trees" in m_type else None,
                max_iter=30 if "hist" in m_type else None,
                random_state=42,
            )
            model.fit(self.df, self.df["revenue"])
            preds = model.predict(self.df)
            self.assertEqual(len(preds), len(self.df))
            self.assertTrue(np.all(preds >= 0.0))

    def test_save_and_load(self):
        model = LogTransformedTreeModel(model_type="lightgbm", n_estimators=40, random_state=42)
        model.fit(self.df, self.df["revenue"])
        orig_preds = model.predict(self.df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_tree.joblib"
            model.save(tmp_path)
            self.assertTrue(tmp_path.exists())

            loaded = LogTransformedTreeModel.load(tmp_path)
            self.assertTrue(loaded.is_fitted)
            loaded_preds = loaded.predict(self.df)
            np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
