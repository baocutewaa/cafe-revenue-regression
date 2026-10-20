"""Unit tests for WeekdayMedianBaseline."""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.baseline import WeekdayMedianBaseline


class TestBaselineModel(unittest.TestCase):

    def setUp(self):
        # Create synthetic history DataFrame
        # Store 1: Mondays -> 10, 12 (median 11); Tuesdays -> 20
        # Store 2: Mondays -> 30, 32 (median 31)
        self.history = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-02", "2024-01-01", "2024-01-08"]),
            "store_id": ["S01", "S01", "S01", "S02", "S02"],
            "revenue": [10.0, 12.0, 20.0, 30.0, 32.0],
        })
        # Global median: [10, 12, 20, 30, 32] -> 20.0
        # S01 median: [10, 12, 20] -> 12.0
        # S02 median: [30, 32] -> 31.0

    def test_fit_and_predict_exact_match(self):
        model = WeekdayMedianBaseline().fit(self.history)
        self.assertTrue(model.is_fitted)

        target = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15"]),  # Monday
            "store_id": ["S01"],
        })
        pred = model.predict(target)
        self.assertEqual(len(pred), 1)
        self.assertAlmostEqual(pred[0], 11.0)

    def test_fallback_to_store_median(self):
        model = WeekdayMedianBaseline().fit(self.history)

        # Wednesday for S01 -> not in history for S01
        target = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-03"]),  # Wednesday
            "store_id": ["S01"],
        })
        pred, fallback = model.predict(target, return_fallback_info=True)
        self.assertAlmostEqual(pred[0], 12.0)  # Store S01 median
        self.assertEqual(fallback.iloc[0], "store_fallback")

    def test_fallback_to_global_median(self):
        model = WeekdayMedianBaseline().fit(self.history)

        # Brand new store S99
        target = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15"]),
            "store_id": ["S99"],
        })
        pred, fallback = model.predict(target, return_fallback_info=True)
        self.assertAlmostEqual(pred[0], 20.0)  # Global median
        self.assertEqual(fallback.iloc[0], "global_fallback")

    def test_save_and_load_serialization(self):
        model = WeekdayMedianBaseline().fit(self.history)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "baseline.json"
            model.save(tmp_path)
            self.assertTrue(tmp_path.exists())

            loaded_model = WeekdayMedianBaseline.load(tmp_path)
            self.assertTrue(loaded_model.is_fitted)
            self.assertEqual(loaded_model.global_median, model.global_median)
            self.assertEqual(loaded_model.store_medians, model.store_medians)
            self.assertEqual(loaded_model.store_weekday_medians, model.store_weekday_medians)

            target = pd.DataFrame({
                "date": pd.to_datetime(["2024-01-15"]),
                "store_id": ["S01"],
            })
            self.assertAlmostEqual(model.predict(target)[0], loaded_model.predict(target)[0])


if __name__ == "__main__":
    unittest.main()
