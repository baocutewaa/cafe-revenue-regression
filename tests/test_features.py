"""Unit tests for feature engineering, cleaning, and joining pipelines."""

import unittest
import numpy as np
import pandas as pd

from src.data.join import clean_stores, clean_holidays, clean_weather, join_all_data
from src.features.calendar import add_calendar_features
from src.features.store import add_store_features
from src.features.weather_holiday import add_promo_weather_holiday_features
from src.features.builder import build_all_features, get_feature_columns


class TestFeaturesAndJoin(unittest.TestCase):

    def setUp(self):
        # Sample stores with both date formats, missing area, and mixed store_type
        self.stores_raw = pd.DataFrame({
            "store_id": ["S01", "S02", "S03"],
            "city": ["Ha Noi", " Ha Noi ", "Ho Chi Minh"],
            "district": ["Ba Dinh", "Dong Da", "District 1"],
            "store_type": ["Office", "street ", "Mall"],
            "area_m2": [120.0, np.nan, 85.0],
            "seats": [40, 30, 25],
            "has_parking": [1, 0, 1],
            "open_date": ["2020-05-15", "25/12/2021", "2023-01-10"],
        })

        # Sample holidays with duplicate dates (Tet + Valentine)
        self.holidays_raw = pd.DataFrame({
            "date": ["2024-02-14", "2024-02-14", "2024-04-30"],
            "holiday_name": ["Tet holiday", "Valentine's Day", "Reunification Day"],
            "type": ["national_holiday", "event", "national_holiday"],
        })

        # Sample weather with Fahrenheit outlier and missing rain
        self.weather_raw = pd.DataFrame({
            "date": ["2024-02-14", "2024-02-14", "2025-06-15"],
            "city": ["Ha Noi", "Ho Chi Minh", "Ho Chi Minh"],
            "temp_c": [18.0, 28.5, 86.0],  # 86.0 Fahrenheit
            "rain_mm": [0.0, np.nan, 15.0],
        })

        # Sample transactions
        self.transactions = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-14", "2024-02-14", "2024-02-15"]),
            "store_id": ["S01", "S03", "S01"],
            "is_promo": [0, 1, 0],
            "revenue": [15.5, 25.0, 14.0],
        })

    def test_clean_stores_open_date_and_normalization(self):
        cleaned = clean_stores(self.stores_raw)
        self.assertEqual(len(cleaned), 3)
        self.assertListEqual(list(cleaned["store_type"]), ["office", "street", "mall"])
        self.assertEqual(cleaned["area_m2_missing"].sum(), 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["open_date_parsed"]))
        self.assertEqual(str(cleaned.loc[1, "open_date_parsed"].date()), "2021-12-25")

    def test_clean_holidays_aggregation_single_row_per_date(self):
        cleaned = clean_holidays(self.holidays_raw)
        self.assertEqual(len(cleaned), 2)  # 2024-02-14 and 2024-04-30
        row_feb14 = cleaned[cleaned["date"] == "2024-02-14"].iloc[0]
        self.assertEqual(row_feb14["holiday_count"], 2)
        self.assertEqual(row_feb14["is_national_holiday"], 1)
        self.assertEqual(row_feb14["is_event"], 1)
        self.assertIn("Valentine's Day", row_feb14["holiday_names"])
        self.assertIn("Tet holiday", row_feb14["holiday_names"])

    def test_clean_weather_fahrenheit_conversion(self):
        cleaned = clean_weather(self.weather_raw)
        # 86.0 Fahrenheit -> (86 - 32) * 5 / 9 = 30.0 Celsius
        hcm_june = cleaned[cleaned["date"] == "2025-06-15"].iloc[0]
        self.assertAlmostEqual(hcm_june["temp_c"], 30.0, places=2)
        self.assertEqual(cleaned["rain_missing"].sum(), 1)

    def test_join_all_data_row_count_preservation(self):
        initial_len = len(self.transactions)
        joined = join_all_data(
            self.transactions, self.stores_raw, self.weather_raw, self.holidays_raw
        )
        self.assertEqual(len(joined), initial_len)
        self.assertIn("store_type", joined.columns)
        self.assertIn("temp_c", joined.columns)
        self.assertIn("is_holiday", joined.columns)
        # Feb 14 was a holiday, Feb 15 was not
        self.assertEqual(joined.loc[joined["date"] == "2024-02-14", "is_holiday"].iloc[0], 1)
        self.assertEqual(joined.loc[joined["date"] == "2024-02-15", "is_holiday"].iloc[0], 0)

    def test_calendar_cyclical_trigonometric_features(self):
        df_cal = add_calendar_features(self.transactions)
        self.assertIn("sin_dow", df_cal.columns)
        self.assertIn("cos_dow", df_cal.columns)
        self.assertIn("sin_month", df_cal.columns)
        self.assertIn("cos_month", df_cal.columns)
        # sin^2 + cos^2 == 1
        dow_identity = df_cal["sin_dow"] ** 2 + df_cal["cos_dow"] ** 2
        np.testing.assert_allclose(dow_identity.values, 1.0, atol=1e-6)

    def test_feature_builder_pipeline(self):
        joined = join_all_data(
            self.transactions, self.stores_raw, self.weather_raw, self.holidays_raw
        )
        full_df = build_all_features(joined)
        for tier in ["E1_calendar_store", "E2_promo", "E3_weather_holiday"]:
            cats, nums, bins = get_feature_columns(tier)
            for c in cats + nums + bins:
                self.assertIn(c, full_df.columns, f"Missing feature {c} in tier {tier}")


if __name__ == "__main__":
    unittest.main()
