"""
Unit Tests cho BƯỚC 3: PREPROCESS & FEATURE ENGINEERING
Tệp: tests/test_step3.py
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.step1_collect_prepare import load_raw_data
from src.step3_preprocess import (
    clean_stores,
    clean_weather,
    clean_holidays,
    clean_train_outliers,
    compute_ramp_up_factor,
    build_features,
    DeSeasonalizedStoreStats,
    run_step3
)

class TestStep3Preprocess(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tables = load_raw_data("data")
        cls.stores_clean = clean_stores(cls.tables["stores"])
        cls.weather_clean = clean_weather(cls.tables["weather"])
        cls.holidays_clean = clean_holidays(cls.tables["holidays"])
        
    def test_clean_stores(self):
        df = self.stores_clean
        self.assertEqual(len(df), 45)
        # 1. 0 NaNs
        self.assertEqual(df.isnull().sum().sum(), 0)
        # 2. store_type exactly 5 categories
        self.assertEqual(set(df["store_type"]), {"street", "office", "mall", "university", "residential"})
        # 3. district cleaned
        self.assertNotIn("Q.10", df["district"].values)
        self.assertIn("District 10", df["district"].values)
        # 4. open_date_dt is datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["open_date_dt"]))
        # 5. S19 and S24 imputed
        s19_area = df.loc[df["store_id"] == "S19", "area_m2"].values[0]
        s24_area = df.loc[df["store_id"] == "S24", "area_m2"].values[0]
        self.assertGreater(s19_area, 100.0)
        self.assertGreater(s24_area, 100.0)
        
    def test_clean_weather(self):
        df = self.weather_clean
        # 821 days * 2 cities = 1642 rows
        self.assertEqual(len(df), 1642)
        self.assertEqual(df.isnull().sum().sum(), 0)
        # Max temp should be normal celsius (< 45)
        self.assertLess(df["temp_c"].max(), 45.0)
        self.assertGreater(df["temp_c"].min(), 5.0)
        # rain_mm should have no negatives and no NaNs
        self.assertGreaterEqual(df["rain_mm"].min(), 0.0)
        
    def test_clean_holidays(self):
        df = self.holidays_clean
        self.assertEqual(len(df), 53)
        self.assertEqual(df["date"].duplicated().sum(), 0)
        self.assertEqual(df.isnull().sum().sum(), 0)
        
    def test_clean_train_outliers(self):
        train_clean, stats = clean_train_outliers(self.tables["train"], self.stores_clean)
        self.assertEqual(len(train_clean), 26500)
        self.assertEqual(stats["duplicates_dropped"], 132)
        self.assertEqual(stats["fixed_1000x_count"], 30)
        self.assertIn(stats["fixed_10x_count"], [52, 53])
        self.assertIn(stats["fixed_01x_count"], [25, 26])
        self.assertLess(train_clean["revenue"].max(), 100.0)
        self.assertGreater(train_clean["revenue"].min(), 1.0)
        
    def test_compute_ramp_up_factor(self):
        days = pd.Series([0, 1, 2, 5, 10, 20, 45, 75])
        factors = compute_ramp_up_factor(days)
        self.assertEqual(factors.iloc[0], 1.70)
        self.assertEqual(factors.iloc[2], 1.70)
        self.assertEqual(factors.iloc[3], 0.61)
        self.assertEqual(factors.iloc[4], 0.67)
        self.assertEqual(factors.iloc[5], 0.73)
        self.assertEqual(factors.iloc[6], 0.86)
        self.assertEqual(factors.iloc[7], 1.00)
        
    def test_build_features(self):
        # Merge sample
        train_clean, _ = clean_train_outliers(self.tables["train"], self.stores_clean)
        m = train_clean.head(100).merge(self.stores_clean, on="store_id", how="left")
        m = m.merge(self.weather_clean, on=["city", "date"], how="left")
        m = m.merge(self.holidays_clean, on="date", how="left")
        
        feats = build_features(m)
        expected_cols = [
            "dow", "is_weekend", "day", "month", "weekofyear", "quarter",
            "sin_dow", "cos_dow", "sin_doy", "cos_doy",
            "days_from_tet", "is_tet_core", "is_pre_tet_shopping", "is_ong_tao", "is_post_tet_recovery",
            "is_uni_tet_break", "is_uni_summer_break", "is_national_holiday", "is_event_holiday",
            "days_since_open", "log_days_since_open", "is_grand_opening", "is_ramp_up_phase", "ramp_up_factor",
            "temp_c", "rain_mm", "log_rain", "is_heavy_rain", "is_extreme_rain",
            "rain_x_outdoor", "rain_x_mall", "promo_x_weekend"
        ]
        for col in expected_cols:
            self.assertIn(col, feats.columns)
            
    def test_de_seasonalized_store_stats_leakage_free(self):
        train_clean, _ = clean_train_outliers(self.tables["train"], self.stores_clean)
        m = train_clean.merge(self.stores_clean, on="store_id", how="left")
        m = m.merge(self.weather_clean, on=["city", "date"], how="left")
        m = m.merge(self.holidays_clean, on="date", how="left")
        feats = build_features(m)
        
        encoder = DeSeasonalizedStoreStats(shrinkage_m=10.0)
        encoder.fit(feats, feats["revenue"].values)
        res = encoder.transform(feats)
        
        self.assertIn("store_base_log_rev", res.columns)
        self.assertIn("store_dow_log_rev", res.columns)
        self.assertIn("store_dow_ratio", res.columns)
        self.assertIn("type_city_dow_log_rev", res.columns)
        self.assertEqual(res["store_base_log_rev"].isnull().sum(), 0)
        self.assertEqual(res["store_dow_log_rev"].isnull().sum(), 0)
        
    def test_run_step3_end_to_end(self):
        train_final, test_final, stores_clean, weather_clean, holidays_clean, stats = run_step3("data", verbose=False)
        self.assertEqual(len(train_final), 26500)
        self.assertEqual(len(test_final), 3869)
        self.assertEqual(train_final.isnull().sum().sum(), 0)
        self.assertEqual(test_final.isnull().sum().sum(), 0)
        self.assertEqual(train_final.shape[1], test_final.shape[1])

if __name__ == "__main__":
    unittest.main()
