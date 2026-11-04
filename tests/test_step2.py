"""
Unit Tests cho BƯỚC 2: EXPLORE (EDA & PHÁT HIỆN LỖI DỮ LIỆU BẨN)
Tệp: tests/test_step2.py
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.step1_collect_prepare import load_raw_data, run_step1
from src.step2_explore import (
    detect_duplicates,
    detect_s12_1000x_anomaly,
    detect_decimal_errors,
    detect_store_type_inconsistency,
    detect_district_inconsistency,
    detect_open_date_formats,
    detect_stores_missing_area,
    detect_fahrenheit_outliers,
    detect_weather_missing_dates_and_rain_nans,
    build_ten_anomalies_table,
    analyze_business_patterns,
    run_step2
)

class TestStep2Explore(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tables = load_raw_data("data")
        cls.train_merged, cls.test_merged, _, _ = run_step1("data", verbose=False)
        
    def test_detect_duplicates(self):
        dup_count, dup_samples = detect_duplicates(self.tables["train"])
        self.assertEqual(dup_count, 132)
        self.assertEqual(len(dup_samples), 264)  # 132 pairs = 264 rows
        
    def test_detect_s12_1000x(self):
        s12_df = detect_s12_1000x_anomaly(self.tables["train"])
        self.assertEqual(len(s12_df), 31)
        self.assertTrue((s12_df["revenue"] > 1000).all())
        self.assertEqual(s12_df["store_id"].unique().tolist(), ["S12"])
        
    def test_detect_decimal_errors(self):
        df_10x, df_01x = detect_decimal_errors(self.tables["train"])
        self.assertGreaterEqual(len(df_10x), 50)
        self.assertGreaterEqual(len(df_01x), 20)
        self.assertTrue((df_10x["sw_ratio"] > 3.8).all())
        self.assertTrue((df_01x["sw_ratio"] < 0.16).all())
        
    def test_detect_store_type_inconsistency(self):
        st_counts = detect_store_type_inconsistency(self.tables["stores"])
        self.assertEqual(len(st_counts), 9)  # 9 variations due to capitalization
        self.assertIn("Street", st_counts.index)
        self.assertIn("street", st_counts.index)
        
    def test_detect_district_inconsistency(self):
        dist_info = detect_district_inconsistency(self.tables["stores"])
        self.assertIn("District 10", dist_info["hcm_districts"])
        self.assertIn("Q.10", dist_info["hcm_districts"])
        self.assertIn("Quan 3", dist_info["hcm_districts"])
        self.assertIn("Quan 7", dist_info["hcm_districts"])
        
    def test_detect_open_date_formats(self):
        od_info = detect_open_date_formats(self.tables["stores"])
        self.assertEqual(od_info["iso_count"], 30)
        self.assertEqual(od_info["vn_count"], 15)
        self.assertEqual(od_info["iso_count"] + od_info["vn_count"], 45)
        
    def test_detect_stores_missing_area(self):
        missing_area = detect_stores_missing_area(self.tables["stores"])
        self.assertEqual(len(missing_area), 2)
        self.assertEqual(set(missing_area["store_id"]), {"S19", "S24"})
        
    def test_detect_fahrenheit_outliers(self):
        f_df = detect_fahrenheit_outliers(self.tables["weather"])
        self.assertEqual(len(f_df), 22)
        self.assertEqual(f_df["city"].unique().tolist(), ["Ho Chi Minh"])
        self.assertGreater(f_df["temp_c"].min(), 45.0)
        
    def test_detect_weather_missing_dates(self):
        w_info = detect_weather_missing_dates_and_rain_nans(self.tables["weather"])
        self.assertEqual(w_info["hanoi_missing_days"], 21)
        self.assertEqual(w_info["hcm_missing_days"], 28)
        self.assertEqual(w_info["total_missing_city_days"], 49)
        self.assertEqual(w_info["rain_mm_nans"], 16)
        
    def test_build_ten_anomalies_table(self):
        anomalies_df = build_ten_anomalies_table(self.tables)
        self.assertEqual(len(anomalies_df), 10)
        self.assertEqual(list(anomalies_df["STT"]), list(range(1, 11)))
        
    def test_analyze_business_patterns(self):
        patterns = analyze_business_patterns(self.train_merged)
        self.assertIn("dow_matrix", patterns)
        self.assertIn("weekend_change_pct", patterns)
        self.assertIn("promo_uplift_pct", patterns)
        self.assertIn("uni_summer_drop_pct", patterns)
        self.assertIn("lifecycle_ratios", patterns)
        
        # Verify office drops significantly on weekend
        self.assertLess(patterns["weekend_change_pct"]["office"], -40.0)
        # Verify mall jumps on weekend
        self.assertGreater(patterns["weekend_change_pct"]["mall"], 50.0)
        # Verify promo uplift is ~30%
        self.assertGreater(patterns["promo_uplift_pct"], 25.0)
        # Verify uni summer drop is ~40%
        self.assertLess(patterns["uni_summer_drop_pct"], -35.0)

if __name__ == "__main__":
    unittest.main()
