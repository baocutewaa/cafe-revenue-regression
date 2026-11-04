"""
Unit Tests cho BƯỚC 1: COLLECT & PREPARE
Tệp: tests/test_step1.py
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from src.step1_collect_prepare import (
    rmsle,
    load_raw_data,
    build_data_dictionary,
    check_merge_keys,
    prepare_holidays_table,
    safe_merge_tables,
    run_step1
)

class TestStep1CollectPrepare(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tables = load_raw_data("data")
        
    def test_rmsle_exact(self):
        # Kiểm tra RMSLE = 0 khi y_true == y_pred
        y_true = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(rmsle(y_true, y_true), 0.0, places=6)
        
        # Kiểm tra tính toán RMSLE với giá trị cụ thể
        y_pred = np.array([11.0, 22.0, 33.0])
        score = rmsle(y_true, y_pred)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.2)
        
    def test_load_raw_data_shapes(self):
        # Kiểm tra kích thước chính xác của 5 bảng
        self.assertEqual(self.tables["train"].shape, (26632, 4))
        self.assertEqual(self.tables["test"].shape, (3869, 4))
        self.assertEqual(self.tables["stores"].shape, (45, 8))
        self.assertEqual(self.tables["weather"].shape, (1593, 4))
        self.assertEqual(self.tables["holidays"].shape, (55, 3))
        
    def test_build_data_dictionary(self):
        data_dict = build_data_dictionary(self.tables)
        self.assertEqual(len(data_dict), 5)
        self.assertIn("Table Name", data_dict.columns)
        self.assertIn("Primary / Merge Key", data_dict.columns)
        
    def test_check_merge_keys(self):
        results = check_merge_keys(self.tables)
        self.assertTrue(results["train_stores_in_catalog"])
        self.assertTrue(results["test_stores_in_catalog"])
        self.assertEqual(results["missing_in_test"], ["S02"])
        self.assertTrue(results["cities_match"])
        self.assertTrue(results["has_holiday_key_duplicate"])
        self.assertEqual(set(results["duplicated_holiday_dates"].keys()), {"2024-02-14", "2026-02-14"})
        self.assertEqual(results["test_naive_exceeded_rows"], 44)
        
    def test_prepare_holidays_table(self):
        holidays_agg = prepare_holidays_table(self.tables["holidays"])
        self.assertEqual(holidays_agg.shape, (53, 4))
        self.assertEqual(holidays_agg["date"].duplicated().sum(), 0)
        self.assertTrue(set(["is_national_holiday", "is_event_holiday", "holiday_name_clean"]).issubset(holidays_agg.columns))
        
        # Kiểm tra ngày 2024-02-14 và 2026-02-14 có cả 2 cờ = 1
        d1 = pd.Timestamp("2024-02-14")
        row1 = holidays_agg[holidays_agg["date"] == d1].iloc[0]
        self.assertEqual(row1["is_national_holiday"], 1)
        self.assertEqual(row1["is_event_holiday"], 1)
        self.assertIn("Tet holiday", row1["holiday_name_clean"])
        self.assertIn("Valentine's Day", row1["holiday_name_clean"])
        
    def test_safe_merge_tables_preservation(self):
        holidays_agg = prepare_holidays_table(self.tables["holidays"])
        train_merged = safe_merge_tables(self.tables["train"], self.tables["stores"], self.tables["weather"], holidays_agg)
        test_merged = safe_merge_tables(self.tables["test"], self.tables["stores"], self.tables["weather"], holidays_agg)
        
        # Bảo toàn số dòng tuyệt đối
        self.assertEqual(len(train_merged), 26632)
        self.assertEqual(len(test_merged), 3869)
        self.assertEqual(train_merged.shape[1], 16)
        self.assertEqual(test_merged.shape[1], 16)
        
        # Kiểm tra không có NaN ở các cột nhị phân lễ tết
        self.assertEqual(train_merged["is_national_holiday"].isnull().sum(), 0)
        self.assertEqual(train_merged["is_event_holiday"].isnull().sum(), 0)
        self.assertEqual(test_merged["is_national_holiday"].isnull().sum(), 0)
        self.assertEqual(test_merged["is_event_holiday"].isnull().sum(), 0)
        
    def test_run_step1_end_to_end(self):
        train_merged, test_merged, data_dict, key_check = run_step1("data", verbose=False)
        self.assertEqual(train_merged.shape, (26632, 16))
        self.assertEqual(test_merged.shape, (3869, 16))
        self.assertEqual(len(data_dict), 5)
        self.assertEqual(key_check["test_naive_exceeded_rows"], 44)

if __name__ == "__main__":
    unittest.main()
