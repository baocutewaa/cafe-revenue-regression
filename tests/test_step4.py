"""
Unit Tests cho BƯỚC 4: TIME-BASED VALIDATION SPLIT
Tệp: tests/test_step4.py
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.step3_preprocess import run_step3
from src.step4_split import (
    get_starter_holdout_split,
    get_expanding_window_folds,
    TimeSeriesExpandingCV,
    compute_weekday_median_baseline,
    evaluate_baseline_across_splits,
    run_step4
)

class TestStep4Split(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.train_final, cls.test_final, _, _, _, _ = run_step3("data", verbose=False)
        cls.train_raw = pd.read_csv("data/train.csv", parse_dates=["date"])
        
    def test_starter_holdout_split(self):
        fit_df, val_df = get_starter_holdout_split(self.train_final, "2025-10-01")
        self.assertEqual(len(fit_df) + len(val_df), 26500)
        self.assertEqual(len(fit_df), 22668)
        self.assertEqual(len(val_df), 3832)
        
        # Kiểm tra không rò rỉ thời gian
        self.assertLess(fit_df["date"].max(), val_df["date"].min())
        self.assertEqual(val_df["date"].min(), pd.Timestamp("2025-10-01"))
        self.assertEqual(val_df["date"].max(), pd.Timestamp("2025-12-31"))
        
    def test_expanding_window_folds_config(self):
        folds = get_expanding_window_folds()
        self.assertEqual(len(folds), 4)
        
        # Fold 1 phải có validation chứa Tết 2025
        f1 = folds[0]
        self.assertEqual(f1["val_start"], "2025-01-01")
        self.assertEqual(f1["val_end"], "2025-03-31")
        
        # Kiểm tra tính mở rộng của cửa sổ train
        train_ends = [pd.Timestamp(f["train_end"]) for f in folds]
        for i in range(len(train_ends) - 1):
            self.assertLess(train_ends[i], train_ends[i+1])
            
    def test_time_series_expanding_cv_split(self):
        cv = TimeSeriesExpandingCV()
        self.assertEqual(cv.get_n_splits(), 4)
        
        splits = list(cv.split(self.train_final))
        self.assertEqual(len(splits), 4)
        
        prev_train_len = 0
        for tr_idx, val_idx in splits:
            self.assertGreater(len(tr_idx), prev_train_len)
            self.assertGreater(len(val_idx), 0)
            
            # Kiểm tra không overlap giữa train và val index
            overlap = set(tr_idx).intersection(set(val_idx))
            self.assertEqual(len(overlap), 0)
            
            # Kiểm tra không rò rỉ thời gian
            tr_dates = self.train_final.iloc[tr_idx]["date"]
            val_dates = self.train_final.iloc[val_idx]["date"]
            self.assertLess(tr_dates.max(), val_dates.min())
            
            prev_train_len = len(tr_idx)
            
    def test_weekday_median_baseline_raw_val(self):
        fit_raw = self.train_raw[self.train_raw["date"] < "2025-10-01"]
        val_raw = self.train_raw[self.train_raw["date"] >= "2025-10-01"]
        pred = compute_weekday_median_baseline(fit_raw, val_raw)
        
        from src.step1_collect_prepare import rmsle
        score = rmsle(val_raw["revenue"].values, pred)
        # Khớp với starter notebook baseline = 0.2387
        self.assertAlmostEqual(score, 0.2387, places=4)
        
    def test_evaluate_baseline_across_splits(self):
        eval_df = evaluate_baseline_across_splits(self.train_raw, self.train_final)
        self.assertEqual(len(eval_df), 6)  # Raw Val + Clean Val + 4 Folds
        
        # Fold 1 RMSLE phải cao hơn Clean Val vì Fold 1 chứa Tết Nguyên Đán
        f1_row = eval_df[eval_df["Tập đánh giá"] == "Fold 1 Validation"].iloc[0]
        cval_row = eval_df[eval_df["Tập đánh giá"] == "Clean Val (Đã sửa 6 lỗi dấu phẩy)"].iloc[0]
        self.assertGreater(f1_row["Baseline RMSLE"], cval_row["Baseline RMSLE"])
        self.assertEqual(f1_row["Có chứa Tết Nguyên Đán?"], "CÓ (Tết Ất Tỵ 2025)")
        
    def test_run_step4_end_to_end(self):
        fit_clean, val_clean, cv_splitter, eval_df = run_step4("data", verbose=False)
        self.assertEqual(len(fit_clean), 22668)
        self.assertEqual(len(val_clean), 3832)
        self.assertEqual(cv_splitter.get_n_splits(), 4)
        self.assertEqual(len(eval_df), 6)

if __name__ == "__main__":
    unittest.main()
