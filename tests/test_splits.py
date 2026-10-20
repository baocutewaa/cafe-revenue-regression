"""Unit tests for deduplication and temporal splits contracts."""

import unittest
import pandas as pd
from src.config import PRIMARY_TRAIN_END, PRIMARY_VAL_END, PRIMARY_TEST_END
from src.data.load import load_raw_train, deduplicate_train, load_splits


class TestDataSplits(unittest.TestCase):

    def test_raw_train_deduplication(self):
        raw_df = load_raw_train()
        self.assertEqual(len(raw_df), 26632)

        clean_df, dup_count = deduplicate_train(raw_df)
        self.assertEqual(dup_count, 132)
        self.assertEqual(len(clean_df), 26500)

        # Ensure no key duplicates on (date, store_id)
        self.assertFalse(clean_df.duplicated(subset=["date", "store_id"]).any())

    def test_splits_row_counts_and_dates(self):
        splits = load_splits()

        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]

        # Exact row counts from Regression_Plan Section 2.3
        self.assertEqual(len(train_df), 18982)
        self.assertEqual(len(val_df), 3686)
        self.assertEqual(len(test_df), 3832)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 26500)

        # Date range validation
        self.assertTrue((train_df["date"] < PRIMARY_TRAIN_END).all())

        self.assertTrue((val_df["date"] >= PRIMARY_TRAIN_END).all())
        self.assertTrue((val_df["date"] < PRIMARY_VAL_END).all())

        self.assertTrue((test_df["date"] >= PRIMARY_VAL_END).all())
        self.assertTrue((test_df["date"] < PRIMARY_TEST_END).all())

        # Zero temporal leakage / non-overlapping dates
        max_train_date = train_df["date"].max()
        min_val_date = val_df["date"].min()
        max_val_date = val_df["date"].max()
        min_test_date = test_df["date"].min()

        self.assertLess(max_train_date, min_val_date)
        self.assertLess(max_val_date, min_test_date)


if __name__ == "__main__":
    unittest.main()
