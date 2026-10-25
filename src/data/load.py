"""Data loading, cleaning, deduplication, and split generation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.config import (
    DATE_COL,
    STORE_COL,
    TARGET_COL,
    PRIMARY_TRAIN_END,
    PRIMARY_VAL_END,
    TRAIN_RAW_FILE,
    STORES_RAW_FILE,
    WEATHER_RAW_FILE,
    HOLIDAYS_RAW_FILE,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
)

logger = logging.getLogger(__name__)


def load_raw_train(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw train.csv with datetime parsing."""
    path = file_path or TRAIN_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"Raw train file not found at: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df


def load_raw_stores(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw stores.csv."""
    path = file_path or STORES_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"Raw stores file not found at: {path}")
    df = pd.read_csv(path)
    return df


def load_raw_weather(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw weather.csv with datetime parsing."""
    path = file_path or WEATHER_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"Raw weather file not found at: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df


def load_raw_holidays(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw holidays.csv with datetime parsing."""
    path = file_path or HOLIDAYS_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"Raw holidays file not found at: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df


def deduplicate_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Audit and remove exact duplicate records from train data.

    Regression_Plan Section 2.2 & 2.3:
    - 26,632 raw rows contains 132 exact duplicate records.
    - After dropping duplicates, exactly 26,500 clean rows remain.
    - Verified that no key duplicate on (date, store_id) remains after exact dedup.

    Returns:
        Tuple of (clean_df, duplicate_count).
    """
    initial_count = len(df)
    exact_dups = int(df.duplicated().sum())

    df_clean = df.drop_duplicates().copy()
    cleaned_count = len(df_clean)

    # Check for remaining conflicting key duplicates
    remaining_key_dups = int(df_clean.duplicated(subset=[DATE_COL, STORE_COL]).sum())
    if remaining_key_dups > 0:
        raise ValueError(
            f"Found {remaining_key_dups} duplicate keys for ({DATE_COL}, {STORE_COL}) with conflicting values!"
        )

    logger.info(
        f"Deduplication complete: initial={initial_count}, exact_duplicates_removed={exact_dups}, clean={cleaned_count}"
    )
    return df_clean, exact_dups


def create_temporal_splits(
    df_clean: pd.DataFrame,
    out_dirs: Optional[List[Path]] = None
) -> Dict[str, pd.DataFrame]:
    """Create the 2-way time-based split matching Starter Notebook (Regression_Plan Section 2.3).

    Splits:
    - Train: date < 2025-10-01 (22,668 rows, 85.54%)
    - Val: date >= 2025-10-01 (3,832 rows, 14.46%, Q4/2025)

    Writes train.csv and val.csv to target directories without modifying raw data.
    """
    if out_dirs is None:
        out_dirs = [SPLITS_DIR]

    train_part = df_clean[df_clean[DATE_COL] < PRIMARY_TRAIN_END].sort_values([DATE_COL, STORE_COL])
    val_part = df_clean[df_clean[DATE_COL] >= PRIMARY_TRAIN_END].sort_values([DATE_COL, STORE_COL])

    splits = {
        "train": train_part,
        "val": val_part,
    }

    # Verify counts
    assert len(train_part) == 22668, f"Expected 22,668 train rows, got {len(train_part)}"
    assert len(val_part) == 3832, f"Expected 3,832 val rows, got {len(val_part)}"

    for target_dir in out_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Clean up legacy test.csv if present
        legacy_test = target_dir / "test.csv"
        if legacy_test.exists():
            legacy_test.unlink()
        for name, part_df in splits.items():
            out_file = target_dir / f"{name}.csv"
            part_df.to_csv(out_file, index=False)
            logger.info(f"Saved {name}.csv to {out_file} ({len(part_df)} rows)")

    return splits


def load_splits(splits_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load train and val splits from CSV files (and test if available)."""
    s_dir = splits_dir or SPLITS_DIR
    splits = {}
    for name in ["train", "val"]:
        file_path = s_dir / f"{name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}. Please run create_temporal_splits first.")
        splits[name] = pd.read_csv(file_path, parse_dates=[DATE_COL])

    # Optional test file (if provided from Kaggle)
    test_path = s_dir / "test.csv"
    if test_path.exists():
        splits["test"] = pd.read_csv(test_path, parse_dates=[DATE_COL])

    return splits
