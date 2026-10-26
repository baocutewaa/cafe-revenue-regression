"""Configuration settings and paths for the cafe-revenue-regression project."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEAN_DATA_DIR = DATA_DIR / "clean"
SPLITS_DIR = PROJECT_ROOT / "regression_splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
for p in [PROCESSED_DATA_DIR, CLEAN_DATA_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Raw Data Files
TRAIN_RAW_FILE = RAW_DATA_DIR / "train.csv"
STORES_RAW_FILE = RAW_DATA_DIR / "stores.csv"
WEATHER_RAW_FILE = RAW_DATA_DIR / "weather.csv"
HOLIDAYS_RAW_FILE = RAW_DATA_DIR / "holidays.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"

# Clean Data Files (Giai đoạn 0)
TRAIN_CLEAN_FILE = CLEAN_DATA_DIR / "train_clean.parquet"
STORES_CLEAN_FILE = CLEAN_DATA_DIR / "stores_clean.parquet"
WEATHER_CLEAN_FILE = CLEAN_DATA_DIR / "weather_clean.parquet"
HOLIDAYS_DAILY_FILE = CLEAN_DATA_DIR / "holidays_daily.parquet"

# Column Definitions
DATE_COL = "date"
STORE_COL = "store_id"
TARGET_COL = "revenue"
PROMO_COL = "is_promo"

# Primary Temporal Split Cutoffs (Regression_Plan Section 2.3 - Starter Notebook Standard)
PRIMARY_TRAIN_END = "2025-10-01"       # Train: date < 2025-10-01 (22,668 clean rows, 21 months)
PRIMARY_VAL_END = "2026-01-01"         # Val: 2025-10-01 <= date < 2026-01-01 (3,832 clean rows, Q4/2025)

# Walk-Forward Evaluation Windows (Regression_Plan Section 7.2)
WALK_FORWARD_FOLDS = {
    "fold_a": {
        "name": "Fold A (Q1/2025)",
        "train_end": "2025-01-01",
        "val_start": "2025-01-01",
        "val_end": "2025-04-01",
        "purpose": "Đo dự báo quý 1 (tương đồng mùa vụ Q1/2026 của test)",
    },
    "fold_b": {
        "name": "Fold B (Q2/2025)",
        "train_end": "2025-04-01",
        "val_start": "2025-04-01",
        "val_end": "2025-07-01",
        "purpose": "Kiểm tra độ ổn định giữa năm (Q2/2025)",
    },
    "fold_c": {
        "name": "Fold C (Q3/2025)",
        "train_end": "2025-07-01",
        "val_start": "2025-07-01",
        "val_end": "2025-10-01",
        "purpose": "Đo độ biến động mùa hè/mưa bão (Q3/2025)",
    },
    "primary_val": {
        "name": "Holdout chính (Q4/2025)",
        "train_end": "2025-10-01",
        "val_start": "2025-10-01",
        "val_end": "2026-01-01",
        "purpose": "Chọn pipeline trên validation đồng bộ Starter Notebook (Q4/2025)",
    },
}

