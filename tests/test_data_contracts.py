"""Automated regression tests and data quality contracts for Phase 0 (Section 5.11).

Validates:
- Exact duplicate removal count (132 rows)
- S12 unit error correction in March 2024 (30 rows, revenue / 1000)
- Outlier spike detection (50 high spikes, 25 low spikes, 5 review flags)
- Store metadata standardization, district mapping, and area_m2 imputation (S19, S24)
- Weather Fahrenheit conversion (22 rows > 45°C) and full grid completion (1,642 rows)
- Daily holiday aggregation and Tet indicators
- Join row preservation and key uniqueness
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    TRAIN_RAW_FILE,
    STORES_RAW_FILE,
    WEATHER_RAW_FILE,
    HOLIDAYS_RAW_FILE,
    CLEAN_DATA_DIR,
    DATE_COL,
    STORE_COL,
    TARGET_COL,
)
from src.data.clean import (
    clean_train,
    clean_stores,
    clean_weather,
    clean_holidays,
    join_cleaned_data,
)


@pytest.fixture(scope="module")
def raw_datasets():
    train = pd.read_csv(TRAIN_RAW_FILE, parse_dates=[DATE_COL])
    stores = pd.read_csv(STORES_RAW_FILE)
    weather = pd.read_csv(WEATHER_RAW_FILE, parse_dates=[DATE_COL])
    holidays = pd.read_csv(HOLIDAYS_RAW_FILE, parse_dates=[DATE_COL])
    return train, stores, weather, holidays


class TestDataContracts:
    """Test suite ensuring strict compliance with Data Quality Register (DQ-01 to DQ-15)."""

    def test_clean_train_contracts(self, raw_datasets):
        train_raw, _, _, _ = raw_datasets
        train_clean, log = clean_train(train_raw, ref_end=None)

        # DQ-01: Exactly 132 duplicate rows removed
        assert log["exact_dups_removed"] == 132, f"Expected 132 dups removed, got {log['exact_dups_removed']}"
        assert len(train_clean) == 26500, f"Expected 26,500 clean rows, got {len(train_clean)}"
        assert not train_clean.duplicated([DATE_COL, STORE_COL]).any(), "Duplicate (date, store_id) keys exist!"

        # DQ-02: Exactly 30 rows of S12 in March 2024 corrected
        assert log["unit_fix_rows"] == 30, f"Expected 30 unit fix rows, got {log['unit_fix_rows']}"
        s12_march = train_clean[
            (train_clean[STORE_COL] == "S12") & train_clean[DATE_COL].between("2024-03-01", "2024-03-31")
        ]
        assert len(s12_march) == 30
        assert 15.0 < s12_march[TARGET_COL].median() < 40.0, f"S12 median out of range: {s12_march[TARGET_COL].median()}"

        # DQ-03 & DQ-03b: Spikes detection
        assert log["spike_rows"] == 50, f"Expected 50 high spikes, got {log['spike_rows']}"
        assert log["spike_low_rows"] == 25, f"Expected 25 low spikes, got {log['spike_low_rows']}"
        assert log["review_rows"] == 5, f"Expected 5 review rows, got {log['review_rows']}"

        # Target positivity and non-spike bound
        assert (train_clean[TARGET_COL] > 0).all(), "Found non-positive revenue!"
        non_spike_rev = train_clean.loc[~train_clean["fix_flag"].str.startswith("spike"), TARGET_COL]
        assert non_spike_rev.max() < 500.0, f"Non-spike max revenue too high: {non_spike_rev.max()}"

    def test_clean_train_ref_end_prevent_leakage(self, raw_datasets):
        train_raw, _, _, _ = raw_datasets
        # When ref_end is set to '2025-10-01', median is computed only from train history
        train_clean, log = clean_train(train_raw, ref_end="2025-10-01")
        assert log["spike_rows"] == 50
        assert log["spike_low_rows"] in (24, 25)

    def test_stores_contracts(self, raw_datasets):
        _, stores_raw, _, _ = raw_datasets
        stores_clean, log = clean_stores(stores_raw)

        # Stores count
        assert len(stores_clean) == 45
        assert stores_clean[STORE_COL].nunique() == 45

        # Store type standardization (5 unique lowercase types)
        expected_types = {"street", "office", "mall", "university", "residential"}
        assert set(stores_clean["store_type"].unique()) == expected_types

        # District standardization (no 'Quan' or 'Q.')
        districts = stores_clean["district"].tolist()
        assert not any("Quan" in d or "Q." in d for d in districts)
        assert "District 10" in districts
        assert "District 3" in districts
        assert "District 7" in districts

        # Missing area_m2 imputation
        assert log["area_m2_missing_count"] == 2
        assert set(log["missing_stores"]) == {"S19", "S24"}
        assert stores_clean["area_m2"].isna().sum() == 0
        assert (stores_clean["area_m2"] > 0).all()
        assert (stores_clean["seats"] > 0).all()

        # open_date parsing
        assert not stores_clean["open_date"].isna().any()

    def test_weather_contracts(self, raw_datasets):
        _, _, weather_raw, _ = raw_datasets
        weather_clean, log = clean_weather(weather_raw, ref_end=None)

        # Grid completeness: 2024-01-01 to 2026-03-31 = 821 days * 2 cities = 1642 rows
        assert len(weather_clean) == 1642
        assert log["missing_dates_count"] == 49
        assert log["temp_fahrenheit_fixed_count"] == 22

        # Physical limits
        assert weather_clean["temp_c"].between(5.0, 42.0).all()
        assert (weather_clean["rain_mm"] >= 0.0).all()
        assert weather_clean.groupby("city")[DATE_COL].apply(lambda d: d.is_unique).all()

        # Derived flags
        assert set(weather_clean["rain_flag"].unique()).issubset({0, 1})
        assert set(weather_clean["heavy_rain"].unique()).issubset({0, 1})
        assert (weather_clean["rain_log"] >= 0.0).all()

    def test_holidays_contracts(self, raw_datasets):
        _, _, _, holidays_raw = raw_datasets
        holidays_clean, log = clean_holidays(holidays_raw)

        # Aggregated 1 row per unique date
        assert len(holidays_clean) == 53
        assert holidays_clean[DATE_COL].is_unique

        # Tet indicators
        tet_day1 = holidays_clean[holidays_clean["is_tet_day1"] == 1]
        assert len(tet_day1) == 3  # 2024-02-10, 2025-01-29, 2026-02-17
        expected_tet_dates = {
            pd.Timestamp("2024-02-10"),
            pd.Timestamp("2025-01-29"),
            pd.Timestamp("2026-02-17"),
        }
        assert set(tet_day1[DATE_COL].tolist()) == expected_tet_dates

        tet_period = holidays_clean[holidays_clean["is_tet_period"] == 1]
        assert len(tet_period) >= 20

    def test_join_contracts(self, raw_datasets):
        train_raw, stores_raw, weather_raw, holidays_raw = raw_datasets
        train_clean, _ = clean_train(train_raw)
        stores_clean, _ = clean_stores(stores_raw)
        weather_clean, _ = clean_weather(weather_raw)
        holidays_clean, _ = clean_holidays(holidays_raw)

        merged = join_cleaned_data(train_clean, stores_clean, weather_clean, holidays_clean)

        # Row preservation
        assert len(merged) == len(train_clean) == 26500

        # Uniqueness of primary key
        assert not merged.duplicated([DATE_COL, STORE_COL]).any()

        # No nulls in key modeling features
        critical_cols = [
            DATE_COL, STORE_COL, TARGET_COL, "city", "store_type", "district",
            "area_m2", "seats", "temp_c", "rain_mm", "is_holiday", "is_promo"
        ]
        for col in critical_cols:
            assert merged[col].isna().sum() == 0, f"Found {merged[col].isna().sum()} nulls in {col}"
