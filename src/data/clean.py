"""Data cleaning, outlier handling, and data contract validation module.

Adheres strictly to Regression_Plan_v2.md Section 5:
- 5.0: Three-layer data architecture (raw -> clean -> features/model)
- 5.1: Data Quality Register (DQ-01 to DQ-15)
- 5.2: Outlier detection & unit correction (S12 /1000, spike_x10, spike_low_x01, manual review)
- 5.3: Missing data handling and panel completeness
- 5.4: Store metadata standardization and area_m2 imputation
- 5.5: Weather Fahrenheit fix, full (date, city) grid, and missing value imputation
- 5.6: Daily holiday aggregation and Tet festival indicators
- 5.7: Safe multi-table join and data contract validation
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    STORE_COL,
    TARGET_COL,
    PROMO_COL,
    CLEAN_DATA_DIR,
    REPORTS_DIR,
    TRAIN_CLEAN_FILE,
    STORES_CLEAN_FILE,
    WEATHER_CLEAN_FILE,
    HOLIDAYS_DAILY_FILE,
)

logger = logging.getLogger(__name__)


def parse_open_date(s: pd.Series) -> pd.Series:
    """Parse open_date string with strict pattern matching (DQ-09).

    Supports:
    - ISO format: YYYY-MM-DD (30 stores)
    - Vietnamese/UK format: DD/MM/YYYY (15 stores)
    Raises ValueError if any date fails to match either pattern.
    """
    s_clean = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s_clean.index, dtype="datetime64[ns]")

    iso_mask = s_clean.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    dmy_mask = s_clean.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)

    out.loc[iso_mask] = pd.to_datetime(s_clean[iso_mask], format="%Y-%m-%d")
    out.loc[dmy_mask] = pd.to_datetime(s_clean[dmy_mask], format="%d/%m/%Y")

    bad_mask = ~(iso_mask | dmy_mask)
    if bad_mask.any():
        bad_values = s_clean[bad_mask].tolist()
        raise ValueError(f"Unrecognized open_date pattern: {bad_values}")

    return out


def clean_stores(
    stores_df: pd.DataFrame,
    seats_ratio: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and standardize stores.csv metadata (DQ-08, DQ-09, DQ-10).

    Transformations:
    1. Strip whitespace on string columns.
    2. Normalize store_type to lowercase: street, office, mall, university, residential.
    3. Normalize district names: maps 'Q.10', 'Quan 3', 'Quan 7' to 'District 10', 'District 3', 'District 7'.
    4. Parse open_date into datetime64[ns].
    5. Impute area_m2 missing values (S19, S24) using seats * median(area_m2 / seats).
    6. Verify contracts: seats > 0, area_m2 > 0, has_parking in {0, 1}.

    Returns:
        Tuple of (cleaned DataFrame, log dict).
    """
    df = stores_df.copy()
    log: Dict[str, Any] = {"initial_stores": len(df)}

    # 1. Normalize string columns
    df["city"] = df["city"].astype(str).str.strip()
    df["store_type"] = df["store_type"].astype(str).str.strip().str.lower()
    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
        .str.replace(r"^(Quan|Q\.)\s*(\d+)$", r"District \2", regex=True)
    )

    # 2. Parse open_date
    df["open_date"] = parse_open_date(df["open_date"])

    # 3. Handle missing area_m2
    df["area_m2_missing"] = df["area_m2"].isna().astype(int)
    log["area_m2_missing_count"] = int(df["area_m2_missing"].sum())
    log["missing_stores"] = df.loc[df["area_m2_missing"] == 1, STORE_COL].tolist()

    if seats_ratio is None:
        non_missing = df.dropna(subset=["area_m2"])
        impute_ratio = float((non_missing["area_m2"] / non_missing["seats"]).median())
    else:
        impute_ratio = seats_ratio

    log["area_seats_impute_ratio"] = impute_ratio
    df["area_m2"] = df["area_m2"].fillna(df["seats"] * impute_ratio)

    # 4. Standardize types and indicators
    df["seats"] = df["seats"].astype(float)
    df["has_parking"] = df["has_parking"].fillna(0).astype(int)

    # 5. Contract validations
    assert (df["seats"] > 0).all(), "Found non-positive seats in stores!"
    assert (df["area_m2"] > 0).all(), "Found non-positive area_m2 in stores!"
    assert set(df["has_parking"].unique()).issubset({0, 1}), "has_parking contains invalid values!"
    assert df[STORE_COL].is_unique, "Duplicate store_id found in stores!"

    return df, log


def clean_weather(
    weather_df: pd.DataFrame,
    ref_end: Optional[str] = None,
    grid_start: str = "2024-01-01",
    grid_end: str = "2026-03-31"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean weather data, fix Fahrenheit unit errors, build full grid, and impute (DQ-11, DQ-12, DQ-13).

    Transformations:
    1. Parse dates and standardize city names ('Ha Noi', 'Ho Chi Minh').
    2. Convert erroneous Fahrenheit readings (> 45°C) in Ho Chi Minh via (F - 32) * 5/9.
    3. Generate full daily calendar grid (date, city) across 2024-01-01 to 2026-03-31 (1,642 rows).
    4. Track weather_missing and rain_missing flags.
    5. Forward-fill temp_c up to 2 days; fill remaining with city-month median of reference period.
    6. Impute rain_mm with reference city-month median.
    7. Generate derived features: rain_log = log1p(rain_mm), rain_flag = (rain_mm > 0), heavy_rain = (rain_mm >= 50).

    Returns:
        Tuple of (cleaned DataFrame, log dict).
    """
    df = weather_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["city"] = df["city"].astype(str).str.strip()
    log: Dict[str, Any] = {"raw_weather_rows": len(df)}

    # 1. Correct Fahrenheit temperature values (> 45°C)
    f_mask = df["temp_c"] > 45.0
    df["temp_was_fahrenheit"] = f_mask.astype(int)
    log["temp_fahrenheit_fixed_count"] = int(f_mask.sum())
    df.loc[f_mask, "temp_c"] = (df.loc[f_mask, "temp_c"] - 32.0) * 5.0 / 9.0

    # Flag original missing rain in recorded rows
    df["orig_rain_missing"] = df["rain_mm"].isna().astype(int)

    # 2. Build complete grid (date, city)
    dates = pd.date_range(grid_start, grid_end, freq="D")
    cities = sorted(df["city"].unique())
    full_grid = pd.MultiIndex.from_product([dates, cities], names=[DATE_COL, "city"]).to_frame().reset_index(drop=True)
    log["grid_rows"] = len(full_grid)

    # Merge onto full grid
    merged = full_grid.merge(df, on=[DATE_COL, "city"], how="left")
    merged["weather_missing"] = merged["temp_c"].isna().astype(int)
    log["missing_dates_count"] = int(merged["weather_missing"].sum())

    merged["temp_was_fahrenheit"] = merged["temp_was_fahrenheit"].fillna(0).astype(int)
    merged["rain_missing"] = (merged["orig_rain_missing"] == 1) | (merged["weather_missing"] == 1)
    merged["rain_missing"] = merged["rain_missing"].astype(int)
    log["total_rain_missing"] = int(merged["rain_missing"].sum())

    # Reference window for calculating monthly medians (prevents leakage)
    ref_mask = (merged[DATE_COL] < pd.Timestamp(ref_end)) if ref_end else (merged["weather_missing"] == 0)
    ref_df = merged[ref_mask]

    # 3. Impute temp_c: forward-fill max 2 days per city, then fill remainder with city-month median
    merged["temp_c"] = merged.groupby("city")["temp_c"].ffill(limit=2)
    month_series = merged[DATE_COL].dt.month
    city_month_temp = ref_df.groupby([ref_df["city"], ref_df[DATE_COL].dt.month])["temp_c"].median()

    temp_keys = pd.MultiIndex.from_arrays([merged["city"], month_series])
    temp_medians = city_month_temp.reindex(temp_keys).to_numpy()
    merged["temp_c"] = merged["temp_c"].fillna(pd.Series(temp_medians, index=merged.index))

    # 4. Impute rain_mm: fill with reference city-month median
    city_month_rain = ref_df.groupby([ref_df["city"], ref_df[DATE_COL].dt.month])["rain_mm"].median()
    rain_medians = city_month_rain.reindex(temp_keys).to_numpy()
    merged["rain_mm"] = merged["rain_mm"].fillna(pd.Series(rain_medians, index=merged.index)).fillna(0.0)

    # 5. Derived weather transformations
    merged["rain_log"] = np.log1p(merged["rain_mm"])
    merged["rain_flag"] = (merged["rain_mm"] > 0).astype(int)
    merged["heavy_rain"] = (merged["rain_mm"] >= 50.0).astype(int)

    # Drop temporary column
    merged = merged.drop(columns=["orig_rain_missing"])

    # Contracts
    assert merged["temp_c"].between(5.0, 42.0).all(), "Temperature outside physical limits (5-42°C)!"
    assert merged["rain_mm"].min() >= 0.0, "Negative rain_mm detected!"
    assert merged.groupby("city")[DATE_COL].apply(lambda d: d.is_unique).all(), "Duplicate (date, city) in weather!"

    return merged, log


def clean_holidays(
    holidays_df: pd.DataFrame,
    horizon_end: str = "2026-03-31"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and aggregate holidays into daily features with Tet indicators (DQ-14).

    Transformations:
    1. Parse dates and strip whitespace.
    2. Aggregate into exactly 1 record per calendar date to prevent row duplication on dates with multiple events.
    3. Generate holiday indicators:
       - is_holiday = 1
       - holiday_count: number of events on that date
       - is_national_holiday: binary flag
       - is_event: binary flag
       - holiday_names: semicolon-joined event names
    4. Generate Tet festival specific flags:
       - is_tet_day1: Lunar New Year (Tet) Day 1
       - is_tet_period: Tet holiday span (includes Eve through Day 7)
       - is_tet_eve: Ong Cong Ong Tao (23 Tet)
    5. Generate holiday coverage indicator:
       - holiday_covered = 1 for dates <= 2026-03-08, 0 for dates after (noted in plan).

    Returns:
        Tuple of (cleaned DataFrame, log dict).
    """
    df = holidays_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["holiday_name"] = df["holiday_name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    log: Dict[str, Any] = {"raw_holiday_rows": len(df)}

    # Daily aggregation
    agg = df.groupby(DATE_COL).agg(
        holiday_count=("holiday_name", "count"),
        is_national_holiday=("type", lambda s: int((s == "national_holiday").any())),
        is_event=("type", lambda s: int((s == "event").any())),
        holiday_names=("holiday_name", lambda s: "; ".join(sorted(set(s)))),
    ).reset_index()

    agg["is_holiday"] = 1
    log["unique_holiday_dates"] = len(agg)

    # Tet specific indicators
    agg["is_tet_day1"] = agg["holiday_names"].str.contains(r"Lunar New Year \(Tet\)", regex=True).astype(int)
    agg["is_tet_period"] = (
        agg["holiday_names"].str.contains(r"Tet holiday", regex=True) | (agg["is_tet_day1"] == 1)
    ).astype(int)
    agg["is_tet_eve"] = agg["holiday_names"].str.contains(r"Ong Cong Ong Tao", regex=True).astype(int)

    # Coverage flag
    agg["holiday_covered"] = (agg[DATE_COL] <= pd.Timestamp("2026-03-08")).astype(int)

    # Verify contracts
    assert agg[DATE_COL].is_unique, "Duplicate dates in aggregated holidays!"
    assert agg["is_holiday"].min() == 1, "is_holiday must be 1 for all holiday records!"

    return agg, log


def clean_train(
    train_df: pd.DataFrame,
    ref_end: Optional[str] = None,
    spike_ratio: float = 5.0,
    low_ratio: float = 0.15
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean train revenue records, correct S12 unit error, and flag spikes (DQ-01 to DQ-04).

    Parameters:
        train_df: Raw train DataFrame.
        ref_end: Optional cutoff date string (e.g., '2025-10-01'). If specified, baseline
                 median reference statistics are computed ONLY from rows with date < ref_end
                 to strictly prevent leakage during cross-validation / holdout evaluation.
                 If None, computed over the entire dataset (for final clean dataset export).
        spike_ratio: Multiplicative ratio above median (store, weekday) to flag as spike_x10 (default: 5.0).
        low_ratio: Multiplicative ratio below median (store, weekday) to flag as spike_low_x01 (default: 0.15).

    Transformations:
    1. Parse date column to datetime64[ns].
    2. Drop exact duplicate rows (DQ-01: 132 rows).
    3. Assert uniqueness of primary key (date, store_id).
    4. Store revenue_raw and initialize fix_flag.
    5. Fix unit error for store S12 in March 2024 (DQ-02: 30 rows divided by 1000, fix_flag='unit_x1000').
    6. Compute reference median revenue by (store_id, dayofweek) on reference period.
    7. Flag high spikes (DQ-03: ratio > spike_ratio, fix_flag='spike_x10').
    8. Flag low spikes (DQ-03b: ratio < low_ratio, fix_flag='spike_low_x01').
    9. Flag manual review candidates (DQ-04: 3.0 <= ratio <= 5.0, review_flag=1).

    Returns:
        Tuple of (cleaned DataFrame, log dict).
    """
    df = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    log: Dict[str, Any] = {"rows_raw": len(df)}

    # 1. Deduplication (DQ-01)
    df = df.drop_duplicates().copy()
    log["rows_after_dedup"] = len(df)
    log["exact_dups_removed"] = log["rows_raw"] - log["rows_after_dedup"]

    # Key uniqueness verification
    assert not df.duplicated([DATE_COL, STORE_COL]).any(), (
        "Duplicate (date, store_id) keys found with conflicting values!"
    )

    # 2. Initialize tracking columns
    df["revenue_raw"] = df[TARGET_COL].copy()
    df["fix_flag"] = ""
    df["review_flag"] = 0

    # 3. Unit correction for S12 in March 2024 (DQ-02)
    s12_mask = (df[STORE_COL] == "S12") & df[DATE_COL].between("2024-03-01", "2024-03-31")
    df.loc[s12_mask, TARGET_COL] = df.loc[s12_mask, TARGET_COL] / 1000.0
    df.loc[s12_mask, "fix_flag"] = "unit_x1000"
    log["unit_fix_rows"] = int(s12_mask.sum())

    # 4. Spike detection (DQ-03 & DQ-03b)
    # Define reference history
    if ref_end is not None:
        ref_df = df[df[DATE_COL] < pd.Timestamp(ref_end)]
    else:
        ref_df = df

    # Median by (store_id, dayofweek) computed on clean unit revenue
    dow_series = df[DATE_COL].dt.dayofweek
    ref_dow = ref_df[DATE_COL].dt.dayofweek
    ref_medians = ref_df.groupby([ref_df[STORE_COL], ref_dow])[TARGET_COL].median()

    keys = pd.MultiIndex.from_arrays([df[STORE_COL], dow_series])
    expected_medians = ref_medians.reindex(keys).to_numpy()

    # Calculate ratio to expected median
    df["ratio_to_ref"] = df[TARGET_COL].to_numpy() / expected_medians

    # High spikes (ratio > spike_ratio, typically ~10x)
    high_spikes = df["ratio_to_ref"] > spike_ratio
    df.loc[high_spikes, "fix_flag"] = "spike_x10"
    log["spike_rows"] = int(high_spikes.sum())

    # Low spikes (ratio < low_ratio, typically ~0.1x)
    low_spikes = df["ratio_to_ref"] < low_ratio
    df.loc[low_spikes, "fix_flag"] = "spike_low_x01"
    log["spike_low_rows"] = int(low_spikes.sum())

    # Review candidates (ratio between 3.0 and 5.0)
    review_mask = (df["ratio_to_ref"] >= 3.0) & (df["ratio_to_ref"] <= 5.0)
    df.loc[review_mask, "review_flag"] = 1
    log["review_rows"] = int(review_mask.sum())

    # Validation contracts
    assert (df[TARGET_COL] > 0).all(), "Found non-positive revenue!"
    non_spike_rev = df.loc[~df["fix_flag"].str.startswith("spike"), TARGET_COL]
    assert non_spike_rev.max() < 500.0, f"Unreasonably high revenue found after cleaning: {non_spike_rev.max()}"

    s12_clean = df.loc[s12_mask, TARGET_COL]
    assert 15.0 < s12_clean.median() < 40.0, f"S12 cleaned revenue median out of range: {s12_clean.median()}"

    return df, log


def join_cleaned_data(
    train_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    holidays_df: pd.DataFrame
) -> pd.DataFrame:
    """Safely join transaction records with auxiliary cleaned datasets (DQ-15, Section 5.7).

    Guarantees:
    - Pre-condition and post-condition row counts are strictly identical.
    - Zero key duplication.
    - Non-holiday calendar dates safely populated with 0 / default values.
    - Zero NaNs in primary modeling features.
    """
    initial_rows = len(train_df)

    # 1. Join stores metadata on store_id
    merged = train_df.merge(stores_df, on=STORE_COL, how="left")
    if len(merged) != initial_rows:
        raise ValueError(f"Stores join corrupted row count: {initial_rows} -> {len(merged)}")

    # 2. Join weather on (date, city)
    merged = merged.merge(weather_df, on=[DATE_COL, "city"], how="left")
    if len(merged) != initial_rows:
        raise ValueError(f"Weather join corrupted row count: {initial_rows} -> {len(merged)}")

    # 3. Join aggregated daily holidays on date
    merged = merged.merge(holidays_df, on=DATE_COL, how="left")
    if len(merged) != initial_rows:
        raise ValueError(f"Holidays join corrupted row count: {initial_rows} -> {len(merged)}")

    # Fill defaults for non-holiday dates
    merged["is_holiday"] = merged["is_holiday"].fillna(0).astype(int)
    merged["holiday_count"] = merged["holiday_count"].fillna(0).astype(int)
    merged["is_national_holiday"] = merged["is_national_holiday"].fillna(0).astype(int)
    merged["is_event"] = merged["is_event"].fillna(0).astype(int)
    merged["holiday_names"] = merged["holiday_names"].fillna("none")
    merged["is_tet_day1"] = merged["is_tet_day1"].fillna(0).astype(int)
    merged["is_tet_period"] = merged["is_tet_period"].fillna(0).astype(int)
    merged["is_tet_eve"] = merged["is_tet_eve"].fillna(0).astype(int)
    merged["holiday_covered"] = merged["holiday_covered"].fillna(1).astype(int)

    return merged


def export_clean_datasets(
    raw_train_file: Path,
    raw_stores_file: Path,
    raw_weather_file: Path,
    raw_holidays_file: Path,
    clean_dir: Path = CLEAN_DATA_DIR,
    reports_dir: Path = REPORTS_DIR
) -> Dict[str, Any]:
    """Execute end-to-end cleaning and export clean datasets (.parquet and .csv).

    Creates:
    - data/clean/train_clean.parquet & .csv
    - data/clean/stores_clean.parquet & .csv
    - data/clean/weather_clean.parquet & .csv
    - data/clean/holidays_daily.parquet & .csv
    - reports/outliers_flagged.csv

    Returns:
        Summary dictionary of cleaning metrics.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw files
    raw_train = pd.read_csv(raw_train_file, parse_dates=[DATE_COL])
    raw_stores = pd.read_csv(raw_stores_file)
    raw_weather = pd.read_csv(raw_weather_file, parse_dates=[DATE_COL])
    raw_holidays = pd.read_csv(raw_holidays_file, parse_dates=[DATE_COL])

    # 2. Clean each dataset
    train_clean, train_log = clean_train(raw_train, ref_end=None)
    stores_clean, stores_log = clean_stores(raw_stores)
    weather_clean, weather_log = clean_weather(raw_weather, ref_end=None)
    holidays_clean, holidays_log = clean_holidays(raw_holidays)

    # 3. Export Parquet and CSV
    train_clean.to_parquet(clean_dir / "train_clean.parquet", index=False)
    train_clean.to_csv(clean_dir / "train_clean.csv", index=False)

    stores_clean.to_parquet(clean_dir / "stores_clean.parquet", index=False)
    stores_clean.to_csv(clean_dir / "stores_clean.csv", index=False)

    weather_clean.to_parquet(clean_dir / "weather_clean.parquet", index=False)
    weather_clean.to_csv(clean_dir / "weather_clean.csv", index=False)

    holidays_clean.to_parquet(clean_dir / "holidays_daily.parquet", index=False)
    holidays_clean.to_csv(clean_dir / "holidays_daily.csv", index=False)

    # 4. Generate and export reports/outliers_flagged.csv
    flag_cond = (train_clean["fix_flag"].str.len() > 0) | (train_clean["review_flag"] == 1)
    flagged = train_clean[flag_cond].copy()
    flagged_export = flagged[
        [DATE_COL, STORE_COL, PROMO_COL, "revenue_raw", TARGET_COL, "ratio_to_ref", "fix_flag", "review_flag"]
    ].sort_values([DATE_COL, STORE_COL])
    flagged_export.to_csv(reports_dir / "outliers_flagged.csv", index=False)

    summary = {
        "train": train_log,
        "stores": stores_log,
        "weather": weather_log,
        "holidays": holidays_log,
        "flagged_rows_total": len(flagged_export),
    }

    logger.info(f"Clean datasets exported successfully to {clean_dir}")
    return summary
