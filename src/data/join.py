"""Data joining, cleaning, and preparation pipeline for cafe revenue regression.

Adheres strictly to Regression_Plan.md Section 5:
- 5.1: Chuẩn hóa ngày và category (store_type, open_date multi-format, city names).
- 5.2: Kiểm tra bản ghi trùng và bảo toàn khóa (date, store_id).
- 5.3: Missing values và giá trị bất thường (area_m2_missing, rain_missing, sửa lỗi nhiệt độ Fahrenheit).
- 5.4: Join an toàn nhiều bảng (stores, weather, holidays tổng hợp 1 dòng/ngày), bảo toàn số dòng.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.config import DATE_COL, STORE_COL, TARGET_COL, PROMO_COL

logger = logging.getLogger(__name__)


def clean_stores(stores_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize store metadata.

    - Normalizes store_type to lowercase stripped string ('street', 'office', 'mall', 'university', 'residential').
    - Dual-format parsing for open_date (ISO YYYY-MM-DD and DD/MM/YYYY).
    - Adds binary indicator area_m2_missing for rows with unrecorded area.
    - Strips whitespace from string columns (city, district).
    """
    df = stores_df.copy()

    # Normalize category strings
    df["store_type"] = df["store_type"].astype(str).str.strip().str.lower()
    df["city"] = df["city"].astype(str).str.strip()
    df["district"] = df["district"].astype(str).str.strip()

    # Parse open_date with regex matching per Regression_Plan Section 5.1
    s_date = df["open_date"].astype("string").str.strip()
    iso_mask = s_date.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    dmy_mask = s_date.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)

    df["open_date_parsed"] = pd.NaT
    df.loc[iso_mask, "open_date_parsed"] = pd.to_datetime(
        s_date[iso_mask], format="%Y-%m-%d", errors="coerce"
    )
    df.loc[dmy_mask, "open_date_parsed"] = pd.to_datetime(
        s_date[dmy_mask], format="%d/%m/%Y", errors="coerce"
    )

    unparsed = df["open_date_parsed"].isna().sum()
    if unparsed > 0:
        raise ValueError(f"Failed to parse open_date for {unparsed} stores.")

    # Binary missing flag for area_m2
    df["area_m2_missing"] = df["area_m2"].isna().astype(int)

    # Ensure integer flags
    df["has_parking"] = df["has_parking"].fillna(0).astype(int)
    df["seats"] = df["seats"].astype(float)

    logger.info(
        f"Cleaned stores: {len(df)} stores, {df['area_m2_missing'].sum()} missing area_m2 records flagged."
    )
    return df


def clean_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and aggregate holidays into exactly one row per date.

    Prevents row multiplication on dates with multiple events (e.g., 2024-02-14 has both
    'Tet holiday' and 'Valentine\'s Day') as warned in Regression_Plan Section 5.4.
    """
    df = holidays_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["holiday_name"] = df["holiday_name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    # Group by date to produce exactly one row per calendar date
    agg_df = df.groupby(DATE_COL).agg(
        holiday_count=("holiday_name", "count"),
        is_national_holiday=("type", lambda s: int((s == "national_holiday").any())),
        is_event=("type", lambda s: int((s == "event").any())),
        holiday_names=("holiday_name", lambda s: "; ".join(sorted(set(s)))),
    ).reset_index()

    agg_df["is_holiday"] = 1

    logger.info(
        f"Aggregated holidays: {len(df)} raw events collapsed to {len(agg_df)} unique holiday dates."
    )
    return agg_df


def clean_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Clean weather data and fix unit discrepancies.

    Investigated in Regression_Plan Section 2.2 & 5.3:
    - In June 2025, Ho Chi Minh temperature values (e.g., 84.6) were erroneously recorded in Fahrenheit.
    - Values > 50°C are converted via (F - 32) * 5 / 9, restoring them to realistic tropical 27-32°C.
    - Generates binary indicators for missing rain/temp and rain occurrence.
    """
    df = weather_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["city"] = df["city"].astype(str).str.strip()

    # Temperature Fahrenheit correction
    f_mask = df["temp_c"] > 50.0
    f_count = int(f_mask.sum())
    if f_count > 0:
        logger.info(f"Detected {f_count} temperature values > 50°C. Converting from Fahrenheit to Celsius...")
        df.loc[f_mask, "temp_c"] = (df.loc[f_mask, "temp_c"] - 32.0) * 5.0 / 9.0

    # Missing flags
    df["temp_missing"] = df["temp_c"].isna().astype(int)
    df["rain_missing"] = df["rain_mm"].isna().astype(int)
    df["is_rainy"] = (df["rain_mm"] > 0).astype(int)

    logger.info(
        f"Cleaned weather: {len(df)} rows, {df['rain_missing'].sum()} missing rain flagged, {f_count} Fahrenheit values corrected."
    )
    return df


def join_all_data(
    transactions_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """Safe join across transaction records and cleaned auxiliary tables.

    Guarantees:
    - Pre-condition and post-condition row counts are strictly identical (no row explosion).
    - Keys remain unique per (date, store_id).
    - Holiday flags default to 0 for regular non-holiday days.
    """
    initial_len = len(transactions_df)

    # 1. Clean auxiliary tables
    stores_clean = clean_stores(stores_df)
    holidays_clean = clean_holidays(holidays_df)
    weather_clean = clean_weather(weather_df)

    # 2. Merge stores metadata on store_id
    merged = transactions_df.merge(stores_clean, on=STORE_COL, how="left")
    if len(merged) != initial_len:
        raise ValueError(
            f"Row count changed after merging stores: {initial_len} -> {len(merged)}"
        )

    # 3. Merge weather on (date, city)
    merged = merged.merge(weather_clean, on=[DATE_COL, "city"], how="left")
    if len(merged) != initial_len:
        raise ValueError(
            f"Row count changed after merging weather: {initial_len} -> {len(merged)}"
        )

    # 4. Merge aggregated holidays on date
    merged = merged.merge(holidays_clean, on=DATE_COL, how="left")
    if len(merged) != initial_len:
        raise ValueError(
            f"Row count changed after merging holidays: {initial_len} -> {len(merged)}"
        )

    # Fill default values for non-holiday days
    merged["is_holiday"] = merged["is_holiday"].fillna(0).astype(int)
    merged["holiday_count"] = merged["holiday_count"].fillna(0).astype(int)
    merged["is_national_holiday"] = merged["is_national_holiday"].fillna(0).astype(int)
    merged["is_event"] = merged["is_event"].fillna(0).astype(int)
    merged["holiday_names"] = merged["holiday_names"].fillna("none")

    # Fill weather missing flags if entire weather row was missing from merge
    merged["temp_missing"] = merged["temp_missing"].fillna(1).astype(int)
    merged["rain_missing"] = merged["rain_missing"].fillna(1).astype(int)
    merged["is_rainy"] = merged["is_rainy"].fillna(0).astype(int)

    logger.info(f"Successfully joined all datasets into table with {len(merged)} rows and {merged.shape[1]} columns.")
    return merged
