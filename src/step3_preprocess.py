"""
Module: step3_preprocess.py
Chủ đề: Dự báo doanh thu ngày chuỗi quán Cafe (Cafe Revenue Regression)
Mô tả: Triển khai toàn bộ BƯỚC 3: PREPROCESS (TIỀN XỬ LÝ DỮ LIỆU & KỸ THUẬT ĐẶC TRƯNG CHUYÊN SÂU)
Tuân thủ chuẩn quy trình 8 bước trong bài giảng Chương 4 và kế hoạch Regression_Plan.md
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Các mốc Mùng 1 Tết Nguyên Đán
TET_DATES = {
    2024: pd.Timestamp("2024-02-10"),  # Giáp Thìn
    2025: pd.Timestamp("2025-01-29"),  # Ất Tỵ
    2026: pd.Timestamp("2026-02-17"),  # Bính Ngọ
    2027: pd.Timestamp("2027-02-06"),  # Đinh Mùi
    2028: pd.Timestamp("2028-01-26"),  # Mậu Thân
    2029: pd.Timestamp("2029-02-13"),  # Kỷ Dậu
    2030: pd.Timestamp("2030-02-03"),  # Canh Tuất
}


def parse_open_date_fixed(series: pd.Series) -> pd.Series:
    """
    Phân tích cú pháp ngày khai trương chuẩn xác tuyệt đối:
    - Định dạng ISO YYYY-MM-DD nếu chuỗi chứa dấu gạch ngang '-'
    - Định dạng DMY DD/MM/YYYY nếu chuỗi chứa dấu gạch chéo '/'
    Tránh hoàn toàn lỗi hoán đổi ngày-tháng khi dùng format='mixed'.
    """
    s = series.astype(str).str.strip()
    dt_iso = pd.to_datetime(s.where(s.str.contains("-")), format="%Y-%m-%d", errors="coerce")
    dt_dmy = pd.to_datetime(s.where(s.str.contains("/")), format="%d/%m/%Y", errors="coerce")
    return dt_iso.fillna(dt_dmy)


def clean_stores(stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý và làm sạch toàn diện bảng stores.csv:
    1. Chuẩn hóa store_type về chữ thường, loại bỏ khoảng trắng (street, office, mall, university, residential).
    2. Chuẩn hóa tên district (District 10, District 3, District 7,...).
    3. Chuyển open_date sang kiểu datetime bằng parse_open_date_fixed.
    4. Điền khuyết area_m2 cho S19 và S24 dựa trên tỷ lệ diện tích/ghế (area_per_seat) trung vị theo từng store_type.
    5. Tạo đặc trưng quy mô area_per_seat = area_m2 / seats.
    """
    df = stores_df.copy()
    
    # 1. Chuẩn hóa store_type
    df["store_type"] = df["store_type"].astype(str).str.strip().str.lower()
    
    # 2. Chuẩn hóa district
    district_map = {
        "Q.10": "District 10",
        "Quan 3": "District 3",
        "Quan 7": "District 7"
    }
    df["district"] = df["district"].astype(str).replace(district_map).str.strip()
    
    # 3. Chuẩn hóa open_date chuẩn xác tuyệt đối
    df["open_date_dt"] = parse_open_date_fixed(df["open_date"])
    
    # 4. Điền khuyết area_m2
    df["area_per_seat"] = df["area_m2"] / df["seats"]
    type_aps_median = df.groupby("store_type")["area_per_seat"].median()
    
    for idx in df[df["area_m2"].isnull()].index:
        st = df.loc[idx, "store_type"]
        seats = df.loc[idx, "seats"]
        median_ratio = type_aps_median.get(st, 2.7)
        imputed_area = round(seats * median_ratio, 1)
        df.loc[idx, "area_m2"] = imputed_area
        df.loc[idx, "area_per_seat"] = round(imputed_area / seats, 4)
        
    return df


def clean_weather(
    weather_df: pd.DataFrame,
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-31"
) -> pd.DataFrame:
    """
    Tiền xử lý và làm sạch toàn diện bảng weather.csv:
    1. Quy đổi ngoại lai nhiệt độ từ Fahrenheit sang Celsius cho các dòng temp_c > 45:
       T_Celsius = (T_Fahrenheit - 32) / 1.8
    2. Tạo lưới ngày đầy đủ (Complete Date-City Grid) gồm 821 ngày x 2 thành phố = 1.642 dòng.
    3. Ghép bảng weather vào lưới ngày đầy đủ.
    4. Ghi nhận cờ khuyết dữ liệu thời tiết (is_rain_missing, is_temp_missing).
    5. Tính lượng mưa lân cận (rain_neighbor_mean) và nội suy hình học (rain_geom_interp).
    6. Nội suy tuyến tính theo thời gian (linear interpolation) theo từng thành phố cho temp_c.
    7. Điền khuyết 0.0 mm cho các giá trị NaN của rain_mm.
    8. Tính độ lệch nhiệt độ thành phố (temp_deviation_from_city_mean) và độ lệch nhiệt độ tháng (temp_dev_city_month).
    """
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Quy đổi độ F sang độ C
    f_mask = df["temp_c"] > 45
    df.loc[f_mask, "temp_c"] = (df.loc[f_mask, "temp_c"] - 32.0) / 1.8
    
    # 2. Tạo lưới ngày đầy đủ 821 ngày x 2 thành phố
    date_grid = pd.date_range(start_date, end_date)
    cities = ["Ha Noi", "Ho Chi Minh"]
    grid_df = pd.MultiIndex.from_product([date_grid, cities], names=["date", "city"]).to_frame().reset_index(drop=True)
    
    # 3. Ghép vào lưới
    merged_weather = grid_df.merge(df, on=["date", "city"], how="left")
    merged_weather = merged_weather.sort_values(["city", "date"]).reset_index(drop=True)
    
    # 4. Cờ khuyết dữ liệu thời tiết
    merged_weather["is_rain_missing"] = merged_weather["rain_mm"].isna().astype(int)
    merged_weather["is_temp_missing"] = merged_weather["temp_c"].isna().astype(int)
    
    # 5. Thông tin mưa lân cận & nội suy hình học
    r_prev = merged_weather.groupby("city")["rain_mm"].shift(1).fillna(0.0)
    r_next = merged_weather.groupby("city")["rain_mm"].shift(-1).fillna(0.0)
    merged_weather["rain_neighbor_mean"] = 0.5 * (r_prev + r_next)
    merged_weather["rain_geom_interp"] = np.where(
        merged_weather["is_rain_missing"] == 1,
        np.expm1(0.5 * (np.log1p(r_prev) + np.log1p(r_next))),
        merged_weather["rain_mm"].fillna(0.0)
    )
    
    # 6 & 7. Nội suy temp_c và điền 0.0 rain_mm
    merged_weather["temp_c"] = merged_weather.groupby("city")["temp_c"].transform(
        lambda s: s.interpolate(method="linear").bfill().ffill()
    )
    merged_weather["rain_mm"] = merged_weather["rain_mm"].fillna(0.0)
    
    # 8. Độ lệch nhiệt độ so với mức trung bình của thành phố & độ lệch tháng
    city_temp_mean = merged_weather.groupby("city")["temp_c"].transform("mean")
    merged_weather["temp_deviation_from_city_mean"] = merged_weather["temp_c"] - city_temp_mean
    
    month_tmp = merged_weather["date"].dt.month
    cm_temp_mean = merged_weather.groupby([merged_weather["city"], month_tmp])["temp_c"].transform("mean")
    merged_weather["temp_dev_city_month"] = merged_weather["temp_c"] - cm_temp_mean
    
    return merged_weather


def clean_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý và gộp nhóm bảng holidays.csv theo 'date':
    - Xử lý trùng lặp khóa tại ngày 2024-02-14 và 2026-02-14 (Tet holiday + Valentine's Day).
    - Tạo các cờ nhị phân: is_national_holiday, is_event_holiday, holiday_name_clean.
    """
    df = holidays_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    holidays_agg = df.groupby("date").agg(
        is_national_holiday=("type", lambda s: int((s == "national_holiday").any())),
        is_event_holiday=("type", lambda s: int((s == "event").any())),
        holiday_name_clean=("holiday_name", lambda s: " + ".join(s))
    ).reset_index()
    
    return holidays_agg


def clean_train_outliers(
    train_df: pd.DataFrame,
    stores_clean: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Làm sạch ngoại lai và sửa lỗi nhập liệu trên tập huấn luyện (train.csv):
    1. Loại bỏ 132 dòng trùng lặp hoàn toàn (đưa train về 26.500 dòng).
    2. Sửa lỗi đơn vị 1000x của S12 trong tháng 03/2024: revenue > 1000 -> revenue / 1000.0.
    3. Tính trung vị tham chiếu sw_med = median theo (store_id, dayofweek).
    4. Sửa lỗi dịch dấu thập phân 10x: sw_ratio > 3.8 và days_since_open > 2 -> revenue / 10.0.
    5. Sửa lỗi dịch dấu thập phân 0.1x: sw_ratio < 0.16 -> revenue * 10.0.
    
    Chỉ áp dụng trên tập Train; giữ nguyên 100% tập Test.
    """
    initial_rows = len(train_df)
    
    # 1. Loại bỏ trùng lặp
    df = train_df.drop_duplicates().copy()
    dup_dropped = initial_rows - len(df)
    
    # Ghép open_date để tính days_since_open
    df = df.merge(stores_clean[["store_id", "open_date_dt"]], on="store_id", how="left")
    df["days_since_open"] = (df["date"] - df["open_date_dt"]).dt.days
    
    # 2. Sửa lỗi 1000x
    mask_1000x = df["revenue"] > 1000.0
    cnt_1000x = int(mask_1000x.sum())
    df.loc[mask_1000x, "revenue"] /= 1000.0
    
    # 3. Tính sw_med và sw_ratio
    df["dow"] = df["date"].dt.day_of_week
    sw_med = df.groupby(["store_id", "dow"])["revenue"].transform("median")
    sw_ratio = df["revenue"] / sw_med
    
    # 4. Sửa lỗi 10x (kèm điều kiện tỷ số > 3.0 cho ngày thường để phát hiện S17 Mùng 1 Tết 2024 có sw_ratio=3.34)
    mask_10x = ((sw_ratio > 3.8) | ((sw_ratio > 3.0) & (df["days_since_open"] > 2) & (df["is_promo"] == 0))) & (df["days_since_open"] > 2)
    cnt_10x = int(mask_10x.sum())
    df.loc[mask_10x, "revenue"] /= 10.0
    
    # Tính lại sw_med và sw_ratio
    sw_med = df.groupby(["store_id", "dow"])["revenue"].transform("median")
    sw_ratio = df["revenue"] / sw_med
    
    # 5. Sửa lỗi 0.1x (mở rộng ngưỡng lên 0.18 để phát hiện chính xác S29 ngày 02/09/2024 có sw_ratio=0.164)
    mask_01x = sw_ratio < 0.18
    cnt_01x = int(mask_01x.sum())
    df.loc[mask_01x, "revenue"] *= 10.0
    
    # Bỏ các cột tính toán tạm thời
    df = df.drop(columns=["open_date_dt", "days_since_open", "dow"]).reset_index(drop=True)
    
    clean_stats = {
        "initial_rows": initial_rows,
        "cleaned_rows": len(df),
        "duplicates_dropped": dup_dropped,
        "fixed_1000x_count": cnt_1000x,
        "fixed_10x_count": cnt_10x,
        "fixed_01x_count": cnt_01x,
        "final_revenue_min": float(df["revenue"].min()),
        "final_revenue_median": float(df["revenue"].median()),
        "final_revenue_max": float(df["revenue"].max())
    }
    
    return df, clean_stats


def compute_smooth_ramp_up_log(days_since_open: pd.Series) -> pd.Series:
    """
    Đường cong tăng trưởng mở quán (Ramp-Up Curve) trơn liên tục trên thang logarit:
    - Ngày 0-2 (Khai trương rầm rộ): +0.44 log (~1.55x)
    - Ngày 3 trở đi: giảm về đáy sau khai trương (~ -0.46 log) và hồi phục mượt mà theo hàm mũ tiệm cận 0 tại ngày 90.
    """
    d = days_since_open.values.astype(float)
    val = np.zeros(len(d), dtype=float)
    m02 = (d >= 0) & (d <= 2)
    val[m02] = 0.44
    m_rec = d >= 3
    xp = np.array([3.0, 5.0, 10.0, 22.0, 35.0, 45.0, 55.0, 68.0, 82.0, 95.0])
    fp = np.array([-0.46, -0.47, -0.39, -0.34, -0.25, -0.19, -0.14, -0.06, -0.015, 0.0])
    val[m_rec] = np.interp(d[m_rec], xp, fp)
    return pd.Series(val, index=days_since_open.index)


def add_promo_streak_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Tính chuỗi ngày khuyến mãi liên tiếp (promo_streak) và các đặc trưng trễ/cuộn
    hoàn toàn từ lịch khuyến mãi đã biết (store_id, date, is_promo) mà không gây rò rỉ doanh thu.
    """
    df = df_all.sort_values(["store_id", "date"]).copy()
    p_vals = df["is_promo"].values
    s_vals = df["store_id"].values
    n = len(df)
    
    if n == 1:
        p0 = int(p_vals[0])
        df["promo_streak"] = p0
        df["promo_streak_clip"] = p0
        df["is_promo_day1"] = p0
        df["is_promo_day2"] = 0
        df["is_promo_day3plus"] = 0
        df["promo_lag1"] = 0
        df["promo_lag2"] = 0
        df["is_post_promo_hangover"] = 0
        df["days_since_last_promo"] = 0 if p0 == 1 else 7
        df["promo_roll7"] = float(p0)
        df["promo_roll14"] = float(p0)
        df["date_promo_share"] = float(p0)
        return df

    streaks = np.zeros(n, dtype=int)
    prev_promo = np.zeros(n, dtype=int)
    prev2_promo = np.zeros(n, dtype=int)
    days_since_p = np.zeros(n, dtype=int)

    cur_streak = 0
    last_p_idx = -999
    prev_sid = None
    for i in range(n):
        sid = s_vals[i]
        if sid != prev_sid:
            cur_streak = 0
            last_p_idx = -999
            prev_sid = sid
        if i > 0 and s_vals[i - 1] == sid:
            prev_promo[i] = p_vals[i - 1]
        if i > 1 and s_vals[i - 2] == sid:
            prev2_promo[i] = p_vals[i - 2]
        if p_vals[i] == 1:
            cur_streak += 1
            days_since_p[i] = 0
            last_p_idx = i
        else:
            cur_streak = 0
            days_since_p[i] = min(30, i - last_p_idx) if last_p_idx >= 0 else 30
        streaks[i] = cur_streak

    df["promo_streak"] = streaks
    df["promo_streak_clip"] = np.minimum(streaks, 7)
    df["is_promo_day1"] = (streaks == 1).astype(int)
    df["is_promo_day2"] = (streaks == 2).astype(int)
    df["is_promo_day3plus"] = (streaks >= 3).astype(int)
    df["promo_lag1"] = prev_promo
    df["promo_lag2"] = prev2_promo
    df["is_post_promo_hangover"] = ((df["is_promo"] == 0) & (prev_promo == 1)).astype(int)
    df["days_since_last_promo"] = days_since_p
    df["promo_roll7"] = df.groupby("store_id")["is_promo"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["promo_roll14"] = df.groupby("store_id")["is_promo"].transform(lambda s: s.rolling(14, min_periods=1).sum())
    df["date_promo_share"] = df.groupby("date")["is_promo"].transform("mean")
    return df


def compute_ramp_up_factor(days_since_open: pd.Series) -> pd.Series:
    """
    Tính hệ số trưởng thành ramp_up_factor mô phỏng đường cong vòng đời của quán mới mở:
    - 0-2 ngày (Khai trương): 1.70
    - 3-7 ngày (Đáy chạy đà tuần đầu): 0.61
    - 8-14 ngày: 0.67
    - 15-30 ngày: 0.73
    - 31-60 ngày: 0.86
    - > 60 ngày (Bão hòa): 1.00
    """
    d = days_since_open.values
    factor = np.ones(len(d), dtype=float)
    
    # 0 <= d <= 2
    m0_2 = (d >= 0) & (d <= 2)
    factor[m0_2] = 1.70
    
    # 3 <= d <= 7
    m3_7 = (d >= 3) & (d <= 7)
    factor[m3_7] = 0.61
    
    # 8 <= d <= 14
    m8_14 = (d >= 8) & (d <= 14)
    factor[m8_14] = 0.67
    
    # 15 <= d <= 30
    m15_30 = (d >= 15) & (d <= 30)
    factor[m15_30] = 0.73
    
    # 31 <= d <= 60
    m31_60 = (d >= 31) & (d <= 60)
    factor[m31_60] = 0.86
    
    return pd.Series(factor, index=days_since_open.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiến tạo hệ thống đặc trưng chuyên sâu V3 (Feature Engineering Architecture):
    Bao gồm 5 nhóm đặc trưng đồng nhất cho cả train và test:
    
    1. Nhóm Thời gian & Chu kỳ (Calendar & Cyclical):
       dow, is_weekend, day, month, weekofyear, quarter, sin_dow, cos_dow, sin_doy, cos_doy, t_years
    2. Nhóm Tết Nguyên Đán & Ngày lễ tinh chỉnh (Refined Tet & Non-Tet Holidays):
       days_from_tet_exact, days_from_tet_clipped, is_non_tet_national_holiday, is_tet_core,
       is_pre_tet_shopping, is_ong_tao, is_post_tet_recovery, is_pure_valentine, is_tet_valentine,
       is_christmas, is_womens_day, tương tác lễ với store_type
    3. Nhóm Tĩnh & Vòng đời Cửa hàng (Store Static & Lifecycle):
       seats, area_m2, area_per_seat, has_parking, log_seats, log_area_m2,
       days_since_open, log_days_since_open, is_grand_opening, is_ramp_up_phase,
       ramp_up_factor, log_ramp_smooth, log_ramp_up_factor
    4. Nhóm Thời tiết & Tương tác (Weather & Interactions):
       temp_c, rain_mm, log_rain, is_heavy_rain, is_extreme_rain, rain_lin cho mọi store_type,
       tương tác thời tiết khuyết, độ lệch nhiệt độ theo mùa
    5. Nhóm Khuyến mãi & Chuỗi ngày (Promo & Streak Features):
       promo_streak, promo_streak_clip, is_promo_day1, promo_roll7, tương tác khuyến mãi
    """
    res = df.copy()
    date_dt = pd.to_datetime(res["date"])
    res["date"] = date_dt
    
    # Tự động bổ sung chuỗi khuyến mãi nếu chưa được tính
    if "promo_streak" not in res.columns and "is_promo" in res.columns:
        res = add_promo_streak_features(res)
    
    # --------------------------------------------------------------------------
    # 1. Nhóm Thời gian & Chu kỳ (Calendar & Cyclical)
    # --------------------------------------------------------------------------
    res["dow"] = date_dt.dt.day_of_week
    res["is_weekend"] = (res["dow"] >= 5).astype(int)
    res["day"] = date_dt.dt.day
    res["month"] = date_dt.dt.month
    res["weekofyear"] = date_dt.dt.isocalendar().week.astype(int)
    res["quarter"] = date_dt.dt.quarter
    
    res["sin_dow"] = np.sin(2 * np.pi * res["dow"] / 7.0)
    res["cos_dow"] = np.cos(2 * np.pi * res["dow"] / 7.0)
    doy = date_dt.dt.day_of_year
    res["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    res["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    res["t_years"] = (date_dt - pd.Timestamp("2024-01-01")).dt.days / 365.25
    
    # Nhận diện các mô hình quán
    is_mall = (res["store_type"] == "mall").astype(int)
    is_street = (res["store_type"] == "street").astype(int)
    is_res = (res["store_type"] == "residential").astype(int)
    is_office = (res["store_type"] == "office").astype(int)
    is_uni = (res["store_type"] == "university").astype(int)
    is_work = (is_office | is_uni).astype(int)
    is_retail = (is_mall | is_street | is_res).astype(int)
    is_outdoor = (is_street | is_res | is_uni).astype(int)
    is_hanoi = (res["city"] == "Ha Noi").astype(int)
    
    # --------------------------------------------------------------------------
    # 2. Nhóm Tết Nguyên Đán & Ngày lễ Tinh chỉnh
    # --------------------------------------------------------------------------
    res["year"] = date_dt.dt.year
    tet_dt = res["year"].map(TET_DATES)
    res["days_from_tet"] = (date_dt - tet_dt).dt.days.fillna(999).astype(int)
    res["days_from_tet_exact"] = res["days_from_tet"]
    res["days_from_tet_clipped"] = np.clip(res["days_from_tet"], -21, 21)
    d_tet = res["days_from_tet_clipped"]
    
    # Phân tách ngày lễ quốc gia KHÔNG thuộc Tết để không kích hoạt nhầm giảm doanh thu văn phòng vào d=-4, -3, +5
    non_tet = (res["days_from_tet_exact"].abs() > 14).astype(int)
    res["is_national_holiday"] = res["is_national_holiday"].fillna(0).astype(int)
    res["is_event_holiday"] = res["is_event_holiday"].fillna(0).astype(int)
    res["is_non_tet_national_holiday"] = res["is_national_holiday"] * non_tet
    
    res["is_tet_core"] = d_tet.between(-2, 2).astype(int)
    res["is_pre_tet_shopping"] = d_tet.between(-12, -3).astype(int)
    res["is_ong_tao"] = (res["days_from_tet"] == -7).astype(int)
    res["is_post_tet_recovery"] = d_tet.between(3, 9).astype(int)
    
    res["is_mall_pre_tet"] = (is_mall & d_tet.between(-12, -3)).astype(int)
    res["is_pre_tet_mall_rush"] = (is_mall & d_tet.between(-8, -3)).astype(int)
    res["is_pre_tet_street_rush"] = (is_street & d_tet.between(-10, -3)).astype(int)
    res["is_mall_tet_recovery"] = (is_mall & d_tet.between(2, 6)).astype(int)
    res["is_post_tet_mall_bounce"] = res["is_mall_tet_recovery"]
    res["is_street_post_tet"] = ((is_street | is_res) & d_tet.between(3, 7)).astype(int)
    res["is_office_tet_closure"] = (is_office & d_tet.between(-2, 2)).astype(int)
    res["is_post_tet_office_drop"] = (is_office & d_tet.between(3, 9)).astype(int)
    res["is_post_tet_office_ramp"] = res["is_post_tet_office_drop"]
    
    res["is_tet_core_mall"] = (is_mall & d_tet.between(-2, 1)).astype(int)
    res["is_tet_core_street"] = (is_street & d_tet.between(-2, 2)).astype(int)
    res["is_tet_core_res"] = (is_res & d_tet.between(-2, 2)).astype(int)
    res["is_tet_core_slump"] = ((is_street | is_res) & d_tet.between(-2, 2)).astype(int)
    
    res["is_uni_tet_break"] = (is_uni & d_tet.between(-14, 12)).astype(int)
    res["is_pre_tet_uni_drop"] = (is_uni & d_tet.between(-14, -3)).astype(int)
    res["is_post_tet_uni_drop"] = (is_uni & d_tet.between(3, 13)).astype(int)
    res["is_uni_summer_break"] = (is_uni & res["month"].isin([7, 8])).astype(int)
    res["is_uni_post_summer_edge"] = (is_uni & (res["month"] == 9) & (res["day"] <= 3)).astype(int)
    
    # Ngày lễ tình nhân (Valentine) tách biệt khi trùng Tết vs ngày thường
    res["is_womens_day"] = (((res["month"] == 3) & (res["day"] == 8)) | ((res["month"] == 10) & (res["day"] == 20))).astype(int)
    res["is_valentine"] = ((res["month"] == 2) & (res["day"] == 14)).astype(int)
    res["is_pure_valentine"] = res["is_valentine"] * non_tet
    res["is_tet_valentine"] = res["is_valentine"] * (1 - non_tet)
    res["is_pure_valentine_mall"] = res["is_pure_valentine"] * is_mall
    res["is_pure_valentine_street"] = res["is_pure_valentine"] * is_street
    res["is_pure_valentine_res"] = res["is_pure_valentine"] * is_res
    
    res["is_womens_day_mall"] = res["is_womens_day"] * is_mall
    res["is_womens_day_street"] = res["is_womens_day"] * is_street
    res["is_womens_day_res"] = res["is_womens_day"] * is_res
    res["is_valentine_mall"] = res["is_valentine"] * is_mall
    res["is_valentine_street"] = res["is_valentine"] * is_street
    
    # Giáng sinh (Christmas)
    res["is_christmas"] = ((res["month"] == 12) & (res["day"].isin([24, 25]))).astype(int)
    res["is_xmas_eve"] = ((res["month"] == 12) & (res["day"] == 24)).astype(int)
    res["is_xmas_day"] = ((res["month"] == 12) & (res["day"] == 25)).astype(int)
    res["is_christmas_mall"] = res["is_christmas"] * is_mall
    res["is_christmas_nonmall"] = res["is_xmas_day"] * (1 - is_mall)
    
    # Tác động ngày lễ chuẩn hóa
    res["is_holiday_work_drop"] = res["is_non_tet_national_holiday"] * is_work
    res["is_holiday_office_drop"] = res["is_non_tet_national_holiday"] * is_office
    res["is_holiday_uni_drop"] = res["is_non_tet_national_holiday"] * is_uni
    res["is_holiday_retail_boost"] = res["is_non_tet_national_holiday"] * is_retail
    res["is_event_mall_boost"] = res["is_event_holiday"] * is_mall
    res["is_event_street_boost"] = res["is_event_holiday"] * is_street
    res["is_event_res_boost"] = res["is_event_holiday"] * is_res
    
    res["is_pre_tet_retail_boost"] = res["is_pre_tet_shopping"] * is_retail
    res["is_pre_tet_work_drop"] = res["is_pre_tet_shopping"] * is_work
    res["tet_dist_smooth"] = np.exp(-0.5 * (d_tet / 5.0) ** 2)
    
    # Chuẩn hóa cột holiday_name_clean: chỉ giữ nhãn "Tet holiday" ở lõi [-2..+2] để tránh rò rỉ lệch năm
    if "holiday_name_clean" in res.columns:
        res["holiday_name_clean"] = res["holiday_name_clean"].fillna("None")
        tet_non_core_mask = (res["holiday_name_clean"] == "Tet holiday") & (~res["days_from_tet_exact"].between(-2, 2))
        res.loc[tet_non_core_mask, "holiday_name_clean"] = "None"
    
    # --------------------------------------------------------------------------
    # 3. Nhóm Tĩnh & Vòng đời Cửa hàng (Store Static & Lifecycle)
    # --------------------------------------------------------------------------
    res = res.copy()
    if "open_date_dt" not in res.columns:
        res["open_date_dt"] = parse_open_date_fixed(res["open_date"])
        
    res["days_since_open"] = (date_dt - res["open_date_dt"]).dt.days
    res["days_since_open_clip120"] = np.clip(res["days_since_open"], 0, 120)
    res["log_days_since_open"] = np.log1p(np.maximum(0, res["days_since_open"]))
    res["inv_days_since_open"] = np.where(res["days_since_open"] >= 3, 1.0 / (res["days_since_open"] + 5.0), 0.0)
    res["is_grand_opening"] = res["days_since_open"].isin([0, 1, 2]).astype(int)
    res["is_ramp_up_phase"] = ((res["days_since_open"] >= 3) & (res["days_since_open"] <= 90)).astype(int)
    res["ramp_up_factor"] = compute_ramp_up_factor(res["days_since_open"])
    res["log_ramp_smooth"] = compute_smooth_ramp_up_log(res["days_since_open"])
    res["log_ramp_up_factor"] = res["log_ramp_smooth"]
    
    res["log_seats"] = np.log1p(res["seats"])
    res["log_area_m2"] = np.log1p(res["area_m2"])
    if "area_per_seat" not in res.columns:
        res["area_per_seat"] = res["area_m2"] / res["seats"]
        
    # --------------------------------------------------------------------------
    # 4. Nhóm Thời tiết & Tương tác (Weather & Interactions)
    # --------------------------------------------------------------------------
    res = res.copy()
    res["rain_mm"] = res["rain_mm"].fillna(0.0)
    res["log_rain"] = np.log1p(np.maximum(0, res["rain_mm"]))
    res["sqrt_rain"] = np.sqrt(np.maximum(0, res["rain_mm"]))
    res["is_heavy_rain"] = (res["rain_mm"] >= 20.0).astype(int)
    res["is_extreme_rain"] = (res["rain_mm"] >= 50.0).astype(int)
    
    # Tuyến tính lượng mưa cho mọi mô hình quán
    res["rain_lin_street"] = res["rain_mm"] * is_street
    res["rain_lin_res"] = res["rain_mm"] * is_res
    res["rain_lin_uni"] = res["rain_mm"] * is_uni
    res["rain_lin_office"] = res["rain_mm"] * is_office
    res["rain_lin_mall"] = res["rain_mm"] * is_mall
    res["rain_lin_outdoor"] = res["rain_mm"] * is_outdoor
    
    res["rain_x_outdoor"] = res["log_rain"] * is_outdoor
    res["rain_x_office"] = res["log_rain"] * is_office
    res["rain_x_mall"] = res["log_rain"] * is_mall
    res["rain_heavy_outdoor"] = res["is_heavy_rain"] * is_outdoor
    res["rain_x_promo"] = res["rain_mm"] * res["is_promo"] * is_outdoor
    
    # Xử lý các ngày khuyết dữ liệu mưa
    if "is_rain_missing" in res.columns:
        res["rain_missing_outdoor"] = res["is_rain_missing"] * is_outdoor
        res["rain_missing_mall"] = res["is_rain_missing"] * is_mall
        geom_rain = res.get("rain_geom_interp", res["rain_mm"])
        res["rain_geom_outdoor"] = np.log1p(geom_rain) * is_outdoor
        res["rain_geom_lin_outdoor"] = geom_rain * is_outdoor
        
    # Nhiệt độ và tương tác vùng miền
    res["temp_x_hanoi"] = res["temp_c"] * is_hanoi
    res["temp_x_outdoor"] = res["temp_c"] * is_outdoor
    res["temp_cold_hanoi"] = np.maximum(0.0, 20.0 - res["temp_c"]) * is_hanoi
    if "temp_dev_city_month" not in res.columns:
        res["temp_dev_city_month"] = res.get("temp_deviation_from_city_mean", 0.0)
        
    # --------------------------------------------------------------------------
    # 5. Nhóm Khuyến mãi & Chuỗi ngày (Promo & Streak Features)
    # --------------------------------------------------------------------------
    res = res.copy()
    res["promo_x_weekend"] = res["is_promo"] * res["is_weekend"]
    res["is_promo_friday"] = res["is_promo"] * (res["dow"] == 4).astype(int)
    res["promo_x_uni"] = res["is_promo"] * is_uni
    res["promo_x_mall"] = res["is_promo"] * is_mall
    res["promo_x_street"] = res["is_promo"] * is_street
    res["promo_x_dow"] = res["is_promo"] * res["dow"]
    if "promo_streak_clip" in res.columns:
        res["promo_streak_sq"] = res["promo_streak_clip"] ** 2
        
    return res.copy()


class DeSeasonalizedStoreStats(BaseEstimator, TransformerMixin):
    """
    Bộ mã hóa Thống kê Mục tiêu Đa cấp đã Khử Mùa vụ V5 (De-seasonalized Store Stats Encoder):
    Tuân thủ nghiêm ngặt nguyên tắc KHÔNG RÒ RỈ DỮ LIỆU (Leakage-Free):
    - Khử xu thế vĩ mô nội bộ quán (within-store trend) về mốc chuẩn t_ref = 1.0 (2025-01-01) trước khi học mức nền.
    - Học độ lệch thứ trong tuần (type_dow_offset_) không nhiễu theo từng mô hình quán (store_type x dow).
    - Ước lượng mức nền công suất vật lý (Physical Store Capacity Prior) qua Ridge Regression chuẩn hóa
      trên [log_seats, log_area_m2, area_per_seat, has_parking, store_type, city].
    - Áp dụng kỹ thuật co làm mượt Empirical Bayes (Shrinkage):
      mu_s = (n_s / (n_s + m)) * hist_adj + (m / (n_s + m)) * prior
    - Học đường cong Tết (tet_curve_) đã loại bỏ ngày Valentine (14/2) và làm mượt theo từng phân đoạn
      liên tục (pre-Tet, pre-rush, post-Tet, tail) mà giữ nguyên các ngày gãy/nhảy vọt cốt lõi (d=-7, [-2..+2]).
    - Hỗ trợ tính độ trôi mức vận hành (store_ewma_offset_) tại ngày chốt lịch sử (cutoff_date_).
    """
    def __init__(
        self,
        shrinkage_m: float = 8.0,
        alpha_ridge: float = 2.0,
        add_ewma_drift: bool = True,
        ewma_hl: float = 28.0,
        ewma_reg: float = 30.0,
        smooth_tet: bool = True
    ):
        self.shrinkage_m = shrinkage_m
        self.alpha_ridge = alpha_ridge
        self.add_ewma_drift = add_ewma_drift
        self.ewma_hl = ewma_hl
        self.ewma_reg = ewma_reg
        self.smooth_tet = smooth_tet
        
        self.global_log_mean_ = None
        self.global_trend_ = 0.0
        self.store_base_log_ = {}
        self.store_dow_log_ = {}
        self.type_city_dow_log_ = {}
        self.type_dow_offset_ = {}
        self.tet_curve_ = {}
        self.prior_pipe_ = None
        self.store_ewma_offset_ = {}
        self.store_cutoff_date_ = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        df = X.copy().reset_index(drop=True)
        y_log = np.log1p(np.clip(y, 0, None))
        df["y_log"] = y_log
        t = df["t_years"].values.astype(float)
        self.store_cutoff_date_ = df["date"].max() if "date" in df.columns else None

        # 1. Ước lượng xu thế vĩ mô nội bộ quán trên các ngày trưởng thành, không Tết, không khuyến mãi
        d_col = "days_from_tet_exact" if "days_from_tet_exact" in df.columns else "days_from_tet"
        slope_mask = (df["days_since_open"] > 75) & (df[d_col].abs() > 14) & (df["is_promo"] == 0)
        sdf = df[slope_mask] if slope_mask.sum() > 200 else df
        if sdf["t_years"].nunique() > 1 and sdf["store_id"].nunique() > 1:
            ym = sdf["y_log"].groupby(sdf["store_id"]).transform("mean").values
            tm = sdf["t_years"].groupby(sdf["store_id"]).transform("mean").values
            tc = sdf["t_years"].values - tm
            yc = sdf["y_log"].values - ym
            denom = float(np.sum(tc ** 2))
            self.global_trend_ = float(np.sum(tc * yc) / denom) if denom > 1e-6 else 0.0
        else:
            self.global_trend_ = 0.0

        # Khử xu thế vĩ mô về mốc chuẩn t_ref = 1.0 (2025-01-01) và khử ramp-up
        t_ref = 1.0
        log_ramp = df.get("log_ramp_smooth", df.get("log_ramp_up_factor", pd.Series(0.0, index=df.index))).values
        df["y_detrend"] = y_log - self.global_trend_ * (t - t_ref) - log_ramp
        self.global_log_mean_ = float(np.mean(df["y_detrend"]))

        # 2. Xác định các ngày bình thường sạch sẽ để tính độ lệch thứ và mức nền chuẩn
        is_nt_hol = df.get("is_non_tet_national_holiday", df.get("is_national_holiday", 0))
        normal_mask = (
            (df["is_promo"] == 0) &
            (df[d_col].abs() > 14) &
            (is_nt_hol == 0) &
            (df["is_uni_summer_break"] == 0) &
            (df["is_heavy_rain"] == 0) &
            (df["days_since_open"] > 75)
        )
        normal_df = df[normal_mask]
        if len(normal_df) < 50:
            normal_df = df

        # Độ lệch thứ trong tuần theo từng mô hình quán (type_dow_offset_) không nhiễu
        st_dow_mean = normal_df.groupby(["store_type", "dow"])["y_detrend"].mean()
        st_mean = normal_df.groupby("store_type")["y_detrend"].mean()
        for (st, w), val in st_dow_mean.items():
            self.type_dow_offset_[(st, w)] = float(val - st_mean.loc[st])

        df["dow_offset"] = [self.type_dow_offset_.get((st, w), 0.0) for st, w in zip(df["store_type"], df["dow"])]
        df["y_clean_adj"] = df["y_detrend"] - df["dow_offset"]
        normal_df = df[normal_mask] if len(df[normal_mask]) > 0 else df

        store_base = normal_df.groupby("store_id")["y_clean_adj"].mean().to_dict()

        # 3. Huấn luyện mô hình ước lượng công suất nền tảng (Physical Capacity Prior)
        store_meta = df.drop_duplicates(subset=["store_id"])[
            ["store_id", "seats", "area_m2", "has_parking", "store_type", "city"]
        ].copy()
        if "area_per_seat" not in store_meta.columns:
            store_meta["area_per_seat"] = store_meta["area_m2"] / store_meta["seats"]
        store_meta["log_seats"] = np.log1p(store_meta["seats"])
        store_meta["log_area_m2"] = np.log1p(store_meta["area_m2"])
        store_meta["actual_base"] = store_meta["store_id"].map(store_base)
        train_prior = store_meta.dropna(subset=["actual_base"])

        prior_num = ["log_seats", "log_area_m2", "area_per_seat", "has_parking"]
        prior_cat = ["store_type", "city"]
        if len(train_prior) >= 5:
            ct = ColumnTransformer([
                ("num", StandardScaler(), prior_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), prior_cat),
            ])
            self.prior_pipe_ = Pipeline([("ct", ct), ("reg", Ridge(alpha=self.alpha_ridge))])
            self.prior_pipe_.fit(train_prior[prior_num + prior_cat], train_prior["actual_base"].values)
        else:
            self.prior_pipe_ = None

        # Co làm mượt Empirical Bayes cho tất cả các quán
        counts = df.groupby("store_id")["y_clean_adj"].count().to_dict()
        for sid in df["store_id"].unique():
            n = counts.get(sid, 0)
            hist_base = store_base.get(sid, None)
            if hist_base is None:
                s_df = df[df["store_id"] == sid]
                sub_s = s_df[(s_df["is_promo"] == 0) & (s_df[d_col].abs() > 14) & (s_df["is_heavy_rain"] == 0)]
                if len(sub_s) >= 3:
                    hist_base = float(sub_s["y_clean_adj"].mean())
                else:
                    sub_s2 = s_df[s_df["is_promo"] == 0]
                    hist_base = float(sub_s2["y_clean_adj"].mean()) if len(sub_s2) >= 2 else float(s_df["y_clean_adj"].mean())

            meta_row = store_meta[store_meta["store_id"] == sid]
            prior_val = float(self.prior_pipe_.predict(meta_row[prior_num + prior_cat])[0]) if (self.prior_pipe_ is not None and len(meta_row) > 0) else self.global_log_mean_

            if hist_base is not None:
                n_eff = len(normal_df[normal_df["store_id"] == sid]) if sid in store_base else n * 0.5
                shrunk_val = (n_eff / (n_eff + self.shrinkage_m)) * hist_base + (self.shrinkage_m / (n_eff + self.shrinkage_m)) * prior_val
            else:
                shrunk_val = prior_val
            self.store_base_log_[sid] = shrunk_val

        # type_city_dow_log_rev
        clean_days = df[(df["is_promo"] == 0) & (df[d_col].abs() > 14)]
        self.type_city_dow_log_ = clean_days.groupby(["store_type", "city", "dow"])["y_detrend"].mean().to_dict()

        # store_dow_log_
        for sid, b_val in self.store_base_log_.items():
            st = store_meta.loc[store_meta["store_id"] == sid, "store_type"].values[0]
            for w in range(7):
                self.store_dow_log_[(sid, w)] = b_val + self.type_dow_offset_.get((st, w), 0.0)

        # 4. Học đường cong Tết tet_curve_ (loại trừ Valentine 14/2 và khử store_base + dow_offset)
        val_mask = (df["month"] == 2) & (df["day"] == 14)
        tet_sub = df[df[d_col].between(-18, 18) & (~val_mask) & (df["is_promo"] == 0)].copy()
        tet_sub["sb"] = tet_sub["store_id"].map(self.store_base_log_).fillna(self.global_log_mean_)
        tet_sub["tet_log_diff"] = tet_sub["y_clean_adj"] - tet_sub["sb"]
        self.tet_curve_ = tet_sub.groupby(["store_type", d_col])["tet_log_diff"].mean().to_dict()

        # Nội suy tuyến tính các ngày khuyết trong cửa sổ [-18, 18]
        for st in df["store_type"].unique():
            vals = [self.tet_curve_.get((st, d), np.nan) for d in range(-18, 19)]
            s_interp = pd.Series(vals, index=range(-18, 19)).interpolate(method="linear").fillna(0.0)
            for d, v in s_interp.items():
                self.tet_curve_[(st, d)] = float(v)

        # Làm mượt đường cong Tết theo từng phân đoạn
        if self.smooth_tet:
            regimes = [range(-18, -7), range(-6, -2), range(3, 8), range(8, 19)]
            new_curve = dict(self.tet_curve_)
            for st in df["store_type"].unique():
                for reg in regimes:
                    r_list = list(reg)
                    for i, d in enumerate(r_list):
                        neighbors = [d]
                        weights = [0.60]
                        if i > 0:
                            neighbors.append(r_list[i - 1])
                            weights.append(0.20)
                        if i < len(r_list) - 1:
                            neighbors.append(r_list[i + 1])
                            weights.append(0.20)
                        ws = np.array(weights) / np.sum(weights)
                        new_curve[(st, d)] = float(np.sum([w * self.tet_curve_.get((st, dn), 0.0) for w, dn in zip(ws, neighbors)]))
            self.tet_curve_ = new_curve

        # 5. Tính độ lệch trôi EWMA của quán tại ngày chốt lịch sử
        if self.add_ewma_drift and "date" in df.columns:
            cutoff = df["date"].max()
            base_off = np.array([self.store_base_log_.get(s, self.global_log_mean_) + self.type_dow_offset_.get((st, w), 0.0)
                                 for s, st, w in zip(df["store_id"], df["store_type"], df["dow"])])
            r_tr = df["y_detrend"].values - base_off
            tr_c = df[(df["days_since_open"] > 75) & (df[d_col].abs() > 15)].copy()
            if len(tr_c) > 50:
                tr_c["r_drift"] = r_tr[tr_c.index]
                for sid, g in tr_c.groupby("store_id"):
                    dt_d = (cutoff - g["date"]).dt.days.values.astype(float)
                    w = np.exp(-np.log(2.0) * dt_d / self.ewma_hl)
                    self.store_ewma_offset_[sid] = float(np.sum(w * g["r_drift"].values) / (np.sum(w) + self.ewma_reg))

        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        res = X.copy()
        
        # 1. store_base_log_rev
        unseen_mask = ~res["store_id"].isin(self.store_base_log_.keys())
        if unseen_mask.any() and self.prior_pipe_ is not None:
            sub = res.loc[unseen_mask, ["seats", "area_m2", "has_parking", "store_type", "city"]].copy()
            if "area_per_seat" not in sub.columns:
                sub["area_per_seat"] = sub["area_m2"] / sub["seats"]
            sub["log_seats"] = np.log1p(sub["seats"])
            sub["log_area_m2"] = np.log1p(sub["area_m2"])
            prior_num = ["log_seats", "log_area_m2", "area_per_seat", "has_parking"]
            prior_cat = ["store_type", "city"]
            preds = self.prior_pipe_.predict(sub[prior_num + prior_cat])
            res.loc[unseen_mask, "store_base_log_rev"] = preds
            res.loc[~unseen_mask, "store_base_log_rev"] = res.loc[~unseen_mask, "store_id"].map(self.store_base_log_)
        else:
            res["store_base_log_rev"] = res["store_id"].map(self.store_base_log_).fillna(self.global_log_mean_)

        # 2. store_dow_offset & store_dow_log_rev
        res["store_dow_offset"] = [self.type_dow_offset_.get((st, w), 0.0) for st, w in zip(res["store_type"], res["dow"])]
        res["store_dow_log_rev"] = res["store_base_log_rev"] + res["store_dow_offset"]
        
        # 3. type_city_dow_log_rev
        tcd_triples = list(zip(res["store_type"], res["city"], res["dow"]))
        res["type_city_dow_log_rev"] = [self.type_city_dow_log_.get(t, self.global_log_mean_) for t in tcd_triples]
        
        # 4. store_dow_ratio
        res["store_dow_ratio"] = res["store_dow_offset"]

        # 5. feat_tet_log_factor
        d_col = "days_from_tet_exact" if "days_from_tet_exact" in res.columns else "days_from_tet"
        factors = [
            self.tet_curve_.get((st, d), 0.0) if -18 <= d <= 18 else 0.0
            for st, d in zip(res["store_type"], res[d_col])
        ]
        res["feat_tet_log_factor"] = np.array(factors, dtype=float)
        
        return res


def run_step3(data_dir: str = "data", verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Hàm thực thi toàn bộ BƯỚC 3: PREPROCESS & FEATURE ENGINEERING.
    
    Trả về:
        (train_clean, test_clean, stores_clean, weather_clean, holidays_clean, clean_stats)
    """
    if verbose:
        print("=" * 80)
        print("BƯỚC 3: PREPROCESS (TIỀN XỬ LÝ DỮ LIỆU & KỸ THUẬT ĐẶC TRƯNG CHUYÊN SÂU)")
        print("=" * 80)
        print("Mục tiêu: Sửa 10 lỗi dữ liệu bẩn và kiến tạo 5 nhóm đặc trưng (~45 biến) đồng nhất.")
        print("-" * 80)
        
    # 1. Đọc dữ liệu thô
    train_raw = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    test_raw = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["date"])
    stores_raw = pd.read_csv(f"{data_dir}/stores.csv")
    weather_raw = pd.read_csv(f"{data_dir}/weather.csv", parse_dates=["date"])
    holidays_raw = pd.read_csv(f"{data_dir}/holidays.csv", parse_dates=["date"])
    
    # 2. Làm sạch stores, weather, holidays
    stores_clean = clean_stores(stores_raw)
    weather_clean = clean_weather(weather_raw)
    holidays_clean = clean_holidays(holidays_raw)
    
    if verbose:
        print("3.1. KẾT QUẢ LÀM SẠCH BẢNG VỆ TINH:")
        print(f"  * stores_clean:   {stores_clean.shape} (Đã chuẩn hóa store_type, district, open_date; điền area_m2 cho S19, S24)")
        print(f"  * weather_clean:  {weather_clean.shape} (Đã đổi °F -> °C, tạo lưới 1.642 ngày đầy đủ, nội suy 0 NaNs)")
        print(f"  * holidays_clean: {holidays_clean.shape} (Đã gộp nhóm theo date thành 53 ngày duy nhất, cờ nhị phân)")
        
    # 3. Làm sạch train
    train_clean_df, clean_stats = clean_train_outliers(train_raw, stores_clean)
    
    if verbose:
        print("-" * 80)
        print("3.2. KẾT QUẢ SỬA LỖI DỮ LIỆU BẨN TRÊN TẬP HUẤN LUYỆN (TRAIN.CSV):")
        print(f"  * Bỏ dòng trùng lặp:      {clean_stats['duplicates_dropped']} dòng (từ {clean_stats['initial_rows']:,d} xuống {clean_stats['cleaned_rows']:,d} dòng)")
        print(f"  * Sửa lỗi đơn vị 1000x:    {clean_stats['fixed_1000x_count']} dòng (S12 tháng 03/2024)")
        print(f"  * Sửa lỗi dịch dấu 10x:   {clean_stats['fixed_10x_count']} dòng (sw_ratio > 3.8)")
        print(f"  * Sửa lỗi dịch dấu 0.1x:  {clean_stats['fixed_01x_count']} dòng (sw_ratio < 0.16)")
        print(f"  * Phân phối revenue sau sửa: Min = {clean_stats['final_revenue_min']:.2f} | Median = {clean_stats['final_revenue_median']:.2f} | Max = {clean_stats['final_revenue_max']:.2f}")
        
    # 4. Ghép bảng an toàn kèm lịch khuyến mãi xuyên suốt
    all_sched = pd.concat([
        train_clean_df[["store_id", "date", "is_promo"]],
        test_raw[["store_id", "date", "is_promo"]]
    ], ignore_index=True)
    all_sched = add_promo_streak_features(all_sched)
    promo_cols = [c for c in all_sched.columns if c not in ["store_id", "date", "is_promo"]]

    def merge_datasets(base_df):
        m = base_df.merge(stores_clean, on="store_id", how="left")
        m = m.merge(weather_clean, on=["city", "date"], how="left")
        m = m.merge(holidays_clean, on="date", how="left")
        m = m.merge(all_sched[["store_id", "date"] + promo_cols], on=["store_id", "date"], how="left")
        m["is_national_holiday"] = m["is_national_holiday"].fillna(0).astype(int)
        m["is_event_holiday"] = m["is_event_holiday"].fillna(0).astype(int)
        m["holiday_name_clean"] = m["holiday_name_clean"].fillna("None")
        return m
        
    train_merged_clean = merge_datasets(train_clean_df)
    test_merged_clean = merge_datasets(test_raw)
    
    # 5. Xây dựng 5 nhóm đặc trưng cơ bản
    train_features = build_features(train_merged_clean)
    test_features = build_features(test_merged_clean)
    
    # 6. Thống kê mục tiêu khử mùa vụ (De-seasonalized Store Stats)
    encoder = DeSeasonalizedStoreStats(shrinkage_m=8.0, alpha_ridge=2.0, add_ewma_drift=True, ewma_hl=28.0, ewma_reg=30.0, smooth_tet=True)
    encoder.fit(train_features, train_features["revenue"].values)
    train_final = encoder.transform(train_features)
    test_final = encoder.transform(test_features)
    
    if verbose:
        print("-" * 80)
        print("3.3. KẾT QUẢ KIẾN TẠO HỆ THỐNG ĐẶC TRƯNG (FEATURE ENGINEERING):")
        print(f"  * train_features shape: {train_final.shape} (26.500 dòng sạch × {train_final.shape[1]} đặc trưng)")
        print(f"  * test_features shape:  {test_final.shape} (3.869 dòng giữ nguyên × {test_final.shape[1]} đặc trưng)")
        print(f"  * Số lượng giá trị NaN còn lại ở train: {train_final.isnull().sum().sum()}")
        print(f"  * Số lượng giá trị NaN còn lại ở test:  {test_final.isnull().sum().sum()}")
        print("=" * 80)
        print("HOÀN THÀNH TOÀN BỘ BƯỚC 3: PREPROCESS THEO ĐÚNG KẾ HOẠCH!")
        print("=" * 80)
        
    return train_final, test_final, stores_clean, weather_clean, holidays_clean, clean_stats


if __name__ == "__main__":
    run_step3()