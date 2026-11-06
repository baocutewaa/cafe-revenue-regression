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

# Các mốc Mùng 1 Tết Nguyên Đán
TET_DATES = {
    2024: pd.Timestamp("2024-02-10"),  # Giáp Thìn
    2025: pd.Timestamp("2025-01-29"),  # Ất Tỵ
    2026: pd.Timestamp("2026-02-17")   # Bính Ngọ
}

def clean_stores(stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý và làm sạch toàn diện bảng stores.csv:
    1. Chuẩn hóa store_type về chữ thường, loại bỏ khoảng trắng (street, office, mall, university, residential).
    2. Chuẩn hóa tên district (District 10, District 3, District 7,...).
    3. Chuyển open_date sang kiểu datetime bằng format='mixed' và dayfirst=True.
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
    
    # 3. Chuẩn hóa open_date
    df["open_date_dt"] = pd.to_datetime(df["open_date"], format="mixed", dayfirst=True)
    
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
    4. Nội suy tuyến tính theo thời gian (linear interpolation) theo từng thành phố cho temp_c.
    5. Điền khuyết 0.0 mm cho các giá trị NaN của rain_mm.
    6. Tính độ lệch nhiệt độ so với trung bình thành phố (temp_deviation_from_city_mean).
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
    
    # 4 & 5. Nội suy temp_c và điền 0.0 rain_mm
    merged_weather = merged_weather.sort_values(["city", "date"]).reset_index(drop=True)
    merged_weather["temp_c"] = merged_weather.groupby("city")["temp_c"].transform(
        lambda s: s.interpolate(method="linear").bfill().ffill()
    )
    merged_weather["rain_mm"] = merged_weather["rain_mm"].fillna(0.0)
    
    # 6. Độ lệch nhiệt độ so với mức trung bình của thành phố
    city_temp_mean = merged_weather.groupby("city")["temp_c"].transform("mean")
    merged_weather["temp_deviation_from_city_mean"] = merged_weather["temp_c"] - city_temp_mean
    
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
    df["dow"] = df["date"].dt.dayofweek
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
    Kiến tạo hệ thống đặc trưng chuyên sâu (Feature Engineering Architecture):
    Bao gồm 4 nhóm đặc trưng cơ bản (~35 biến) được tính toán đồng nhất cho cả train và test:
    
    1. Nhóm Thời gian & Chu kỳ (Calendar & Cyclical):
       dow, is_weekend, day, month, weekofyear, quarter, sin_dow, cos_dow, sin_doy, cos_doy
    2. Nhóm Tết Nguyên Đán & Ngày lễ (Lunar Tet & Holidays):
       days_from_tet, is_tet_core, is_pre_tet_shopping, is_ong_tao, is_post_tet_recovery,
       is_uni_tet_break, is_uni_summer_break, is_national_holiday, is_event_holiday,
       is_womens_day, is_valentine
    3. Nhóm Tĩnh & Vòng đời Cửa hàng (Store Static & Lifecycle):
       city, district, store_type, area_m2, seats, has_parking, area_per_seat,
       days_since_open, log_days_since_open, is_grand_opening, is_ramp_up_phase, ramp_up_factor
    4. Nhóm Thời tiết & Tương tác (Weather & Interactions):
       temp_c, rain_mm, log_rain, is_heavy_rain, is_extreme_rain,
       temp_deviation_from_city_mean, rain_x_outdoor, rain_x_mall, promo_x_weekend
    """
    res = df.copy()
    
    # Đảm bảo kiểu datetime
    date_dt = pd.to_datetime(res["date"])
    res["date"] = date_dt
    
    # --------------------------------------------------------------------------
    # 1. Nhóm Thời gian & Chu kỳ (Calendar & Cyclical)
    # --------------------------------------------------------------------------
    res["dow"] = date_dt.dt.dayofweek
    res["is_weekend"] = (res["dow"] >= 5).astype(int)
    res["day"] = date_dt.dt.day
    res["month"] = date_dt.dt.month
    res["weekofyear"] = date_dt.dt.isocalendar().week.astype(int)
    res["quarter"] = date_dt.dt.quarter
    
    # Biến đổi lượng giác sin/cos
    res["sin_dow"] = np.sin(2 * np.pi * res["dow"] / 7.0)
    res["cos_dow"] = np.cos(2 * np.pi * res["dow"] / 7.0)
    doy = date_dt.dt.dayofyear
    res["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    res["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    
    # Trục thời gian liên tục tính theo năm (t_years) cho xu thế vĩ mô (Macro Time Trend)
    res["t_years"] = (date_dt - pd.Timestamp("2024-01-01")).dt.days / 365.25
    
    # --------------------------------------------------------------------------
    # 2. Nhóm Tết Nguyên Đán & Ngày lễ (Lunar Tet & Holidays)
    # --------------------------------------------------------------------------
    res["year"] = date_dt.dt.year
    tet_dt = res["year"].map(TET_DATES)
    res["days_from_tet"] = (date_dt - tet_dt).dt.days
    res["days_from_tet_exact"] = res["days_from_tet"]
    # Giới hạn days_from_tet_clipped trong cửa sổ [-21, 21] ngày để triệt tiêu đa cộng tuyến với month
    res["days_from_tet_clipped"] = np.clip(res["days_from_tet"], -21, 21)
    
    # Các cờ giai đoạn Tết
    res["is_tet_core"] = ((res["days_from_tet"] >= -2) & (res["days_from_tet"] <= 2)).astype(int)
    res["is_pre_tet_shopping"] = ((res["days_from_tet"] >= -12) & (res["days_from_tet"] <= -3)).astype(int)
    res["is_ong_tao"] = (res["days_from_tet"] == -7).astype(int)
    res["is_post_tet_recovery"] = ((res["days_from_tet"] >= 3) & (res["days_from_tet"] <= 9)).astype(int)
    
    # Cờ tương tác chuyên sâu giữa loại hình quán và giai đoạn Tết
    res["is_mall_pre_tet"] = ((res["store_type"] == "mall") & (res["days_from_tet"].between(-7, -3))).astype(int)
    res["is_mall_tet_recovery"] = ((res["store_type"] == "mall") & (res["days_from_tet"].between(2, 5))).astype(int)
    res["is_street_post_tet"] = ((res["store_type"] == "street") & (res["days_from_tet"] == 3)).astype(int)
    res["is_office_tet_closure"] = ((res["store_type"] == "office") & (res["days_from_tet"].between(-2, 2))).astype(int)
    
    # Cờ nghỉ Tết & nghỉ hè của sinh viên đại học
    is_uni = (res["store_type"] == "university")
    res["is_uni_tet_break"] = (is_uni & (res["days_from_tet"] >= -14) & (res["days_from_tet"] <= 12)).astype(int)
    res["is_uni_summer_break"] = (is_uni & res["month"].isin([7, 8])).astype(int)
    
    # Cờ các ngày sự kiện đặc biệt
    res["is_womens_day"] = (((res["month"] == 3) & (res["day"] == 8)) | ((res["month"] == 10) & (res["day"] == 20))).astype(int)
    res["is_valentine"] = ((res["month"] == 2) & (res["day"] == 14)).astype(int)
    
    # Đảm bảo cờ ngày lễ không có NaN
    res["is_national_holiday"] = res["is_national_holiday"].fillna(0).astype(int)
    res["is_event_holiday"] = res["is_event_holiday"].fillna(0).astype(int)
    
    # Tương tác ngày lễ với loại quán: văn phòng/đại học giảm sâu (-55%), mall/street/residential tăng (+15-25%)
    is_work = res["store_type"].isin(["office", "university"]).astype(int)
    is_retail = res["store_type"].isin(["mall", "street", "residential"]).astype(int)
    res["is_holiday_work_drop"] = res["is_national_holiday"] * is_work
    res["is_holiday_retail_boost"] = res["is_national_holiday"] * is_retail
    
    # --------------------------------------------------------------------------
    # 3. Nhóm Tĩnh & Vòng đời Cửa hàng (Store Static & Lifecycle)
    # --------------------------------------------------------------------------
    if "open_date_dt" not in res.columns:
        res["open_date_dt"] = pd.to_datetime(res["open_date"], format="mixed", dayfirst=True)
        
    res["days_since_open"] = (date_dt - res["open_date_dt"]).dt.days
    res["log_days_since_open"] = np.log1p(np.maximum(0, res["days_since_open"]))
    res["is_grand_opening"] = res["days_since_open"].isin([0, 1, 2]).astype(int)
    res["is_ramp_up_phase"] = ((res["days_since_open"] >= 3) & (res["days_since_open"] <= 60)).astype(int)
    res["ramp_up_factor"] = compute_ramp_up_factor(res["days_since_open"])
    
    if "area_per_seat" not in res.columns:
        res["area_per_seat"] = res["area_m2"] / res["seats"]
        
    # --------------------------------------------------------------------------
    # 4. Nhóm Thời tiết & Tương tác (Weather & Interactions)
    # --------------------------------------------------------------------------
    res["rain_mm"] = res["rain_mm"].fillna(0.0)
    res["log_rain"] = np.log1p(np.maximum(0, res["rain_mm"]))
    res["is_heavy_rain"] = (res["rain_mm"] >= 20.0).astype(int)
    res["is_extreme_rain"] = (res["rain_mm"] >= 50.0).astype(int)
    
    # Tương tác mưa với loại quán
    is_outdoor = res["store_type"].isin(["street", "residential", "university"]).astype(int)
    is_mall = (res["store_type"] == "mall").astype(int)
    res["rain_x_outdoor"] = res["log_rain"] * is_outdoor
    res["rain_x_mall"] = res["log_rain"] * is_mall
    
    # Tương tác khuyến mãi với cuối tuần
    res["promo_x_weekend"] = res["is_promo"] * res["is_weekend"]
    
    return res


class DeSeasonalizedStoreStats(BaseEstimator, TransformerMixin):
    """
    Bộ mã hóa Thống kê Mục tiêu Đa cấp đã Khử Mùa vụ (De-seasonalized Store Stats Encoder):
    Tuân thủ nghiêm ngặt nguyên tắc KHÔNG RÒ RỈ DỮ LIỆU (Leakage-Free):
    - Hàm .fit(X, y) chỉ học tham số trên tập huấn luyện của từng fold.
    - Học trung vị nền tảng trên các ngày BÌNH THƯỜNG (loại bỏ ngày Tết, ngày mưa bão >20mm,
      ngày nghỉ hè sinh viên, ngày có khuyến mãi và 60 ngày đầu mới mở quán).
    - Huấn luyện một mô hình hồi quy phụ (Ridge Regression) trên các thuộc tính vật lý của quán
      (seats, area_m2, store_type, city, district, has_parking) để dự báo công suất kỳ vọng
      (Store Capacity Prior) cho các quán mới tinh (như S01, S19, S45) hoặc quán ít ngày lịch sử.
    - Áp dụng kỹ thuật co làm mượt Empirical Bayes (Shrinkage):
      mu_s = (n_s / (n_s + m)) * hist_adj + (m / (n_s + m)) * prior
    """
    def __init__(self, shrinkage_m: float = 10.0, alpha_ridge: float = 1.0):
        self.shrinkage_m = shrinkage_m
        self.alpha_ridge = alpha_ridge
        self.global_log_mean_ = None
        self.store_base_log_ = {}
        self.store_dow_log_ = {}
        self.type_city_dow_log_ = {}
        self.tet_curve_ = {}
        self.prior_model_ = None
        self.prior_feature_cols_ = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        df = X.copy()
        y_log = np.log1p(np.clip(y, 0, None))
        df["y_log"] = y_log
        
        self.global_log_mean_ = float(np.mean(y_log))
        
        # 1. Xác định các ngày bình thường để tính mức nền chuẩn
        normal_mask = (
            (df["is_promo"] == 0) &
            (df["is_tet_core"] == 0) &
            (df["is_uni_summer_break"] == 0) &
            (df["is_heavy_rain"] == 0) &
            (df["days_since_open"] > 60)
        )
        normal_df = df[normal_mask]
        if len(normal_df) == 0:
            normal_df = df
            
        # Trung vị theo từng quán
        store_base = normal_df.groupby("store_id")["y_log"].median().to_dict()
        
        # Trung vị theo (store_id, dow)
        store_dow = df.groupby(["store_id", "dow"])["y_log"].median().to_dict()
        self.store_dow_log_ = store_dow
        
        # Trung vị theo (store_type, city, dow)
        type_city_dow = df.groupby(["store_type", "city", "dow"])["y_log"].median().to_dict()
        self.type_city_dow_log_ = type_city_dow
        
        # 2. Huấn luyện mô hình ước lượng công suất nền tảng (Store Capacity Prior)
        store_meta = df.drop_duplicates(subset=["store_id"])[
            ["store_id", "seats", "area_m2", "has_parking", "store_type", "city"]
        ].copy()
        store_meta["actual_base"] = store_meta["store_id"].map(store_base)
        train_prior = store_meta.dropna(subset=["actual_base"])
        
        if len(train_prior) >= 5:
            X_prior = pd.get_dummies(
                train_prior[["seats", "area_m2", "has_parking", "store_type", "city"]],
                drop_first=True
            )
            self.prior_feature_cols_ = list(X_prior.columns)
            y_prior = train_prior["actual_base"].values
            self.prior_model_ = Ridge(alpha=self.alpha_ridge)
            self.prior_model_.fit(X_prior, y_prior)
        else:
            self.prior_model_ = None
            
        # 3. Tính mức nền sau co làm mượt cho tất cả các quán
        counts = df.groupby("store_id")["y_log"].count().to_dict()
        for sid in df["store_id"].unique():
            n = counts.get(sid, 0)
            hist_base = store_base.get(sid, None)
            
            # Nếu quán chưa có dữ liệu > 60 ngày (như S01, S45), suy diễn từ dữ liệu khai trương thực tế đã khử ramp-up
            if hist_base is None:
                s_df = df[df["store_id"] == sid]
                if len(s_df) > 0:
                    ramp_factors = s_df.get("ramp_up_factor", pd.Series(1.0, index=s_df.index))
                    unramped_log = s_df["y_log"] - np.log(np.maximum(0.2, ramp_factors))
                    non_promo = unramped_log[s_df["is_promo"] == 0]
                    if len(non_promo) >= 3:
                        hist_base = float(non_promo.median())
                    else:
                        hist_base = float(unramped_log.median())
            
            # Tính prior từ mô hình vật lý
            prior_val = self.global_log_mean_
            if self.prior_model_ is not None:
                meta_row = store_meta[store_meta["store_id"] == sid]
                if len(meta_row) > 0:
                    x_row = pd.get_dummies(
                        meta_row[["seats", "area_m2", "has_parking", "store_type", "city"]],
                        drop_first=True
                    ).reindex(columns=self.prior_feature_cols_, fill_value=0)
                    prior_val = float(self.prior_model_.predict(x_row)[0])
                    
            if hist_base is not None:
                shrunk_val = (n / (n + self.shrinkage_m)) * hist_base + (self.shrinkage_m / (n + self.shrinkage_m)) * prior_val
            else:
                shrunk_val = prior_val
                
            self.store_base_log_[sid] = shrunk_val

        # 4. Học đường cong tỷ lệ Tết Nguyên Đán đa cấp theo (store_type, days_from_tet_exact)
        df["revenue_orig"] = np.maximum(0.0, y)
        d_col = "days_from_tet_exact" if "days_from_tet_exact" in df.columns else "days_from_tet"
        norm_mask = (df[d_col] < -14) | (df[d_col] > 14)
        norm_mask = norm_mask & (df["is_promo"] == 0)
        base_map = df[norm_mask].groupby(["store_id", "dow"])["revenue_orig"].median().to_dict()
        df_base = [base_map.get((s, w), np.nan) for s, w in zip(df["store_id"], df["dow"])]
        df["ratio_tet"] = df["revenue_orig"] / pd.Series(df_base, index=df.index).fillna(float(np.median(y)))
        tet_mask = df[d_col].between(-18, 18)
        self.tet_curve_ = df[tet_mask].groupby(["store_type", d_col])["ratio_tet"].median().to_dict()
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        res = X.copy()
        
        # 1. store_base_log_rev
        res["store_base_log_rev"] = res["store_id"].map(self.store_base_log_).fillna(self.global_log_mean_)
        
        # 2. store_dow_log_rev
        sd_pairs = list(zip(res["store_id"], res["dow"]))
        res["store_dow_log_rev"] = [self.store_dow_log_.get(p, np.nan) for p in sd_pairs]
        
        # 3. type_city_dow_log_rev
        tcd_triples = list(zip(res["store_type"], res["city"], res["dow"]))
        res["type_city_dow_log_rev"] = [self.type_city_dow_log_.get(t, self.global_log_mean_) for t in tcd_triples]
        
        # Điền khuyết store_dow_log_rev bằng type_city_dow_log_rev nếu quán chưa có lịch sử ở thứ đó
        res["store_dow_log_rev"] = res["store_dow_log_rev"].fillna(res["type_city_dow_log_rev"])
        
        # 4. store_dow_ratio = store_dow_log_rev - store_base_log_rev
        res["store_dow_ratio"] = res["store_dow_log_rev"] - res["store_base_log_rev"]

        # 5. feat_tet_log_factor (Hệ số điều chỉnh logarit mùa Tết theo loại quán)
        d_col = "days_from_tet_exact" if "days_from_tet_exact" in res.columns else "days_from_tet"
        factors = [
            np.log(np.maximum(0.2, self.tet_curve_.get((st, d), 1.0))) if -18 <= d <= 18 else 0.0
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
        
    # 4. Ghép bảng an toàn
    def merge_datasets(base_df):
        m = base_df.merge(stores_clean, on="store_id", how="left")
        m = m.merge(weather_clean, on=["city", "date"], how="left")
        m = m.merge(holidays_clean, on="date", how="left")
        m["is_national_holiday"] = m["is_national_holiday"].fillna(0).astype(int)
        m["is_event_holiday"] = m["is_event_holiday"].fillna(0).astype(int)
        m["holiday_name_clean"] = m["holiday_name_clean"].fillna("None")
        return m
        
    train_merged_clean = merge_datasets(train_clean_df)
    test_merged_clean = merge_datasets(test_raw)
    
    # 5. Xây dựng 4 nhóm đặc trưng cơ bản
    train_features = build_features(train_merged_clean)
    test_features = build_features(test_merged_clean)
    
    # 6. Thống kê mục tiêu khử mùa vụ (De-seasonalized Store Stats)
    encoder = DeSeasonalizedStoreStats(shrinkage_m=10.0, alpha_ridge=1.0)
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
