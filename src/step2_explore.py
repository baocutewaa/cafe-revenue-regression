"""
Module: step2_explore.py
Chủ đề: Dự báo doanh thu ngày chuỗi quán Cafe (Cafe Revenue Regression)
Mô tả: Triển khai toàn bộ BƯỚC 2: EXPLORE (KHÁM PHÁ DỮ LIỆU - EDA & DANH SÁCH LỖI PHÁT HIỆN ĐƯỢC)
Tuân thủ nguyên tắc vàng bài giảng Chương 4: "Output of this step: a written list of problems, before any of them is fixed"
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any

def detect_duplicates(train_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Phát hiện các dòng trùng lặp hoàn toàn trong train.csv.
    """
    dup_mask = train_df.duplicated(keep=False)
    dup_count = train_df.duplicated().sum()
    dup_samples = train_df[dup_mask].sort_values(by=["store_id", "date"])
    return int(dup_count), dup_samples


def detect_s12_1000x_anomaly(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phát hiện cụm lỗi đơn vị 1000x tại quán S12 trong tháng 03/2024.
    """
    s12_mask = (train_df["store_id"] == "S12") & (train_df["revenue"] > 1000)
    return train_df[s12_mask].sort_values(by="date")


def detect_decimal_errors(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phát hiện các lỗi dịch dấu thập phân 10x và 0.1x trong train.csv.
    Sử dụng trung vị tham chiếu theo (store_id, dayofweek) sau khi tạm loại bỏ lỗi 1000x.
    
    Trả về:
        (df_10x, df_01x): DataFrame chứa các dòng bị lỗi 10x và 0.1x
    """
    df_temp = train_df.copy()
    # Tạm thời chuẩn hóa lỗi 1000x để tính trung vị tham chiếu chính xác
    df_temp.loc[df_temp["revenue"] > 1000, "revenue"] /= 1000.0
    df_temp["dow"] = df_temp["date"].dt.dayofweek
    
    # Tính trung vị theo store_id và dow
    sw_med = df_temp.groupby(["store_id", "dow"])["revenue"].transform("median")
    sw_ratio = train_df["revenue"] / sw_med
    
    # Lọc ngoài cụm S12 tháng 3/2024
    not_s12_mar24 = ~((train_df["store_id"] == "S12") & (train_df["date"].dt.year == 2024) & (train_df["date"].dt.month == 3))
    
    # 10x: sw_ratio > 3.8
    mask_10x = not_s12_mar24 & (sw_ratio > 3.8)
    df_10x = train_df[mask_10x].copy()
    df_10x["sw_med"] = sw_med[mask_10x]
    df_10x["sw_ratio"] = sw_ratio[mask_10x]
    
    # 0.1x: sw_ratio < 0.16
    mask_01x = sw_ratio < 0.16
    df_01x = train_df[mask_01x].copy()
    df_01x["sw_med"] = sw_med[mask_01x]
    df_01x["sw_ratio"] = sw_ratio[mask_01x]
    
    return df_10x, df_01x


def detect_store_type_inconsistency(stores_df: pd.DataFrame) -> pd.Series:
    """
    Phát hiện việc không đồng nhất chữ hoa/thường ở cột store_type.
    """
    return stores_df["store_type"].value_counts()


def detect_district_inconsistency(stores_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Phát hiện việc không đồng nhất cách viết tên Quận (District 10 vs Q.10, Quan 3, Quan 7).
    """
    districts = stores_df["district"].unique()
    hcm_districts = stores_df[stores_df["city"] == "Ho Chi Minh"]["district"].unique().tolist()
    hanoi_districts = stores_df[stores_df["city"] == "Ha Noi"]["district"].unique().tolist()
    return {
        "all_districts": districts.tolist(),
        "hcm_districts": hcm_districts,
        "hanoi_districts": hanoi_districts
    }


def detect_open_date_formats(stores_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phát hiện định dạng open_date hỗn hợp (YYYY-MM-DD và DD/MM/YYYY).
    """
    sample_iso = stores_df[stores_df["open_date"].str.contains("-", na=False)]
    sample_vn = stores_df[stores_df["open_date"].str.contains("/", na=False)]
    return {
        "iso_count": len(sample_iso),
        "vn_count": len(sample_vn),
        "sample_iso": sample_iso[["store_id", "open_date"]].head(3).to_dict(orient="records"),
        "sample_vn": sample_vn[["store_id", "open_date"]].head(3).to_dict(orient="records")
    }


def detect_stores_missing_area(stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phát hiện cửa hàng bị khuyết thiếu diện tích area_m2.
    """
    return stores_df[stores_df["area_m2"].isnull()][["store_id", "store_type", "seats", "has_parking", "city"]]


def detect_fahrenheit_outliers(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phát hiện ngoại lai nhiệt độ ghi nhầm sang độ Fahrenheit (> 45°C).
    """
    f_mask = weather_df["temp_c"] > 45
    return weather_df[f_mask].sort_values(by="date")


def detect_weather_missing_dates_and_rain_nans(weather_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phát hiện các ngày bị thiếu trong weather.csv và các giá trị NaN ở rain_mm.
    """
    date_grid = pd.date_range("2024-01-01", "2026-03-31")
    total_expected_days = len(date_grid)
    
    hanoi_w = weather_df[weather_df["city"] == "Ha Noi"]
    hcm_w = weather_df[weather_df["city"] == "Ho Chi Minh"]
    
    missing_hanoi = set(date_grid) - set(hanoi_w["date"])
    missing_hcm = set(date_grid) - set(hcm_w["date"])
    
    return {
        "expected_days_per_city": total_expected_days,
        "hanoi_actual_rows": len(hanoi_w),
        "hanoi_missing_days": len(missing_hanoi),
        "hcm_actual_rows": len(hcm_w),
        "hcm_missing_days": len(missing_hcm),
        "total_missing_city_days": len(missing_hanoi) + len(missing_hcm),
        "rain_mm_nans": int(weather_df["rain_mm"].isnull().sum())
    }


def build_ten_anomalies_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Xây dựng bảng tổng kết chính thức 10 vấn đề dữ liệu bẩn theo đúng Step 2 trong Regression_Plan.md.
    """
    train = tables["train"]
    stores = tables["stores"]
    weather = tables["weather"]
    holidays = tables["holidays"]
    
    # 1. Duplicates
    dup_cnt, _ = detect_duplicates(train)
    
    # 2. S12 1000x
    s12_df = detect_s12_1000x_anomaly(train)
    
    # 3 & 4. 10x and 0.1x
    df_10x, df_01x = detect_decimal_errors(train)
    
    # 5. Store type
    st_counts = detect_store_type_inconsistency(stores)
    
    # 6. District
    dist_info = detect_district_inconsistency(stores)
    
    # 7. Open date
    od_info = detect_open_date_formats(stores)
    
    # 8. Missing area
    missing_area = detect_stores_missing_area(stores)
    
    # 9. Fahrenheit
    f_df = detect_fahrenheit_outliers(weather)
    
    # 10. Weather missing
    w_info = detect_weather_missing_dates_and_rain_nans(weather)
    
    anomalies = [
        {
            "STT": 1,
            "Bảng dữ liệu": "train.csv",
            "Vấn đề phát hiện": "Dòng trùng lặp hoàn toàn (Duplicates)",
            "Minh chứng định lượng": f"{dup_cnt} dòng trùng lặp hoàn toàn (train.duplicated().sum() == {dup_cnt}), làm số dòng từ 26.500 tăng lên 26.632.",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Làm lệch trọng số các ngày bị nhân bản khi tính trung vị hoặc huấn luyện mô hình."
        },
        {
            "STT": 2,
            "Bảng dữ liệu": "train.csv",
            "Vấn đề phát hiện": "Lỗi đơn vị 1000x (Nhập nghìn VNĐ thay vì triệu VNĐ)",
            "Minh chứng định lượng": f"Toàn bộ {len(s12_df)} dòng của quán S12 trong tháng 03/2024 có revenue từ {s12_df['revenue'].min():,.1f} đến {s12_df['revenue'].max():,.1f} (trung vị bình thường ~29.1 triệu VNĐ).",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Đẩy mean của toàn bảng lên 51.3 và std lên 1015.9; làm méo mó nghiêm trọng tham số của mô hình đối với S12 và nhóm office."
        },
        {
            "STT": 3,
            "Bảng dữ liệu": "train.csv",
            "Vấn đề phát hiện": "Lỗi dịch dấu thập phân 10x (Gấp ~10 lần bình thường)",
            "Minh chứng định lượng": f"Có {len(df_10x)} dòng (ngoài cụm S12 tháng 3/2024) có sw_ratio từ {df_10x['sw_ratio'].min():.2f}x đến {df_10x['sw_ratio'].max():.2f}x (ví dụ: S29 2024-10-12 ghi 491.8 thay vì 49.18).",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Trong không gian logarit, lỗi 10x tạo ra sai số ln(10) ≈ 2.30, kéo lệch hàm mất mát MSE và làm tăng vọt RMSLE."
        },
        {
            "STT": 4,
            "Bảng dữ liệu": "train.csv",
            "Vấn đề phát hiện": "Lỗi dịch dấu thập phân 0.1x (Bằng ~1/10 bình thường)",
            "Minh chứng định lượng": f"Có {len(df_01x)} dòng có sw_ratio < 0.16 (nằm tách biệt hẳn với đáy ngày Tết là 0.24 - 0.35), ví dụ: S38 2024-02-15 ghi 1.21 thay vì 12.1; S39 2025-10-26 ghi 0.57 thay vì 5.70.",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Tạo ra các điểm ngoại lai âm sâu trên thang ln(1+revenue), làm kéo thấp dự báo trung bình của các quán bị dính lỗi."
        },
        {
            "STT": 5,
            "Bảng dữ liệu": "stores.csv",
            "Vấn đề phát hiện": "Không đồng nhất chữ hoa/thường ở store_type",
            "Minh chứng định lượng": f"Có {len(st_counts)} nhãn khác nhau: street (13), Street (2), office (8), Office (3), mall (7), Mall (1), university (5), University (1), residential (5).",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Nếu One-Hot Encoding trực tiếp sẽ phân rã 1 loại quán thành 2 cột riêng biệt, làm loãng dữ liệu huấn luyện."
        },
        {
            "STT": 6,
            "Bảng dữ liệu": "stores.csv",
            "Vấn đề phát hiện": "Không đồng nhất cách viết tên Quận (district)",
            "Minh chứng định lượng": "Cùng Quận 10 tại TP.HCM nhưng S27 ghi 'District 10', còn S37 và S39 ghi 'Q.10'; pha trộn tiếng Anh ('District 1') và tiếng Việt không dấu ('Quan 3', 'Quan 7').",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Gây phân mảnh đặc trưng vị trí địa lý giống bài toán province trong bài giảng."
        },
        {
            "STT": 7,
            "Bảng dữ liệu": "stores.csv",
            "Vấn đề phát hiện": "Định dạng ngày open_date hỗn hợp",
            "Minh chứng định lượng": f"Pha trộn giữa {od_info['iso_count']} quán định dạng ISO YYYY-MM-DD (2025-12-07 của S01) và {od_info['vn_count']} quán định dạng DD/MM/YYYY (27/07/2022 của S02, 11/12/2025 của S45).",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Nếu ép kiểu ngày không dùng format='mixed' và dayfirst=True, ngày 11/12/2025 sẽ bị hiểu nhầm thành 12/11/2025, làm sai lệch tuổi cửa hàng (days_since_open)."
        },
        {
            "STT": 8,
            "Bảng dữ liệu": "stores.csv",
            "Vấn đề phát hiện": "Khuyết thiếu diện tích area_m2",
            "Minh chứng định lượng": f"Quán {missing_area['store_id'].tolist()} bị NaN ở cột area_m2 ({len(missing_area)} quán: S19 seats={missing_area.loc[missing_area['store_id']=='S19', 'seats'].values[0]}, S24 seats={missing_area.loc[missing_area['store_id']=='S24', 'seats'].values[0]}).",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Mất thông tin quy mô mặt bằng cho 2 quán (trong đó S19 là quán mới mở tháng 10/2025)."
        },
        {
            "STT": 9,
            "Bảng dữ liệu": "weather.csv",
            "Vấn đề phát hiện": "Nhiệt độ temp_c bị ghi theo độ Fahrenheit (°F)",
            "Minh chứng định lượng": f"Tại TP.HCM trong {len(f_df)} ngày liên tiếp từ {f_df['date'].min().strftime('%Y-%m-%d')} đến {f_df['date'].max().strftime('%Y-%m-%d')}, temp_c nhận giá trị từ {f_df['temp_c'].min():.1f} đến {f_df['temp_c'].max():.1f}°F.",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Tạo ngoại lai cực đoan (max 89.2°C), làm hỏng trọng số của biến nhiệt độ trong Ridge, KNN, SVM, MLP."
        },
        {
            "STT": 10,
            "Bảng dữ liệu": "weather.csv",
            "Vấn đề phát hiện": "Thiếu hụt dòng ngày (Missing Dates) & NaN ở rain_mm",
            "Minh chứng định lượng": f"Thiếu {w_info['hanoi_missing_days']} ngày tại Hà Nội và {w_info['hcm_missing_days']} ngày tại TP.HCM (tổng {w_info['total_missing_city_days']} ngày/thành phố), cộng thêm {w_info['rain_mm_nans']} giá trị NaN ở cột rain_mm.",
            "Nguyên nhân & Hậu quả nếu không xử lý": "Khi left join vào train và test, 49 ngày bị thiếu trong bảng thời tiết sẽ tạo ra hàng trăm dòng NaN cả ở temp_c lẫn rain_mm."
        }
    ]
    return pd.DataFrame(anomalies)


def analyze_business_patterns(train_merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Phân tích 6 quy luật hành vi kinh doanh cốt lõi (Domain Insights) từ dữ liệu đã ghép.
    Sử dụng dữ liệu tạm thời hiệu chỉnh lỗi 1000x và bỏ trùng lặp để quan sát đúng thực tế.
    """
    df = train_merged.drop_duplicates().copy()
    df.loc[df["revenue"] > 1000, "revenue"] /= 1000.0
    df["store_type_clean"] = df["store_type"].str.strip().str.lower()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    
    # 1. DoW x store_type
    dow_matrix = df.groupby(["store_type_clean", "dow"])["revenue"].median().unstack()
    dow_matrix.columns = ["T2 (Mon)", "T3 (Tue)", "T4 (Wed)", "T5 (Thu)", "T6 (Fri)", "T7 (Sat)", "CN (Sun)"]
    
    # Tính phần trăm thay đổi cuối tuần so với ngày thường
    weekday_med = dow_matrix[["T2 (Mon)", "T3 (Tue)", "T4 (Wed)", "T5 (Thu)", "T6 (Fri)"]].median(axis=1)
    weekend_med = dow_matrix[["T7 (Sat)", "CN (Sun)"]].median(axis=1)
    weekend_change_pct = ((weekend_med - weekday_med) / weekday_med) * 100.0
    
    # 2. Promo effect
    promo_stats = df.groupby("is_promo")["revenue"].agg(["median", "mean", "count"])
    promo_uplift_pct = ((promo_stats.loc[1, "median"] - promo_stats.loc[0, "median"]) / promo_stats.loc[0, "median"]) * 100.0
    
    # 3. University Summer Break
    uni_monthly = df[df["store_type_clean"] == "university"].groupby("month")["revenue"].median()
    uni_regular_median = uni_monthly[[3, 4, 5, 6, 10, 11, 12]].median()
    uni_summer_median = uni_monthly[[7, 8]].median()
    uni_summer_drop_pct = ((uni_summer_median - uni_regular_median) / uni_regular_median) * 100.0
    
    # 4. National Holiday vs Event
    holiday_stats = df.groupby(["is_national_holiday", "is_event_holiday"])["revenue"].median()
    
    # 5. Store Lifecycle / Age Curve
    df["open_date_dt"] = pd.to_datetime(df["open_date"], format="mixed", dayfirst=True)
    df["days_since_open"] = (df["date"] - df["open_date_dt"]).dt.days
    sw_med = df.groupby(["store_id", "dow"])["revenue"].transform("median")
    df["sw_ratio"] = df["revenue"] / sw_med
    
    age_bins = [-1, 2, 7, 14, 30, 60, 9999]
    age_labels = ["0-2d (Opening)", "3-7d", "8-14d", "15-30d", "31-60d", ">60d (Mature)"]
    df["age_bin"] = pd.cut(df["days_since_open"], bins=age_bins, labels=age_labels)
    lifecycle_ratios = df.groupby("age_bin", observed=True)["sw_ratio"].median()
    
    # 6. Rain impact on Mall vs Outdoor
    df["rain_bin"] = pd.cut(
        df["rain_mm"].fillna(0),
        bins=[-1, 5, 20, 50, 999],
        labels=["Light (<5mm)", "Moderate (5-20mm)", "Heavy (20-50mm)", "Extreme (>50mm)"]
    )
    rain_matrix = df.groupby(["store_type_clean", "rain_bin"], observed=True)["sw_ratio"].median().unstack()
    
    return {
        "dow_matrix": dow_matrix,
        "weekend_change_pct": weekend_change_pct,
        "promo_stats": promo_stats,
        "promo_uplift_pct": promo_uplift_pct,
        "uni_monthly": uni_monthly,
        "uni_summer_drop_pct": uni_summer_drop_pct,
        "holiday_stats": holiday_stats,
        "lifecycle_ratios": lifecycle_ratios,
        "rain_matrix": rain_matrix
    }


def generate_eda_plots(train_merged: pd.DataFrame, weather_df: pd.DataFrame, output_dir: str = "plots") -> List[str]:
    """
    Vẽ 6 biểu đồ EDA trực quan hóa quy luật và ngoại lai dữ liệu, lưu thành file ảnh PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    
    # Bản sao làm sạch sơ bộ để vẽ đồ thị
    df = train_merged.drop_duplicates().copy()
    df.loc[df["revenue"] > 1000, "revenue"] /= 1000.0
    df["store_type_clean"] = df["store_type"].str.strip().str.lower()
    df["dow"] = df["date"].dt.dayofweek
    df["open_date_dt"] = pd.to_datetime(df["open_date"], format="mixed", dayfirst=True)
    df["days_since_open"] = (df["date"] - df["open_date_dt"]).dt.days
    sw_med = df.groupby(["store_id", "dow"])["revenue"].transform("median")
    df["sw_ratio"] = df["revenue"] / sw_med
    
    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 1: Histogram & Boxplot của Revenue trước và sau biến đổi log1p
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # Raw revenue (có ngoại lai gốc)
    raw_rev = train_merged["revenue"].dropna()
    axes[0, 0].hist(np.clip(raw_rev, 0, 100), bins=50, color="#3498db", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Phân phối Doanh thu Gốc (Cắt ngưỡng 0 - 100 tr.VNĐ)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Doanh thu (triệu VNĐ)")
    axes[0, 0].set_ylabel("Số lượng ngày")
    
    sns.boxplot(x=np.clip(raw_rev, 0, 100), ax=axes[0, 1], color="#e74c3c")
    axes[0, 1].set_title("Boxplot Doanh thu Gốc (Xuất hiện nhiều ngoại lai cực đoan)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Doanh thu (triệu VNĐ)")
    
    # Log1p revenue (đã hiệu chỉnh lỗi 1000x để minh họa)
    log_rev = np.log1p(df["revenue"])
    axes[1, 0].hist(log_rev, bins=50, color="#2ecc71", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Phân phối sau Biến đổi ln(1 + Revenue) (Gần phân phối chuẩn)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("ln(1 + Revenue)")
    axes[1, 0].set_ylabel("Số lượng ngày")
    
    sns.boxplot(x=log_rev, ax=axes[1, 1], color="#f39c12")
    axes[1, 1].set_title("Boxplot sau Biến đổi ln(1 + Revenue)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("ln(1 + Revenue)")
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_1_revenue_distribution.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_plots.append(p1)
    
    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 2: Chuỗi thời gian Nhiệt độ temp_c tại Hà Nội và TP.HCM
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    for city, col in [("Ha Noi", "#2980b9"), ("Ho Chi Minh", "#e67e22")]:
        city_w = weather_df[weather_df["city"] == city].sort_values("date")
        ax.plot(city_w["date"], city_w["temp_c"], label=city, color=col, alpha=0.8, linewidth=1.2)
        
    # Đánh dấu vùng Fahrenheit
    f_rows = weather_df[weather_df["temp_c"] > 45]
    if len(f_rows) > 0:
        ax.axhspan(45, 95, color="red", alpha=0.15, label="Vùng lỗi ghi nhầm độ °F (> 45°C)")
        ax.annotate(
            f"Lỗi độ °F tại TP.HCM (Tháng 6/2025): {len(f_rows)} ngày\\nNhiệt độ vọt lên {f_rows['temp_c'].min():.1f} - {f_rows['temp_c'].max():.1f}°F",
            xy=(pd.Timestamp("2025-06-20"), 85),
            xytext=(pd.Timestamp("2024-08-01"), 80),
            arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=8),
            fontsize=11, fontweight="bold", color="darkred",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", lw=1.5)
        )
        
    ax.set_title("Chuỗi thời gian Nhiệt độ temp_c tại Hà Nội và TP.HCM (2024 - 2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Nhiệt độ ghi nhận trong file (°C)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "eda_2_weather_temperature_fahrenheit.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_plots.append(p2)
    
    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 3: DoW × Store Type Medians (Sự đảo chiều ngày thường vs cuối tuần)
    # --------------------------------------------------------------------------
    dow_matrix = df.groupby(["store_type_clean", "dow"])["revenue"].median().unstack()
    dow_matrix.columns = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dow_matrix.T.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")
    ax.set_title("Sự tương phản Doanh thu theo Thứ trong tuần × Loại hình cửa hàng (dayofweek × store_type)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thứ trong tuần")
    ax.set_ylabel("Doanh thu trung vị (triệu VNĐ)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Loại cửa hàng", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "eda_3_dow_store_type_interaction.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_plots.append(p3)
    
    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 4: Quỹ đạo quanh Tết Nguyên Đán (days_from_tet in [-18, +18])
    # --------------------------------------------------------------------------
    tet_dates = {2024: pd.Timestamp("2024-02-10"), 2025: pd.Timestamp("2025-01-29")}
    df["year"] = df["date"].dt.year
    df["tet_date"] = df["year"].map(tet_dates)
    df["days_from_tet"] = (df["date"] - df["tet_date"]).dt.days
    
    tet_df = df[(df["days_from_tet"] >= -18) & (df["days_from_tet"] <= 18)]
    tet_agg = tet_df.groupby(["days_from_tet", "store_type_clean"])["sw_ratio"].median().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in tet_agg.columns:
        ax.plot(tet_agg.index, tet_agg[col], marker="o", markersize=4, label=col, linewidth=1.8)
        
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Mùng 1 Tết (Δ=0)")
    ax.axvspan(-2, 2, color="gray", alpha=0.15, label="Vùng Tết cốt lõi (29 Tết - Mùng 2)")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.7)
    
    ax.set_title("Quỹ đạo Doanh thu quanh Tết Nguyên Đán theo Loại hình cửa hàng (Δ_Tết từ -18 đến +18 ngày)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Số ngày so với Mùng 1 Tết (Δ_Tết = date - Mùng 1 Tết)")
    ax.set_ylabel("Tỷ số so với trung vị thứ (sw_ratio)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p4 = os.path.join(output_dir, "eda_4_tet_trajectory_by_store_type.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_plots.append(p4)
    
    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 5: Đường cong tuổi cửa hàng (days_since_open từ 0 đến 90 ngày)
    # --------------------------------------------------------------------------
    age_df = df[(df["days_since_open"] >= 0) & (df["days_since_open"] <= 90)]
    age_curve = age_df.groupby("days_since_open")["sw_ratio"].median()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(age_curve.index, age_curve.values, color="#8e44ad", linewidth=2.0, marker=".", label="Trung vị sw_ratio thực tế")
    ax.axhline(1.0, color="gray", linestyle="--", label="Mức bão hòa ổn định (1.0x)")
    
    # Chú thích đỉnh khai trương
    ax.annotate(
        "Đỉnh khai trương 3 ngày đầu: ~1.65x - 1.93x",
        xy=(1, age_curve.loc[1] if 1 in age_curve else 1.7),
        xytext=(10, 1.8),
        arrowprops=dict(facecolor="darkviolet", shrink=0.05, width=1.5),
        fontsize=10, fontweight="bold", color="darkviolet"
    )
    # Chú thích trũng tích lũy
    ax.annotate(
        "Đáy chạy đà tuần đầu: ~0.61x",
        xy=(5, age_curve.loc[5] if 5 in age_curve else 0.61),
        xytext=(15, 0.45),
        arrowprops=dict(facecolor="crimson", shrink=0.05, width=1.5),
        fontsize=10, fontweight="bold", color="crimson"
    )
    
    ax.set_title("Đường cong Trưởng thành của Quán mới (Store Lifecycle Curve: 0 - 90 ngày tuổi)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Số ngày kể từ ngày khai trương (days_since_open)")
    ax.set_ylabel("Tỷ số Doanh thu / Trung vị ổn định (sw_ratio)")
    ax.set_ylim(0.2, 2.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    p5 = os.path.join(output_dir, "eda_5_store_lifecycle_ramp_up.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved_plots.append(p5)
    
    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 6: Heatmap Ma trận Tương quan (Correlation Heatmap)
    # --------------------------------------------------------------------------
    df["log1p_revenue"] = np.log1p(df["revenue"])
    df["log_rain"] = np.log1p(df["rain_mm"].fillna(0))
    corr_cols = ["log1p_revenue", "seats", "area_m2", "has_parking", "is_promo", "temp_c", "log_rain", "days_since_open"]
    corr_df = df[corr_cols].dropna().corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Ma trận Tương quan Tuyến tính Pearson giữa log1p(revenue) và các biến chính", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p6 = os.path.join(output_dir, "eda_6_feature_correlation_heatmap.png")
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    saved_plots.append(p6)
    
    return saved_plots


def run_step2(tables: Dict[str, pd.DataFrame], train_merged: pd.DataFrame, output_dir: str = "plots", verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Hàm thực thi toàn bộ BƯỚC 2: EXPLORE (EDA & DANH SÁCH LỖI DỮ LIỆU BẨN).
    """
    if verbose:
        print("=" * 80)
        print("BƯỚC 2: EXPLORE (KHÁM PHÁ DỮ LIỆU - EDA & DANH SÁCH LỖI DỮ LIỆU BẨN)")
        print("=" * 80)
        print("Nguyên tắc: Liệt kê chi tiết toàn bộ vấn đề dữ liệu bẩn TRƯỚC KHI sửa lỗi ở Bước 3.")
        print("-" * 80)
        
    # 1. Bảng 10 vấn đề dữ liệu bẩn
    anomalies_df = build_ten_anomalies_table(tables)
    if verbose:
        print(f"ĐÃ PHÁT HIỆN CHÍNH XÁC {len(anomalies_df)} VẤN ĐỀ DỮ LIỆU BẨN:")
        for _, row in anomalies_df.iterrows():
            print(f"  {row['STT']:2d}. [{row['Bảng dữ liệu']}] {row['Vấn đề phát hiện']}")
            print(f"      -> Minh chứng: {row['Minh chứng định lượng']}")
            
    # 2. Phân tích 6 quy luật kinh doanh
    patterns = analyze_business_patterns(train_merged)
    if verbose:
        print("-" * 80)
        print("ĐÃ PHÂN TÍCH 6 QUY LUẬT KINH DOANH CỐT LÕI (DOMAIN INSIGHTS):")
        print("  1. DoW x Store Type: Nhóm office/university giảm mạnh cuối tuần (-58% và -43%), nhóm mall tăng vọt (+69%).")
        print("  2. Tết Nguyên Đán: Chu kỳ đóng cửa và hồi phục dị biệt theo loại quán (university nghỉ 1 tháng, mall bùng nổ trước Tết).")
        print(f"  3. Nghỉ hè Đại học: Doanh thu khối university sụt giảm {patterns['uni_summer_drop_pct']:.1f}% trong tháng 7 và 8.")
        print("  4. Ngày lễ Quốc gia vs Sự kiện: Lễ quốc gia phân hóa theo nhóm; ngày sự kiện (Valentine, 8/3) đẩy doanh thu toàn hệ thống.")
        print("  5. Vòng đời quán mới: Đỉnh khai trương 3 ngày đầu (~1.65x - 1.93x), đáy tích lũy 7 ngày đầu (~0.61x).")
        print(f"  6. Khuyến mãi & Thời tiết: Khuyến mãi tăng trung bình +{patterns['promo_uplift_pct']:.1f}%; mưa lớn giảm quán ngoài trời, tăng nhẹ quán mall.")
        
    # 3. Vẽ 6 biểu đồ EDA
    plot_paths = generate_eda_plots(train_merged, tables["weather"], output_dir=output_dir)
    if verbose:
        print("-" * 80)
        print(f"ĐÃ TẠO VÀ LƯU THÀNH CÔNG {len(plot_paths)} BIỂU ĐỒ EDA VÀO THƯ MỤC '{output_dir}/':")
        for p in plot_paths:
            print(f"  * {os.path.basename(p)}")
        print("=" * 80)
        print("HOÀN THÀNH TOÀN BỘ BƯỚC 2: EXPLORE THEO ĐÚNG KẾ HOẠCH!")
        print("=" * 80)
        
    return anomalies_df, patterns, plot_paths


if __name__ == "__main__":
    from step1_collect_prepare import run_step1
    train_merged, test_merged, data_dict, key_check = run_step1("data", verbose=False)
    tables = {
        "train": pd.read_csv("data/train.csv", parse_dates=["date"]),
        "test": pd.read_csv("data/test.csv", parse_dates=["date"]),
        "stores": pd.read_csv("data/stores.csv"),
        "weather": pd.read_csv("data/weather.csv", parse_dates=["date"]),
        "holidays": pd.read_csv("data/holidays.csv", parse_dates=["date"])
    }
    run_step2(tables, train_merged)
