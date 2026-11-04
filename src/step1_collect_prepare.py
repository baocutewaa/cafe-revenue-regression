"""
Module: step1_collect_prepare.py
Chủ đề: Dự báo doanh thu ngày chuỗi quán Cafe (Cafe Revenue Regression)
Mô tả: Triển khai toàn bộ BƯỚC 1: COLLECT & PREPARE (THU THẬP, ĐỊNH NGHĨA BÀI TOÁN & GHÉP BẢNG)
Tuân thủ chuẩn quy trình 8 bước trong bài giảng Chương 4 và kế hoạch Regression_Plan.md
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính chỉ số RMSLE (Root Mean Squared Logarithmic Error) chính thức của bài toán:
    RMSLE = sqrt( 1/n * sum( (ln(1 + max(0, y_pred)) - ln(1 + y_true))^2 ) )
    
    Tham số:
        y_true: Doanh thu thực tế (triệu VNĐ)
        y_pred: Doanh thu dự báo (triệu VNĐ)
        
    Trả về:
        Giá trị RMSLE (float)
    """
    y_true_clipped = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred_clipped = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2)))


def load_raw_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Đọc toàn bộ 5 bảng dữ liệu thô và file sample_submission.
    Áp dụng parse_dates=['date'] cho các bảng có chứa trường thời gian.
    
    Tham số:
        data_dir: Đường dẫn thư mục chứa dữ liệu
        
    Trả về:
        Dictionary chứa 5 DataFrames: train, test, stores, weather, holidays (và sample_submission)
    """
    files_to_load = {
        "train": ("train.csv", ["date"]),
        "test": ("test.csv", ["date"]),
        "stores": ("stores.csv", None),
        "weather": ("weather.csv", ["date"]),
        "holidays": ("holidays.csv", ["date"]),
        "sample_submission": ("sample_submission.csv", None)
    }
    
    tables: Dict[str, pd.DataFrame] = {}
    for key, (filename, parse_dates) in files_to_load.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
        if parse_dates:
            df = pd.read_csv(file_path, parse_dates=parse_dates)
        else:
            df = pd.read_csv(file_path)
            
        tables[key] = df
        
    return tables


def build_data_dictionary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Xây dựng bảng Từ điển dữ liệu (Data Dictionary) chi tiết cho 5 bảng dữ liệu chính.
    
    Tham số:
        tables: Dictionary các bảng dữ liệu
        
    Trả về:
        DataFrame tổng hợp thông tin cấu trúc, kích thước, khóa nối và phạm vi ngày.
    """
    summary_records = [
        {
            "Table Name": "train.csv",
            "Rows": tables["train"].shape[0],
            "Columns": tables["train"].shape[1],
            "Date Range": f"{tables['train']['date'].min().strftime('%Y-%m-%d')} to {tables['train']['date'].max().strftime('%Y-%m-%d')} (731 days)",
            "Primary / Merge Key": "(store_id, date)",
            "Key Columns & Types": "date (datetime64), store_id (str), is_promo (int64), revenue (float64)",
            "Missing Values": f"{tables['train'].isnull().sum().sum()} NaNs",
            "Business Context": "Doanh thu lịch sử ngày của 45 quán trong 2 năm 2024-2025"
        },
        {
            "Table Name": "test.csv",
            "Rows": tables["test"].shape[0],
            "Columns": tables["test"].shape[1],
            "Date Range": f"{tables['test']['date'].min().strftime('%Y-%m-%d')} to {tables['test']['date'].max().strftime('%Y-%m-%d')} (90 days)",
            "Primary / Merge Key": "id, (store_id, date)",
            "Key Columns & Types": "id (int64), date (datetime64), store_id (str), is_promo (int64)",
            "Missing Values": f"{tables['test'].isnull().sum().sum()} NaNs",
            "Business Context": "Dữ liệu cần dự báo cho 44 quán (không có S02) trong Q1/2026"
        },
        {
            "Table Name": "stores.csv",
            "Rows": tables["stores"].shape[0],
            "Columns": tables["stores"].shape[1],
            "Date Range": "Tĩnh theo cửa hàng (open_date từ 2021 đến 2025)",
            "Primary / Merge Key": "store_id",
            "Key Columns & Types": "store_id (str), city (str), district (str), store_type (str), area_m2 (float64), seats (int64), has_parking (int64), open_date (str)",
            "Missing Values": f"{tables['stores'].isnull().sum().sum()} NaNs (area_m2: 2 NaNs)",
            "Business Context": "Đặc tính vật lý và địa bàn hoạt động của 45 cửa hàng"
        },
        {
            "Table Name": "weather.csv",
            "Rows": tables["weather"].shape[0],
            "Columns": tables["weather"].shape[1],
            "Date Range": f"{tables['weather']['date'].min().strftime('%Y-%m-%d')} to {tables['weather']['date'].max().strftime('%Y-%m-%d')} (821 days)",
            "Primary / Merge Key": "(city, date)",
            "Key Columns & Types": "date (datetime64), city (str), temp_c (float64), rain_mm (float64)",
            "Missing Values": f"{tables['weather'].isnull().sum().sum()} NaNs (rain_mm: 16 NaNs)",
            "Business Context": "Thời tiết ngày tại 2 thành phố (Hà Nội, TP.HCM)"
        },
        {
            "Table Name": "holidays.csv",
            "Rows": tables["holidays"].shape[0],
            "Columns": tables["holidays"].shape[1],
            "Date Range": f"{tables['holidays']['date'].min().strftime('%Y-%m-%d')} to {tables['holidays']['date'].max().strftime('%Y-%m-%d')} (53 unique days)",
            "Primary / Merge Key": "date (Cảnh báo: Bị trùng lặp khóa!)",
            "Key Columns & Types": "date (datetime64), holiday_name (str), type (str)",
            "Missing Values": f"{tables['holidays'].isnull().sum().sum()} NaNs",
            "Business Context": "Danh mục ngày lễ Quốc gia và ngày Sự kiện đặc biệt (55 dòng)"
        }
    ]
    return pd.DataFrame(summary_records)


def check_merge_keys(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Kiểm tra tính toàn vẹn của các khóa nối (Merge Keys Integrity) giữa 5 bảng.
    Phát hiện xung đột khóa nối và minh chứng hiện tượng dội dòng khi merge trực tiếp.
    
    Tham số:
        tables: Dictionary các bảng dữ liệu
        
    Trả về:
        Dictionary chứa kết quả kiểm tra chi tiết.
    """
    train = tables["train"]
    test = tables["test"]
    stores = tables["stores"]
    weather = tables["weather"]
    holidays = tables["holidays"]
    
    results = {}
    
    # 1. Khóa store_id
    train_stores = set(train["store_id"].unique())
    test_stores = set(test["store_id"].unique())
    stores_catalog = set(stores["store_id"].unique())
    
    results["train_stores_in_catalog"] = train_stores.issubset(stores_catalog)
    results["test_stores_in_catalog"] = test_stores.issubset(stores_catalog)
    results["stores_count"] = len(stores_catalog)
    results["train_stores_count"] = len(train_stores)
    results["test_stores_count"] = len(test_stores)
    results["missing_in_test"] = list(stores_catalog - test_stores)
    
    # 2. Khóa city
    stores_cities = set(stores["city"].unique())
    weather_cities = set(weather["city"].unique())
    results["cities_match"] = stores_cities == weather_cities
    results["cities_list"] = list(stores_cities)
    
    # 3. Khóa date trong holidays
    date_freq = holidays["date"].value_counts()
    dup_dates = date_freq[date_freq > 1]
    results["has_holiday_key_duplicate"] = len(dup_dates) > 0
    results["duplicated_holiday_dates"] = {
        d.strftime("%Y-%m-%d"): holidays[holidays["date"] == d]["holiday_name"].tolist()
        for d in dup_dates.index
    }
    
    # 4. Mô phỏng rủi ro Naive Merge trên tập test
    test_naive = test.merge(stores, on="store_id", how="left")
    test_naive = test_naive.merge(weather, on=["city", "date"], how="left")
    test_naive = test_naive.merge(holidays, on="date", how="left")
    
    results["test_original_rows"] = len(test)
    results["test_naive_merged_rows"] = len(test_naive)
    results["test_naive_exceeded_rows"] = len(test_naive) - len(test)
    
    return results


def prepare_holidays_table(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý và gộp nhóm bảng holidays.csv theo 'date' để đảm bảo tính duy nhất 1-1.
    Tạo các cột cờ nhị phân (binary flags) và tên sự kiện ghép chuỗi:
    - is_national_holiday: 1 nếu là ngày nghỉ lễ quốc gia, ngược lại 0
    - is_event_holiday: 1 nếu là ngày sự kiện đặc biệt, ngược lại 0
    - holiday_name_clean: Tên sự kiện (ghép chuỗi nếu trùng ngày)
    
    Tham số:
        holidays_df: Bảng holidays gốc (55 dòng)
        
    Trả về:
        DataFrame holidays đã gộp nhóm (53 dòng duy nhất theo date)
    """
    holidays_agg = holidays_df.groupby("date").agg(
        is_national_holiday=("type", lambda s: int((s == "national_holiday").any())),
        is_event_holiday=("type", lambda s: int((s == "event").any())),
        holiday_name_clean=("holiday_name", lambda s: " + ".join(s))
    ).reset_index()
    
    return holidays_agg


def safe_merge_tables(
    base_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    holidays_agg_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Thực hiện ghép bảng an toàn (Safe Left-Merge) tuân thủ nguyên tắc kiểm soát số dòng:
    1. Ghép với stores_df theo 'store_id' (để lấy thuộc tính quán và 'city')
    2. Ghép với weather_df theo ['city', 'date'] (để lấy 'temp_c', 'rain_mm')
    3. Ghép với holidays_agg_df theo 'date' (để lấy đặc trưng ngày lễ/sự kiện)
    
    Điền giá trị mặc định cho các cột ngày lễ nếu ngày đó không phải ngày lễ:
    - is_national_holiday = 0
    - is_event_holiday = 0
    - holiday_name_clean = 'None'
    
    Kiểm tra tự động: Số dòng trước và sau khi merge phải bằng nhau tuyệt đối!
    """
    initial_rows = len(base_df)
    
    # Bước A: Ghép thông tin cửa hàng
    merged = base_df.merge(stores_df, on="store_id", how="left")
    assert len(merged) == initial_rows, f"Số dòng thay đổi sau khi merge stores: {len(merged)} != {initial_rows}"
    
    # Bước B: Ghép thông tin thời tiết
    merged = merged.merge(weather_df, on=["city", "date"], how="left")
    assert len(merged) == initial_rows, f"Số dòng thay đổi sau khi merge weather: {len(merged)} != {initial_rows}"
    
    # Bước C: Ghép thông tin ngày lễ đã gộp nhóm
    merged = merged.merge(holidays_agg_df, on="date", how="left")
    assert len(merged) == initial_rows, f"Số dòng thay đổi sau khi merge holidays: {len(merged)} != {initial_rows}"
    
    # Điền giá trị mặc định cho các ngày thường (không có trong holidays)
    merged["is_national_holiday"] = merged["is_national_holiday"].fillna(0).astype(int)
    merged["is_event_holiday"] = merged["is_event_holiday"].fillna(0).astype(int)
    merged["holiday_name_clean"] = merged["holiday_name_clean"].fillna("None")
    
    return merged


def run_step1(data_dir: str = "data", verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Hàm thực thi toàn bộ BƯỚC 1: COLLECT & PREPARE.
    
    Tham số:
        data_dir: Thư mục chứa dữ liệu CSV
        verbose: In thông tin chi tiết ra màn hình
        
    Trả về:
        Tuple gồm (train_merged, test_merged, data_dict_df, key_check_results)
    """
    if verbose:
        print("=" * 80)
        print("BƯỚC 1: COLLECT & PREPARE (THU THẬP, ĐỊNH NGHĨA BÀI TOÁN & GHÉP BẢNG)")
        print("=" * 80)
        print("1.1. ĐỊNH NGHĨA BÀI TOÁN KINH DOANH:")
        print("  - Câu hỏi kinh doanh: Dự báo doanh thu ngày (triệu VNĐ) của chuỗi quán Cafe trong Q1/2026.")
        print("  - Loại bài toán: Hồi quy chuỗi thời gian đa cửa hàng (Multi-Store Time-Series Regression).")
        print("  - Biến mục tiêu: 'revenue' (triệu VNĐ, giá trị >= 0).")
        print("  - Thước đo đánh giá: RMSLE (Root Mean Squared Logarithmic Error).")
        print("-" * 80)
        
    # 1. Tải 5 bảng
    tables = load_raw_data(data_dir)
    if verbose:
        print("1.2. ĐỌC THÀNH CÔNG 5 BẢNG DỮ LIỆU:")
        for name in ["train", "test", "stores", "weather", "holidays"]:
            print(f"  - {name}.csv: shape = {tables[name].shape}")
            
    # 2. Xây dựng Data Dictionary
    data_dict_df = build_data_dictionary(tables)
    
    # 3. Kiểm tra khóa nối
    key_check = check_merge_keys(tables)
    if verbose:
        print("-" * 80)
        print("1.3. KIỂM TRA TÍNH TOÀN VẸN CỦA KHÓA NỐI:")
        print(f"  - Khóa 'store_id' của train nằm trọn trong stores: {key_check['train_stores_in_catalog']} ({key_check['train_stores_count']}/45 quán)")
        print(f"  - Khóa 'store_id' của test nằm trọn trong stores: {key_check['test_stores_in_catalog']} ({key_check['test_stores_count']}/45 quán)")
        print(f"  - Quán có trong stores nhưng không cần dự báo ở test: {key_check['missing_in_test']}")
        print(f"  - Khóa 'city' giữa stores và weather khớp hoàn toàn: {key_check['cities_match']} {key_check['cities_list']}")
        print(f"  - CẢNH BÁO TRÙNG LẶP KHÓA 'date' TRONG HOLIDAYS: {key_check['has_holiday_key_duplicate']}")
        for d, events in key_check["duplicated_holiday_dates"].items():
            print(f"    * Ngày {d}: {events}")
        print(f"  - RỦI RO NAIVE MERGE: test từ {key_check['test_original_rows']} dòng -> {key_check['test_naive_merged_rows']} dòng (BỊ DỘI THÊM {key_check['test_naive_exceeded_rows']} DÒNG!)")
        
    # 4. Tiền xử lý holidays_agg
    holidays_agg = prepare_holidays_table(tables["holidays"])
    if verbose:
        print(f"  - ĐÃ GIẢI QUYẾT: Gộp nhóm holidays_agg -> {holidays_agg.shape} (mỗi ngày đúng 1 dòng duy nhất)")
        
    # 5. Ghép bảng có kiểm soát
    train_merged = safe_merge_tables(tables["train"], tables["stores"], tables["weather"], holidays_agg)
    test_merged = safe_merge_tables(tables["test"], tables["stores"], tables["weather"], holidays_agg)
    
    if verbose:
        print("-" * 80)
        print("1.4. KẾT QUẢ GHÉP BẢNG AN TOÀN (SAFE MERGE):")
        print(f"  - train_merged: {train_merged.shape} (Kích thước gốc: {tables['train'].shape[0]} dòng -> BẢO TOÀN 100%)")
        print(f"  - test_merged:  {test_merged.shape}  (Kích thước gốc: {tables['test'].shape[0]} dòng -> BẢO TOÀN 100%)")
        print(f"  - Danh sách 16 cột đã ghép:")
        for i, col in enumerate(train_merged.columns, 1):
            print(f"    {i:2d}. {col:25s} (dtype: {train_merged[col].dtype})")
        print("=" * 80)
        print("HOÀN THÀNH TOÀN BỘ BƯỚC 1: COLLECT & PREPARE THEO ĐÚNG KẾ HOẠCH!")
        print("=" * 80)
        
    return train_merged, test_merged, data_dict_df, key_check


if __name__ == "__main__":
    run_step1()
