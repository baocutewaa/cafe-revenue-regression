"""
BƯỚC 8: DEPLOY (HUẤN LUYỆN LẠI TRÊN 100% DỮ LIỆU SẠCH, XUẤT SUBMISSION.CSV NỘP KAGGLE, ĐÓNG GÓI ARTIFACT VÀ PHỤC VỤ API/UI)

Kiến trúc thực hiện theo Regression_Plan.md:
1. Huấn luyện lại toàn bộ mô hình trong Weighted Blend Ensemble trên 100% dữ liệu train sạch (26.500 dòng).
2. Tạo tệp dự báo `submission.csv` (3.869 dòng) nộp lên Kaggle và thực hiện bộ kiểm tra tự động (Sanity Checks).
3. Đóng gói toàn bộ tài nguyên vào `artifacts/cafe_revenue_bundle.joblib` và `artifacts/model_metrics_summary.json`.
4. Cung cấp hàm tiện ích dự báo đơn lẻ (Single Prediction / What-If) và dự báo hàng loạt (Batch Prediction)
   dùng chung cho FastAPI Backend và Streamlit Dashboard.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone

try:
    from .step1_collect_prepare import rmsle, load_raw_data
    from .step3_preprocess import run_step3, build_features, DeSeasonalizedStoreStats
    from .step5_scale import (
        build_leakage_free_pipeline,
        get_feature_column_types
    )
    from .step6_model_train import (
        WeightedBlendEnsemble,
        get_candidate_models
    )
except ImportError:
    from step1_collect_prepare import rmsle, load_raw_data
    from step3_preprocess import run_step3, build_features, DeSeasonalizedStoreStats
    from step5_scale import (
        build_leakage_free_pipeline,
        get_feature_column_types
    )
    from step6_model_train import (
        WeightedBlendEnsemble,
        get_candidate_models
    )


# Trọng số tối ưu chuẩn xác đa mùa qua kiểm định chéo 4 Folds có bao phủ Tết Nguyên Đán
DEFAULT_ENSEMBLE_WEIGHTS = {
    "CatBoost": 0.4000,
    "LightGBM": 0.2500,
    "XGBoost": 0.2000,
    "Ridge (L2 Linear)": 0.1500
}


def retrain_ensemble_on_full_data(
    train_clean_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    ensemble_weights: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> Tuple[WeightedBlendEnsemble, Dict[str, Any]]:
    """
    Huấn luyện lại toàn bộ 4 mô hình trong Ensemble trên 100% tập dữ liệu lịch sử sạch
    (26.500 dòng từ 2024-01-01 đến 2025-12-31).
    """
    if ensemble_weights is None:
        ensemble_weights = DEFAULT_ENSEMBLE_WEIGHTS.copy()
        
    features = numeric_cols + categorical_cols
    features_with_time = [c for c in train_clean_df.columns if c in features or c in ["t_years", "date"]]
    X_full = train_clean_df[features_with_time]
    y_full = train_clean_df["revenue"].values
    
    candidate_models = get_candidate_models()
    fitted_pipelines = {}
    
    if verbose:
        print(f"Bắt đầu huấn luyện lại Ensemble trên 100% dữ liệu ({len(train_clean_df):,} dòng, {len(features)} đặc trưng)...")
        
    for name in ensemble_weights.keys():
        t0 = time.time()
        pipeline = build_leakage_free_pipeline(clone(candidate_models[name]), numeric_cols, categorical_cols, use_trend_aware=True)
        pipeline.fit(X_full, y_full)
        elapsed = time.time() - t0
        fitted_pipelines[name] = pipeline
        if verbose:
            print(f"  * [Đã huấn luyện] {name:20s} trong {elapsed:.2f}s")
            
    final_ensemble = WeightedBlendEnsemble(
        models_dict=fitted_pipelines,
        weights=ensemble_weights
    )
    
    return final_ensemble, fitted_pipelines


def generate_kaggle_submission(
    fitted_ensemble: WeightedBlendEnsemble,
    test_clean_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    output_path: str = "submission.csv",
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tạo tệp dự báo `submission.csv` chuẩn định dạng Kaggle (3.869 dòng) và thực hiện các kiểm tra tự động.
    """
    features = numeric_cols + categorical_cols
    features_with_time = [c for c in test_clean_df.columns if c in features or c in ["t_years", "date"]]
    X_test = test_clean_df[features_with_time]
    
    # Dự báo doanh thu bằng Ensemble
    preds = fitted_ensemble.predict(X_test)
    preds_clean = np.maximum(0.0, preds)
    
    sub = pd.DataFrame({
        "id": test_clean_df["id"].values,
        "revenue": np.round(preds_clean, 2)
    })
    
    # BỘ KIỂM TRA TỰ ĐỘNG (SANITY CHECKS)
    check_rows = (len(sub) == 3869)
    check_cols = (list(sub.columns) == ["id", "revenue"])
    check_no_null = (sub.isnull().sum().sum() == 0)
    check_positive = ((sub["revenue"] <= 0).sum() == 0)
    check_min = float(sub["revenue"].min())
    check_median = float(sub["revenue"].median())
    check_max = float(sub["revenue"].max())
    check_reasonable_range = (check_min >= 1.0 and check_max <= 100.0)
    
    all_passed = all([check_rows, check_cols, check_no_null, check_positive, check_reasonable_range])
    
    sanity_results = {
        "all_passed": all_passed,
        "row_count": len(sub),
        "columns": list(sub.columns),
        "null_count": int(sub.isnull().sum().sum()),
        "zero_or_negative_count": int((sub["revenue"] <= 0).sum()),
        "min_revenue": check_min,
        "median_revenue": check_median,
        "mean_revenue": float(sub["revenue"].mean()),
        "max_revenue": check_max
    }
    
    # Lưu tệp CSV ra đĩa
    sub.to_csv(output_path, index=False)
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"KẾT QUẢ TẠO TỆP NỘP BÀI KAGGLE: {output_path}")
        print("=" * 80)
        print(f"  * Số dòng dự báo:              {len(sub):,} dòng (Kỳ vọng: 3,869)")
        print(f"  * Danh sách cột:               {list(sub.columns)} (Kỳ vọng: ['id', 'revenue'])")
        print(f"  * Giá trị NaN/Inf/Null:        {sanity_results['null_count']}")
        print(f"  * Giá trị âm hoặc bằng 0:      {sanity_results['zero_or_negative_count']}")
        print(f"  * Phân phối doanh thu dự báo:  Min = {check_min:.2f} | Median = {check_median:.2f} | Mean = {sanity_results['mean_revenue']:.2f} | Max = {check_max:.2f} tr.VNĐ")
        print(f"  * Kết luận kiểm tra:           {'PASSED 100% (HỢP LỆ ĐỂ NỘP KAGGLE)' if all_passed else 'FAILED'}")
        print("=" * 80)
        
    return sub, sanity_results


def package_artifact_bundle(
    fitted_ensemble: WeightedBlendEnsemble,
    numeric_cols: List[str],
    categorical_cols: List[str],
    stores_clean: pd.DataFrame,
    weather_clean: pd.DataFrame,
    holidays_clean: pd.DataFrame,
    store_stats_encoder: Any,
    metrics_summary: Optional[Dict[str, Any]] = None,
    bundle_path: str = "artifacts/cafe_revenue_bundle.joblib",
    json_path: str = "artifacts/model_metrics_summary.json",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Đóng gói toàn bộ mô hình, pipeline và các bảng tham chiếu thành một gói duy nhất (`joblib`)
    và lưu tệp JSON tổng hợp chỉ số để phục vụ FastAPI Backend và Streamlit Dashboard.
    """
    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
    
    if metrics_summary is None:
        metrics_summary = {
            "model_name": "Weighted Blend Ensemble (SOTA)",
            "holdout_val_rmsle": 0.1159,
            "seasonal_tet_rmsle": 0.1138,
            "cv_mean_rmsle": 0.1130,
            "val_mae_m_vnd": 1.95,
            "val_rmse_m_vnd": 2.74,
            "val_r2": 0.9080,
            "starter_baseline_rmsle": 0.2276,
            "relative_improvement_pct": 50.3,
            "ensemble_weights": DEFAULT_ENSEMBLE_WEIGHTS
        }
        
    bundle = {
        "models": fitted_ensemble.models_dict,
        "weights": fitted_ensemble.weights,
        "fitted_weights": fitted_ensemble.fitted_weights_,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "features": numeric_cols + categorical_cols,
        "stores_lookup": stores_clean,
        "weather_lookup": weather_clean,
        "holidays_lookup": holidays_clean,
        "store_stats_encoder": store_stats_encoder,
        "metrics": metrics_summary
    }
    
    joblib.dump(bundle, bundle_path, compress=3)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        
    if verbose:
        bundle_size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
        print(f"[Đã đóng gói] Gói tài nguyên: {bundle_path} ({bundle_size_mb:.2f} MB)")
        print(f"[Đã đóng gói] Tệp chỉ số:   {json_path}")
        
    return bundle


def predict_single_instance(
    bundle: Dict[str, Any],
    store_id: str,
    date_str: str,
    is_promo: int = 0,
    temp_c: Optional[float] = None,
    rain_mm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Dự báo doanh thu cho 1 ngày tại 1 cửa hàng, hỗ trợ mô phỏng kịch bản What-If.
    Hàm này được dùng chung bởi cả FastAPI Backend và Streamlit Frontend.
    """
    stores_df = bundle["stores_lookup"]
    weather_df = bundle["weather_lookup"]
    holidays_df = bundle["holidays_lookup"]
    encoder = bundle["store_stats_encoder"]
    features = bundle["features"]
    
    # 1. Tra cứu thông tin quán
    store_row = stores_df[stores_df["store_id"] == store_id]
    if len(store_row) == 0:
        raise ValueError(f"Không tìm thấy mã cửa hàng '{store_id}' trong hệ thống!")
    store_info = store_row.iloc[0].to_dict()
    city = store_info["city"]
    
    # 2. Tra cứu hoặc điền thời tiết
    dt = pd.to_datetime(date_str)
    w_match = weather_df[(weather_df["city"] == city) & (weather_df["date"] == dt)]
    if len(w_match) > 0:
        if temp_c is None:
            temp_c = float(w_match.iloc[0]["temp_c"])
        if rain_mm is None:
            rain_mm = float(w_match.iloc[0]["rain_mm"])
        temp_dev = float(w_match.iloc[0]["temp_deviation_from_city_mean"])
    else:
        if temp_c is None:
            temp_c = 28.0 if city == "Ho Chi Minh" else 22.0
        if rain_mm is None:
            rain_mm = 0.0
        city_mean = 27.5 if city == "Ho Chi Minh" else 23.5
        temp_dev = temp_c - city_mean
            
    # 3. Tạo DataFrame dòng đơn và ghép bảng theo đúng quy trình huấn luyện
    sample_df = pd.DataFrame([{
        "date": dt,
        "store_id": store_id,
        "is_promo": int(is_promo),
        "revenue": np.nan
    }])
    
    sample_merged = sample_df.merge(stores_df, on="store_id", how="left")
    sample_merged = sample_merged.merge(
        pd.DataFrame([{
            "date": dt,
            "city": city,
            "temp_c": temp_c,
            "rain_mm": rain_mm,
            "temp_deviation_from_city_mean": temp_dev
        }]),
        on=["date", "city"], how="left"
    )
    sample_merged = sample_merged.merge(holidays_df, on="date", how="left")
    sample_merged["is_national_holiday"] = sample_merged["is_national_holiday"].fillna(0).astype(int)
    sample_merged["is_event_holiday"] = sample_merged["is_event_holiday"].fillna(0).astype(int)
    sample_merged["holiday_name_clean"] = sample_merged["holiday_name_clean"].fillna("None")
    
    # Tạo đặc trưng cơ sở
    sample_feat = build_features(sample_merged)
    # Áp dụng StoreStatsEncoder để gắn store_base_log_rev, store_dow_log_rev, store_dow_ratio
    sample_final = encoder.transform(sample_feat)
    
    # 4. Dự báo bằng Ensemble
    preds_log = []
    component_preds = {}
    weights = bundle["weights"]
    
    for name, model_pipe in bundle["models"].items():
        sample_with_time = [c for c in sample_final.columns if c in features or c in ["t_years", "date"]]
        p_val = float(model_pipe.predict(sample_final[sample_with_time])[0])
        p_val_safe = max(0.0, p_val)
        component_preds[name] = round(p_val_safe, 2)
        preds_log.append(np.log1p(p_val_safe))
        
    w_vec = np.array([weights[name] for name in bundle["models"].keys()])
    w_vec = w_vec / np.sum(w_vec)
    blend_log = float(np.dot(w_vec, preds_log))
    final_pred = float(np.expm1(blend_log))
    final_pred = round(max(0.0, final_pred), 2)
    
    # 5. Tính kịch bản What-If khi đổi trạng thái khuyến mãi (Promo Uplift)
    sample_opp = sample_merged.copy()
    opp_promo = 1 - int(is_promo)
    sample_opp["is_promo"] = opp_promo
    sample_opp_feat = build_features(sample_opp)
    sample_opp_final = encoder.transform(sample_opp_feat)
    
    opp_preds_log = []
    for name, model_pipe in bundle["models"].items():
        p_opp = float(model_pipe.predict(sample_opp_final[features])[0])
        opp_preds_log.append(np.log1p(max(0.0, p_opp)))
    opp_blend_log = float(np.dot(w_vec, opp_preds_log))
    opp_pred = round(max(0.0, float(np.expm1(opp_blend_log))), 2)
    
    promo_uplift = round(opp_pred - final_pred if is_promo == 0 else final_pred - opp_pred, 2)
    base_val_calc = final_pred if is_promo == 0 else opp_pred
    promo_uplift_pct = round((promo_uplift / max(0.1, base_val_calc)) * 100, 1)
    
    # 6. Thứ trong tuần
    dow = dt.day_name()
    dow_map = {0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm", 4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật"}
    dow_name_vi = dow_map.get(dt.weekday(), dow)
    
    # 7. Khuyến nghị vận hành tự động (Smart Operational Recommendation)
    rec = []
    if final_pred > 30.0:
        rec.append("Dự kiến doanh thu cao vượt trội (> 30 triệu). Khuyến nghị bổ sung 30-40% nguyên liệu pha chế và bố trí thêm 2 nhân viên ca cao điểm.")
    elif final_pred < 12.0:
        rec.append("Dự kiến ngày thấp điểm (< 12 triệu). Có thể tối ưu chi phí nhân sự và tập trung kiểm kê nguyên vật liệu.")
        
    if is_promo == 1 and rain_mm > 25:
        rec.append("Cảnh báo: Mưa lớn trùng với ngày khuyến mãi. Hiệu ứng kích cầu tại quán ngoài trời sẽ giảm, khuyến nghị đẩy mạnh kênh giao hàng mang về.")
    elif is_promo == 1:
        rec.append(f"Chương trình khuyến mãi ước tính mang lại thêm {abs(promo_uplift):.2f} triệu VNĐ (+{abs(promo_uplift_pct)}%).")
        
    if sample_final.iloc[0]["days_from_tet"] == 0:
        rec.append("Hôm nay là Mùng 1 Tết Nguyên Đán! Nhu cầu đồ uống du xuân tăng cao ở các quán mặt phố và TTTM.")
        
    recommendation_text = " | ".join(rec) if rec else "Hoạt động vận hành ở mức bình thường."
    
    return {
        "store_id": store_id,
        "date": date_str,
        "day_of_week": dow_name_vi,
        "store_type": store_info["store_type"],
        "city": store_info["city"],
        "seats": store_info["seats"],
        "area_m2": store_info["area_m2"],
        "is_promo": int(is_promo),
        "temp_c": temp_c,
        "rain_mm": rain_mm,
        "predicted_revenue_m_vnd": final_pred,
        "component_predictions": component_preds,
        "what_if_opposite_promo_pred": opp_pred,
        "promo_uplift_m_vnd": promo_uplift,
        "promo_uplift_pct": promo_uplift_pct,
        "operational_recommendation": recommendation_text
    }


def predict_batch_dataframe(bundle: Dict[str, Any], input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý dự báo hàng loạt theo lô (Batch Prediction) cho danh sách bản ghi bất kỳ có:
    id, date, store_id, is_promo.
    """
    stores_df = bundle["stores_lookup"]
    weather_df = bundle["weather_lookup"]
    holidays_df = bundle["holidays_lookup"]
    encoder = bundle["store_stats_encoder"]
    features = bundle["features"]
    weights = bundle["weights"]
    
    df = input_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Ghép bảng chuẩn hóa
    df = df.merge(stores_df, on="store_id", how="left")
    df = df.merge(weather_df, on=["date", "city"], how="left")
    df = df.merge(holidays_df, on="date", how="left")
    df["is_national_holiday"] = df["is_national_holiday"].fillna(0).astype(int)
    df["is_event_holiday"] = df["is_event_holiday"].fillna(0).astype(int)
    df["holiday_name_clean"] = df["holiday_name_clean"].fillna("None")
    
    # Điền thiếu thời tiết nếu có ngày tương lai
    df["temp_c"] = df["temp_c"].fillna(26.5)
    df["rain_mm"] = df["rain_mm"].fillna(0.0)
    df["temp_deviation_from_city_mean"] = df["temp_deviation_from_city_mean"].fillna(0.0)
    
    # Tạo đặc trưng
    df_feat = build_features(df)
    df_final = encoder.transform(df_feat)
    
    # Dự báo bằng Ensemble
    preds_log = []
    w_vec = np.array([weights[name] for name in bundle["models"].keys()])
    w_vec = w_vec / np.sum(w_vec)
    
    df_with_time = [c for c in df_final.columns if c in features or c in ["t_years", "date"]]
    for name, model_pipe in bundle["models"].items():
        p_val = model_pipe.predict(df_final[df_with_time])
        preds_log.append(np.log1p(np.maximum(0.0, p_val)))
        
    P_log = np.column_stack(preds_log)
    blend_log = P_log @ w_vec
    final_preds = np.round(np.maximum(0.0, np.expm1(blend_log)), 2)
    
    res_df = pd.DataFrame({
        "id": input_df["id"].values if "id" in input_df.columns else range(len(input_df)),
        "date": input_df["date"].values,
        "store_id": input_df["store_id"].values,
        "is_promo": input_df["is_promo"].values,
        "revenue": final_preds
    })
    return res_df


def run_step8(
    data_dir: str = "data",
    output_dir: str = "artifacts",
    submission_path: str = "submission.csv",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Hàm điều phối thực thi trọn vẹn BƯỚC 8: DEPLOY.
    """
    if verbose:
        print("=" * 85)
        print("BƯỚC 8: DEPLOY (HUẤN LUYỆN LẠI 100%, XUẤT SUBMISSION.CSV & ĐÓNG GÓI ARTIFACTS)")
        print("=" * 85)
        print("Nguyên tắc: Huấn luyện lại trên 100% dữ liệu sạch để thu nạp trọn vẹn thông tin")
        print("và đóng gói sẵn sàng cho dịch vụ FastAPI + giao diện Streamlit.")
        print("-" * 85)
        
    # 1. Tải dữ liệu Bước 3
    train_clean, test_clean, stores_clean, weather_clean, holidays_clean, clean_stats = run_step3(data_dir, verbose=False)
    numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(train_clean)
    
    # Học StoreStatsEncoder trên 100% dữ liệu lịch sử sạch
    store_stats_encoder = DeSeasonalizedStoreStats(shrinkage_m=10.0, alpha_ridge=1.0)
    store_stats_encoder.fit(train_clean, train_clean["revenue"].values)
    
    if verbose:
        print(f"Dữ liệu huấn luyện lại 100%:  {len(train_clean):,} dòng (2024-01-01 -> 2025-12-31)")
        print(f"Dữ liệu kiểm tra Test:         {len(test_clean):,} dòng (2026-01-01 -> 2026-03-31)")
        print(f"Số lượng đặc trưng sử dụng:    {len(numeric_cols)} numeric + {len(categorical_cols)} categorical = {len(numeric_cols)+len(categorical_cols)} cột")
        print("-" * 85)
        
    # 2. Huấn luyện lại Ensemble trên 100% dữ liệu
    ensemble, fitted_pipes = retrain_ensemble_on_full_data(
        train_clean, numeric_cols, categorical_cols,
        ensemble_weights=DEFAULT_ENSEMBLE_WEIGHTS,
        verbose=verbose
    )
    
    # 3. Tạo submission.csv và kiểm tra Sanity Checks
    sub_df, sanity_results = generate_kaggle_submission(
        ensemble, test_clean, numeric_cols, categorical_cols,
        output_path=submission_path,
        verbose=verbose
    )
    
    # 4. Đóng gói Artifacts bundle và JSON metrics
    bundle_path = os.path.join(output_dir, "cafe_revenue_bundle.joblib")
    json_path = os.path.join(output_dir, "model_metrics_summary.json")
    
    bundle = package_artifact_bundle(
        ensemble, numeric_cols, categorical_cols,
        stores_clean, weather_clean, holidays_clean,
        store_stats_encoder=store_stats_encoder,
        bundle_path=bundle_path, json_path=json_path,
        verbose=verbose
    )
    
    # 5. Kiểm thử nhanh hàm dự báo đơn lẻ
    if verbose:
        print("-" * 85)
        print("8.4. KIỂM THỬ TÍNH NĂNG DỰ BÁO ĐƠN LẺ & WHAT-IF TRÊN QUÁN S01 (MÙNG 1 TẾT 2026):")
        test_pred = predict_single_instance(bundle, store_id="S01", date_str="2026-02-17", is_promo=1, temp_c=28.5, rain_mm=0.0)
        print(f"  * Cửa hàng:          {test_pred['store_id']} ({test_pred['store_type'].upper()}, {test_pred['city']})")
        print(f"  * Ngày:              {test_pred['date']} ({test_pred['day_of_week']})")
        print(f"  * Doanh thu dự báo:  {test_pred['predicted_revenue_m_vnd']} triệu VNĐ")
        print(f"  * Lợi nhuận thêm KM: {test_pred['promo_uplift_m_vnd']} triệu VNĐ ({test_pred['promo_uplift_pct']:+}%)")
        print(f"  * Khuyến nghị:       {test_pred['operational_recommendation']}")
        print("=" * 85)
        
    return {
        "ensemble": ensemble,
        "submission_df": sub_df,
        "sanity_results": sanity_results,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "json_path": json_path
    }


if __name__ == "__main__":
    results = run_step8(data_dir="data", output_dir="artifacts", submission_path="submission.csv", verbose=True)
