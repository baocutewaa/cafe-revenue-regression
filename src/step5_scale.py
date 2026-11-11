"""
Module: step5_scale.py
Chủ đề: Dự báo doanh thu ngày chuỗi quán Cafe (Cafe Revenue Regression)
Mô tả: Triển khai toàn bộ BƯỚC 5: SCALE (CHUẨN HÓA THANG ĐO & ĐÓNG GÓI PIPELINE CHỐNG RÒ RỈ)
Tuân thủ nguyên tắc bài giảng Chương 4: "Scale features inside pipeline. Never fit scalers on full data."
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

try:
    from .step1_collect_prepare import rmsle
except ImportError:
    from step1_collect_prepare import rmsle

# ------------------------------------------------------------------------------
# 1. BIẾN ĐỔI MỤC TIÊU SANG THANG LOGARIT
# ------------------------------------------------------------------------------
def transform_target(y: np.ndarray) -> np.ndarray:
    """
    Biến đổi biến mục tiêu sang thang logarit: y_log = ln(1 + max(0, y)).
    Phù hợp trực tiếp với hàm mất mát MSE để tối ưu hóa chỉ số RMSLE.
    """
    y_arr = np.asarray(y, dtype=float)
    return np.log1p(np.clip(y_arr, 0.0, None))


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ngược từ thang logarit về đơn vị triệu VNĐ: y = max(0, exp(y_log) - 1).
    """
    y_log_arr = np.asarray(y_log, dtype=float)
    return np.clip(np.expm1(y_log_arr), 0.0, None)


class TrendAwareLogTargetWrapper(BaseEstimator, RegressorMixin):
    """
    Lớp bao bọc hồi quy chuẩn hóa kết hợp Khử xu thế thời gian (De-trending)
    và Biến đổi logarit biến mục tiêu ln(1 + y) trực tiếp:
    - Trong fit(X, y):
      1. Trích xuất t = (date - 2024-01-01) / 365.25 (đơn vị năm).
      2. Học hệ số xu thế tuyến tính: ln(1 + y) = alpha + beta * t.
      3. Biến đổi mục tiêu dừng: y* = ln(1 + y) - beta * t.
      4. Huấn luyện base_pipeline trên (X, y*).
    - Trong predict(X):
      1. Dự báo y*_pred từ base_pipeline.
      2. Tái lập xu thế tương lai: ln(1 + y) = y*_pred + beta * t.
      3. Chuyển đổi ngược về doanh thu thực: y = max(0, exp(ln(1 + y)) - 1).
    """
    def __init__(self, regressor: Any):
        self.regressor = regressor
        self.regressor_ = None
        self.slope_ = 0.0
        self.intercept_ = 0.0

    def _extract_t(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if "t_years" in X.columns:
                return X["t_years"].values.astype(float)
            elif "date" in X.columns:
                dt = pd.to_datetime(X["date"])
                return (dt - pd.Timestamp("2024-01-01")).dt.days.values / 365.25
        return np.zeros(len(X), dtype=float)

    def fit(self, X: Any, y: np.ndarray):
        t = self._extract_t(X)
        y_arr = np.asarray(y, dtype=float)
        y_log = np.log1p(np.clip(y_arr, 0.0, None))
        
        if np.ptp(t) > 0.01:
            if isinstance(X, pd.DataFrame) and "store_id" in X.columns and X["store_id"].nunique() > 1:
                # Within-Store Panel Fixed Effects (Store-demeaned OLS slope):
                # Loại bỏ độ lệch cơ cấu (composition bias) từ việc mở các quán mới trong năm 2025.
                # Chuẩn hóa khử trung bình theo quán cho CẢ y_log và t để đạt ước lượng OLS Within-Estimator kinh tế lượng chuẩn xác.
                s_mean_y = pd.Series(y_log, index=X.index).groupby(X["store_id"]).transform("mean").values
                s_mean_t = pd.Series(t, index=X.index).groupby(X["store_id"]).transform("mean").values
                y_c = y_log - s_mean_y
                t_c = t - s_mean_t
                denom = np.sum(t_c ** 2)
                if denom > 1e-6:
                    self.slope_ = float(np.sum(t_c * y_c) / denom)
                    self.intercept_ = float(np.mean(y_log) - self.slope_ * np.mean(t))
                else:
                    self.slope_, self.intercept_ = 0.0, float(np.mean(y_log))
            else:
                self.slope_, self.intercept_ = np.polyfit(t, y_log, 1)
        else:
            self.slope_, self.intercept_ = 0.0, float(np.mean(y_log))
            
        y_star = y_log - self.slope_ * t
        self.regressor_ = self.regressor
        self.regressor_.fit(X, y_star)
        return self

    def predict(self, X: Any) -> np.ndarray:
        t = self._extract_t(X)
        y_star_pred = self.regressor_.predict(X)
        pred_log = y_star_pred + self.slope_ * t
        return np.clip(np.expm1(pred_log), 0.0, None)


class LogTargetRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Lớp bao bọc mô hình hồi quy với biến đổi logarit mục tiêu:
    - Trong .fit(X, y): tự động biến đổi y_log = ln(1 + y) và huấn luyện mô hình cơ sở.
    - Trong .predict(X): nhận dự báo y_pred_log và chuyển ngược lại về triệu VNĐ bằng expm1(y_log).
    """
    def __init__(self, regressor: Any):
        self.regressor = regressor
        self.regressor_ = None
        
    def fit(self, X: Any, y: np.ndarray):
        y_log = transform_target(y)
        self.regressor_ = self.regressor
        self.regressor_.fit(X, y_log)
        return self
        
    def predict(self, X: Any) -> np.ndarray:
        pred_log = self.regressor_.predict(X)
        return inverse_transform_target(pred_log)


class MultiSeedBaggedPipeline(BaseEstimator, RegressorMixin):
    """
    Bộ bao bọc Multi-Seed Bagging cho danh sách các mô hình Pipeline:
    Huấn luyện nhiều mô hình với các hạt ngẫu nhiên (random seeds) khác nhau,
    sau đó tổng hợp dự báo bằng trung bình logarit nhằm triệt tiêu phương sai (variance)
    và chống rung lắc (shakeup) tối đa trên tập Private Leaderboard (71%).
    """
    def __init__(self, pipelines: Optional[List[Any]] = None):
        self.pipelines = pipelines if pipelines is not None else []
        
    def fit(self, X: Any, y: np.ndarray):
        for p in self.pipelines:
            p.fit(X, y)
        return self
        
    def predict(self, X: Any) -> np.ndarray:
        if not self.pipelines:
            return np.zeros(len(X))
        preds_log = [np.log1p(np.maximum(0.0, p.predict(X))) for p in self.pipelines]
        avg_log = np.mean(preds_log, axis=0)
        return np.maximum(0.0, np.expm1(avg_log))


# ------------------------------------------------------------------------------
# 2. PHÂN LOẠI CÁC CỘT ĐẶC TRƯNG & XÂY DỰNG COLUMNTANSFORMER
# ------------------------------------------------------------------------------
def get_feature_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Phân loại các cột đặc trưng trong bảng dữ liệu:
    - categorical_cols: Các biến phân loại chữ cần One-Hot Encoding (city, district, store_type, holiday_name_clean).
    - numeric_cols: Các biến số thực, biến đếm và cờ nhị phân cần chuẩn hóa (StandardScaler) cho các mô hình tuyến tính/KNN/SVM/MLP.
    - ignore_cols: Các cột định danh, ngày tháng hoặc nhãn mục tiêu không đưa trực tiếp vào ma trận đặc trưng.
    """
    categorical_cols = ["city", "district", "store_type", "holiday_name_clean"]
    
    ignore_cols = [
        "revenue", "date", "open_date", "open_date_dt", "store_id", "id", "year", "t_years", "days_from_tet", "days_from_tet_exact"
    ]
    
    numeric_cols = [
        col for col in df.columns
        if col not in categorical_cols and col not in ignore_cols
    ]
    
    return numeric_cols, categorical_cols, ignore_cols


def build_feature_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler_type: str = "standard"
) -> ColumnTransformer:
    """
    Xây dựng ColumnTransformer chuẩn Scikit-Learn:
    - Nhánh số (numeric): SimpleImputer(strategy='median') -> StandardScaler (hoặc RobustScaler).
    - Nhánh phân loại (categorical): OneHotEncoder(handle_unknown='ignore', sparse_output=False).
    - Các cột khác (remainder): Bỏ qua ('drop').
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])
    
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor


def build_leakage_free_pipeline(
    regressor: Any,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler_type: str = "standard",
    use_log_target: bool = True,
    use_trend_aware: bool = False
) -> Any:
    """
    Đóng gói toàn bộ quy trình Chuẩn hóa thang đo (Scale) và Mô hình dự báo (Model)
    vào một Pipeline duy nhất tuân thủ nguyên tắc CHỐNG RÒ RỈ DỮ LIỆU:
    - Preprocessor chỉ fit trên tập Train.
    - Scaler chỉ tính mean và std trên tập Train.
    - use_trend_aware=True: Tự động khử xu thế (De-trend) trên không gian ln(1 + y) và tái hợp xu thế (Re-trend).
    - use_log_target=True: Target tự động biến đổi logarit np.log1p và chuyển ngược expm1.
    """
    preprocessor = build_feature_preprocessor(numeric_cols, categorical_cols, scaler_type=scaler_type)
    
    base_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", regressor)
    ])
    
    if use_trend_aware:
        return TrendAwareLogTargetWrapper(base_pipeline)
    elif use_log_target:
        return TransformedTargetRegressor(
            regressor=base_pipeline,
            func=transform_target,
            inverse_func=inverse_transform_target
        )
    else:
        return base_pipeline


def run_step5(data_dir: str = "data", verbose: bool = True) -> Tuple[ColumnTransformer, Any, Dict[str, Any]]:
    """
    Hàm thực thi toàn bộ BƯỚC 5: SCALE (CHUẨN HÓA THANG ĐO & ĐÓNG GÓI PIPELINE CHỐNG RÒ RỈ).
    
    Trả về:
        (preprocessor, fitted_pipeline, metrics_dict)
    """
    try:
        from .step3_preprocess import run_step3
        from .step4_split import get_starter_holdout_split
    except ImportError:
        from step3_preprocess import run_step3
        from step4_split import get_starter_holdout_split
    
    if verbose:
        print("=" * 80)
        print("BƯỚC 5: SCALE (CHUẨN HÓA THANG ĐO & ĐÓNG GÓI PIPELINE CHỐNG RÒ RỈ)")
        print("=" * 80)
        print("Nguyên tắc vàng: Chuẩn hóa thang đo bên trong Pipeline; chỉ fit trên Train.")
        print("Mục tiêu: Đóng gói Preprocessor và kiểm tra độ chính xác của mô hình Ridge.")
        print("-" * 80)
        
    # 1. Đọc dữ liệu sau Bước 3 & 4
    train_final, test_final, _, _, _, _ = run_step3(data_dir, verbose=False)
    fit_df, val_df = get_starter_holdout_split(train_final, split_date="2025-10-01")
    
    # 2. Phân loại cột
    numeric_cols, categorical_cols, ignore_cols = get_feature_column_types(fit_df)
    if verbose:
        print("5.1. PHÂN LOẠI DANH MỤC CÁC ĐẶC TRƯNG:")
        print(f"  * Số biến số (Numeric Scaled):       {len(numeric_cols)} biến (diện tích, ghế, nhiệt độ, mưa, chu kỳ, tỷ số...)")
        print(f"  * Số biến phân loại (One-Hot Encoded): {len(categorical_cols)} biến ({categorical_cols})")
        print(f"  * Số biến định danh bỏ qua:           {len(ignore_cols)} biến ({ignore_cols})")
        
    # 3. Xây dựng và kiểm tra ColumnTransformer
    preprocessor = build_feature_preprocessor(numeric_cols, categorical_cols)
    X_fit = fit_df.drop(columns=["revenue"])
    y_fit = fit_df["revenue"].values
    
    X_val = val_df.drop(columns=["revenue"])
    y_val = val_df["revenue"].values
    
    X_fit_trans = preprocessor.fit_transform(X_fit, y_fit)
    X_val_trans = preprocessor.transform(X_val)
    
    if verbose:
        print("-" * 80)
        print("5.2. KIỂM TRA TÍNH TOÀN VẸN CỦA MA TRẬN ĐẶC TRƯNG:")
        print(f"  * Kích thước X_fit sau biến đổi: {X_fit_trans.shape} (22.668 dòng × {X_fit_trans.shape[1]} cột one-hot & scaled)")
        print(f"  * Kích thước X_val sau biến đổi: {X_val_trans.shape} (3.832 dòng × {X_val_trans.shape[1]} cột)")
        print(f"  * Số lượng giá trị NaN ở X_fit:  {np.isnan(X_fit_trans).sum()}")
        print(f"  * Số lượng giá trị NaN ở X_val:  {np.isnan(X_val_trans).sum()}")
        print("  * Xác nhận: Scaler chỉ học mean và std trên X_fit, áp dụng sang X_val -> KHÔNG RÒ RỈ!")
        
    # 4. Thử nghiệm huấn luyện mô hình Ridge với Pipeline hoàn chỉnh
    ridge_pipeline = build_leakage_free_pipeline(
        regressor=Ridge(alpha=1.0),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scaler_type="standard",
        use_log_target=True
    )
    
    ridge_pipeline.fit(X_fit, y_fit)
    val_pred = ridge_pipeline.predict(X_val)
    val_rmsle = rmsle(y_val, val_pred)
    
    # Thử nghiệm trên Fold 1 (Jan-Mar 2025 có Tết)
    tr_fold1 = train_final[(train_final["date"] >= "2024-01-01") & (train_final["date"] <= "2024-12-31")]
    val_fold1 = train_final[(train_final["date"] >= "2025-01-01") & (train_final["date"] <= "2025-03-31")]
    
    f1_pipeline = build_leakage_free_pipeline(
        regressor=Ridge(alpha=1.0),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scaler_type="standard",
        use_log_target=True
    )
    f1_pipeline.fit(tr_fold1.drop(columns=["revenue"]), tr_fold1["revenue"].values)
    pred_f1 = f1_pipeline.predict(val_fold1.drop(columns=["revenue"]))
    fold1_rmsle = rmsle(val_fold1["revenue"].values, pred_f1)
    
    metrics_dict = {
        "num_features_in": len(numeric_cols) + len(categorical_cols),
        "num_features_out": X_fit_trans.shape[1],
        "ridge_val_rmsle": float(val_rmsle),
        "ridge_fold1_tet_rmsle": float(fold1_rmsle),
        "starter_baseline_rmsle": 0.2387,
        "starter_clean_val_rmsle": 0.2276,
        "fold1_baseline_rmsle": 0.2761,
        "improvement_pct_val": float(((0.2276 - val_rmsle) / 0.2276) * 100.0),
        "improvement_pct_fold1": float(((0.2761 - fold1_rmsle) / 0.2761) * 100.0)
    }
    
    if verbose:
        print("-" * 80)
        print("5.3. KẾT QUẢ ĐỐI CHIẾU HIỆU NĂNG MÔ HÌNH RIDGE VỚI BASELINE:")
        print(f"  * Validation Quý 4/2025 (Holdout):")
        print(f"      - Baseline (weekday_median): 0.2276")
        print(f"      - Ridge Pipeline (Bước 5):   {val_rmsle:.4f} (GIẢM -{metrics_dict['improvement_pct_val']:.1f}% RMSLE!)")
        print(f"  * Validation Quý 1/2025 (Chứa Tết Nguyên Đán):")
        print(f"      - Baseline (weekday_median): 0.2761")
        print(f"      - Ridge Pipeline (Bước 5):   {fold1_rmsle:.4f} (GIẢM -{metrics_dict['improvement_pct_fold1']:.1f}% RMSLE!)")
        print("=" * 80)
        print("HOÀN THÀNH TOÀN BỘ BƯỚC 5: SCALE THEO ĐÚNG KẾ HOẠCH!")
        print("=" * 80)
        
    return preprocessor, ridge_pipeline, metrics_dict


if __name__ == "__main__":
    run_step5()
