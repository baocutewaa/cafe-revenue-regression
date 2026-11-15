"""
Module: step4_split.py
Chủ đề: Dự báo doanh thu ngày chuỗi quán Cafe (Cafe Revenue Regression)
Mô tả: Triển khai toàn bộ BƯỚC 4: SPLIT (CHIẾN LƯỢC PHÂN CHIA DỮ LIỆU THEO CHUỖI THỜI GIAN)
Tuân thủ nguyên tắc bài giảng Chương 4: "Time-based split: hold out the future, never use random split on time-series."
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Generator
try:
    from .step1_collect_prepare import rmsle
except ImportError:
    from step1_collect_prepare import rmsle

def get_starter_holdout_split(
    df: pd.DataFrame,
    split_date: str = "2025-10-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Thiết lập Split 1 - Mốc chuẩn đối chiếu với Starter Notebook (Oct-Dec 2025 Holdout):
    - fit_df: Dữ liệu từ ngày đầu đến trước split_date (2024-01-01 -> 2025-09-30)
    - val_df: Dữ liệu Quý 4/2025 (2025-10-01 -> 2025-12-31)
    
    Mục đích:
    1. So sánh 1-1 với mốc Baseline RMSLE = 0.2387 của starter_cafe.ipynb.
    2. Đánh giá khả năng dự báo Cold-Start cho các quán mới mở muộn: S19 (11/10/2025), S01 (07/12/2025), S45 (11/12/2025).
    """
    ts = pd.Timestamp(split_date)
    fit_df = df[df["date"] < ts].copy().reset_index(drop=True)
    val_df = df[df["date"] >= ts].copy().reset_index(drop=True)
    
    assert fit_df["date"].max() < val_df["date"].min(), "Rò rỉ thời gian giữa fit và val!"
    return fit_df, val_df


def get_expanding_window_folds() -> List[Dict[str, Any]]:
    """
    Định nghĩa cấu hình 4 Folds của Kiểm định Chéo Chuỗi Thời gian Mở rộng
    (4-Fold Expanding Window Time-Series Cross Validation).
    """
    return [
        {
            "fold_id": 1,
            "fold_name": "Fold 1 (Seasonal Mirror — Q1/2025)",
            "train_start": "2024-01-01",
            "train_end": "2024-12-31",
            "val_start": "2025-01-01",
            "val_end": "2025-03-31",
            "description": "Perfect mirror of test.csv Q1/2026 structure: Contains full Lunar New Year 2025, Valentine, and Mar 8."
        },
        {
            "fold_id": 2,
            "fold_name": "Fold 2 (Rainy Season — Q2/2025)",
            "train_start": "2024-01-01",
            "train_end": "2025-03-31",
            "val_start": "2025-04-01",
            "val_end": "2025-06-30",
            "description": "Tests Hung Kings' holiday, Apr 30 - May 1, and the start of rainy season in HCMC."
        },
        {
            "fold_id": 3,
            "fold_name": "Fold 3 (Summer Vacation — Q3/2025)",
            "train_start": "2024-01-01",
            "train_end": "2025-06-30",
            "val_start": "2025-07-01",
            "val_end": "2025-09-30",
            "description": "Tests university summer break drop in July-August and National Day holiday."
        },
        {
            "fold_id": 4,
            "fold_name": "Fold 4 (Year-End Peak — Q4/2025)",
            "train_start": "2024-01-01",
            "train_end": "2025-09-30",
            "val_start": "2025-10-01",
            "val_end": "2025-12-31",
            "description": "Tests year-end holidays, Christmas, and cold-start for new stores S19, S01, S45."
        }
    ]


def plot_cv_timeline(output_path: str = "plots/time_series_cv_timeline.png") -> str:
    """
    Trực quan hóa sơ đồ 4-Fold Expanding Window Time-Series Cross-Validation hoàn toàn bằng tiếng Anh.
    """
    import matplotlib.pyplot as plt
    folds_config = get_expanding_window_folds()
    fig, ax = plt.subplots(figsize=(14, 5))
    fold_names = [f["fold_name"].split(" — ")[0] for f in folds_config]
    y_pos = np.arange(len(fold_names))
    
    for i, fold in enumerate(folds_config):
        tr_s = pd.Timestamp(fold["train_start"])
        tr_e = pd.Timestamp(fold["train_end"])
        v_s = pd.Timestamp(fold["val_start"])
        v_e = pd.Timestamp(fold["val_end"])
        
        ax.barh(i, (tr_e - tr_s).days, left=tr_s, height=0.45, color="#2980b9", alpha=0.85, label="Train Set (Expanding Window)" if i == 0 else "")
        val_color = "#e74c3c" if fold["fold_id"] == 1 else "#f39c12"
        ax.barh(i, (v_e - v_s).days, left=v_s, height=0.45, color=val_color, alpha=0.85, label="Validation (with Tet)" if fold["fold_id"] == 1 else ("Validation Set" if i == 1 else ""))

    ax.axvline(pd.Timestamp("2024-02-10"), color="red", linestyle=":", linewidth=1.5, label="Lunar New Year 2024")
    ax.axvline(pd.Timestamp("2025-01-29"), color="darkred", linestyle="--", linewidth=1.8, label="Lunar New Year 2025 (Fold 1 Mirror)")
    ax.axvline(pd.Timestamp("2026-02-17"), color="purple", linestyle="--", linewidth=1.8, label="Lunar New Year 2026 (Kaggle Test)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fold_names, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("4-Fold Expanding Window Time-Series Cross-Validation Architecture", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


class TimeSeriesExpandingCV:
    """
    Bộ chia fold kiểm định chéo tương thích chuẩn Scikit-Learn:
    Áp dụng kỹ thuật Cửa sổ Mở rộng theo Thời gian (Expanding Window):
    - Tập Train luôn bắt đầu từ ngày đầu tiên (2024-01-01) và mở rộng dần qua từng fold.
    - Tập Validation di chuyển tuần tự qua 4 quý của năm 2025.
    - Tuyệt đối không rò rỉ dữ liệu tương lai (Train max date < Val min date).
    """
    def __init__(self):
        self.folds_config = get_expanding_window_folds()
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.folds_config)
        
    def split(self, X: pd.DataFrame, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        dates = pd.to_datetime(X["date"])
        for fold in self.folds_config:
            tr_mask = (dates >= fold["train_start"]) & (dates <= fold["train_end"])
            val_mask = (dates >= fold["val_start"]) & (dates <= fold["val_end"])
            
            tr_idx = np.where(tr_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            yield tr_idx, val_idx


def compute_weekday_median_baseline(history: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """
    Mô hình Baseline từ starter_cafe.ipynb:
    Dự báo bằng trung vị của (store_id, dayofweek) từ tập history.
    Nếu cửa hàng chưa từng xuất hiện ở thứ đó, điền bằng trung vị chung của history.
    """
    h = history.assign(dow=history["date"].dt.dayofweek)
    med = h.groupby(["store_id", "dow"])["revenue"].median().rename("pred")
    
    t = target.assign(dow=target["date"].dt.dayofweek)
    pred_series = t.join(med, on=["store_id", "dow"])["pred"]
    
    # Điền khuyết nếu quán mới chưa có lịch sử ở thứ đó
    global_med = float(h["revenue"].median())
    return pred_series.fillna(global_med).values


def evaluate_baseline_across_splits(
    train_raw: pd.DataFrame,
    train_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Đánh giá mô hình Baseline (weekday_median) trên cả 2 chiến lược kiểm định:
    1. Split 1: Raw Validation (như starter_cafe.ipynb) và Clean Validation.
    2. Split 2: Cả 4 Folds của Expanding Window Time-Series CV.
    
    Minh chứng định lượng tại sao Starter Validation (Q4/2025) không đo được Tết,
    trong khi Fold 1 (Q1/2025) có RMSLE nhảy vọt vì chứa Tết Nguyên Đán!
    """
    records = []
    
    # --------------------------------------------------------------------------
    # Split 1: Starter Holdout
    # --------------------------------------------------------------------------
    # A. Raw Validation (giữ nguyên như trong starter notebook)
    fit_raw = train_raw[train_raw["date"] < "2025-10-01"].copy()
    val_raw = train_raw[train_raw["date"] >= "2025-10-01"].copy()
    pred_raw = compute_weekday_median_baseline(fit_raw, val_raw)
    raw_rmsle = rmsle(val_raw["revenue"].values, pred_raw)
    
    records.append({
        "Chiến lược kiểm định": "Split 1: Starter Holdout (Oct-Dec 2025)",
        "Tập đánh giá": "Raw Val (Chưa sửa 6 lỗi dấu phẩy)",
        "Số dòng Train": len(fit_raw),
        "Số dòng Val": len(val_raw),
        "Khoảng thời gian Val": "2025-10-01 đến 2025-12-31",
        "Có chứa Tết Nguyên Đán?": "KHÔNG",
        "Baseline RMSLE": raw_rmsle,
        "Ghi chú kinh doanh": "Mốc chuẩn 0.2387 từ starter notebook; bị phạt do 6 lỗi dấu phẩy & 3 quán mới S19, S01, S45."
    })
    
    # B. Clean Validation (đã sửa 6 lỗi dấu phẩy)
    fit_clean, val_clean = get_starter_holdout_split(train_clean, "2025-10-01")
    pred_clean = compute_weekday_median_baseline(fit_clean, val_clean)
    clean_rmsle = rmsle(val_clean["revenue"].values, pred_clean)
    
    records.append({
        "Chiến lược kiểm định": "Split 1: Starter Holdout (Oct-Dec 2025)",
        "Tập đánh giá": "Clean Val (Đã sửa 6 lỗi dấu phẩy)",
        "Số dòng Train": len(fit_clean),
        "Số dòng Val": len(val_clean),
        "Khoảng thời gian Val": "2025-10-01 đến 2025-12-31",
        "Có chứa Tết Nguyên Đán?": "KHÔNG",
        "Baseline RMSLE": clean_rmsle,
        "Ghi chú kinh doanh": "Mức chính xác thực chất của Baseline khi không bị nhiễu bởi lỗi nhập liệu ở tập Val."
    })
    
    # --------------------------------------------------------------------------
    # Split 2: 4 Folds Expanding Window
    # --------------------------------------------------------------------------
    cv = TimeSeriesExpandingCV()
    for fold in cv.folds_config:
        tr_df = train_clean[(train_clean["date"] >= fold["train_start"]) & (train_clean["date"] <= fold["train_end"])]
        v_df = train_clean[(train_clean["date"] >= fold["val_start"]) & (train_clean["date"] <= fold["val_end"])]
        
        p = compute_weekday_median_baseline(tr_df, v_df)
        score = rmsle(v_df["revenue"].values, p)
        
        has_tet = "CÓ (Tết Ất Tỵ 2025)" if fold["fold_id"] == 1 else "KHÔNG"
        records.append({
            "Chiến lược kiểm định": f"Split 2: {fold['fold_name']}",
            "Tập đánh giá": f"Fold {fold['fold_id']} Validation",
            "Số dòng Train": len(tr_df),
            "Số dòng Val": len(v_df),
            "Khoảng thời gian Val": f"{fold['val_start']} đến {fold['val_end']}",
            "Có chứa Tết Nguyên Đán?": has_tet,
            "Baseline RMSLE": score,
            "Ghi chú kinh doanh": fold["description"]
        })
        
    return pd.DataFrame(records)


def run_step4(data_dir: str = "data", verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, TimeSeriesExpandingCV, pd.DataFrame]:
    """
    Hàm thực thi toàn bộ BƯỚC 4: TIME-BASED VALIDATION SPLIT.
    
    Trả về:
        (fit_clean, val_clean, cv_splitter, baseline_eval_df)
    """
    try:
        from .step3_preprocess import run_step3
    except ImportError:
        from step3_preprocess import run_step3
    
    if verbose:
        print("=" * 80)
        print("BƯỚC 4: SPLIT (CHIẾN LƯỢC PHÂN CHIA DỮ LIỆU THEO CHUỖI THỜI GIAN)")
        print("=" * 80)
        print("Nguyên tắc: Time-based split để ngăn chặn 100% rò rỉ dữ liệu tương lai (Data Leakage).")
        print("Chiến lược: Kiểm định kép (Dual Validation: Starter Holdout + 4-Fold Expanding Window).")
        print("-" * 80)
        
    # 1. Đọc dữ liệu sạch sau Bước 3
    train_final, test_final, _, _, _, _ = run_step3(data_dir, verbose=False)
    train_raw = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    
    # 2. Thiết lập Split 1
    fit_clean, val_clean = get_starter_holdout_split(train_final, "2025-10-01")
    if verbose:
        print("4.1. CHIẾN LƯỢC 1 — STARTER NOTEBOOK HOLDOUT (OCT–DEC 2025):")
        print(f"  * Train set (fit): {fit_clean.shape[0]:,d} dòng ({fit_clean['date'].min().strftime('%Y-%m-%d')} đến {fit_clean['date'].max().strftime('%Y-%m-%d')})")
        print(f"  * Val set (val):   {val_clean.shape[0]:,d} dòng ({val_clean['date'].min().strftime('%Y-%m-%d')} đến {val_clean['date'].max().strftime('%Y-%m-%d')})")
        print("  * Kiểm tra rò rỉ:  Train max date < Val min date -> ĐẠT CHUẨN!")
        
    # 3. Thiết lập Split 2
    cv_splitter = TimeSeriesExpandingCV()
    if verbose:
        print("-" * 80)
        print("4.2. CHIẾN LƯỢC 2 — 4-FOLD EXPANDING WINDOW TIME-SERIES CV:")
        for fold in cv_splitter.folds_config:
            print(f"  * {fold['fold_name']}:")
            print(f"      - Train: {fold['train_start']} -> {fold['train_end']}")
            print(f"      - Val:   {fold['val_start']} -> {fold['val_end']}")
            
    # 4. Đánh giá Baseline qua các tập split
    eval_df = evaluate_baseline_across_splits(train_raw, train_final)
    if verbose:
        print("-" * 80)
        print("4.3. BẢNG SO SÁNH HIỆU NĂNG BASELINE (WEEKDAY MEDIAN) QUA CÁC SPLIT:")
        for _, r in eval_df.iterrows():
            print(f"  [{r['Tập đánh giá']:32s}] RMSLE: {r['Baseline RMSLE']:.4f} | Tết: {r['Có chứa Tết Nguyên Đán?']:20s}")
        print("=" * 80)
        print("HOÀN THÀNH TOÀN BỘ BƯỚC 4: TIME-BASED SPLIT THEO ĐÚNG KẾ HOẠCH!")
        print("=" * 80)
        
    return fit_clean, val_clean, cv_splitter, eval_df


if __name__ == "__main__":
    run_step4()
