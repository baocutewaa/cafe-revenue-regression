"""
FASTAPI BACKEND SERVICE: app/api_server.py
Cafe Revenue Forecasting Multi-Store Platform
"""

import os
import sys
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Đảm bảo đường dẫn gốc và src có mặt trong sys.path để unpickle và import
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Alias sys.modules để hỗ trợ unpickle nếu pickle lưu dưới tên cục bộ
import src.step3_preprocess as step3_preprocess
import src.step5_scale as step5_scale
import src.step6_model_train as step6_model_train
sys.modules["step3_preprocess"] = step3_preprocess
sys.modules["step5_scale"] = step5_scale
sys.modules["step6_model_train"] = step6_model_train

from src.step8_deploy import predict_single_instance, predict_batch_dataframe, run_step8

# Biến toàn cục lưu trữ tài nguyên mô hình
BUNDLE_PATH = os.path.join(ROOT_DIR, "artifacts", "cafe_revenue_bundle.joblib")
global_bundle: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Tải gói tài nguyên model bundle vào RAM một lần duy nhất khi khởi động server.
    """
    global global_bundle
    print(f"[Startup] Đang nạp tài nguyên từ {BUNDLE_PATH}...")
    if not os.path.exists(BUNDLE_PATH):
        print(f"[Startup] Không tìm thấy {BUNDLE_PATH}. Tiến hành tạo gói tự động qua run_step8()...")
        run_step8(data_dir=os.path.join(ROOT_DIR, "data"), output_dir=os.path.join(ROOT_DIR, "artifacts"), verbose=False)
        
    global_bundle = joblib.load(BUNDLE_PATH)
    print(f"[Startup] Nạp thành công {len(global_bundle['models'])} mô hình thành phần và 45 cửa hàng tham chiếu!")
    yield
    print("[Shutdown] Giải phóng tài nguyên server.")


app = FastAPI(
    title="Cafe Revenue Forecasting REST API",
    description="Hệ thống API dự báo doanh thu chuỗi Cafe đa cửa hàng kết hợp kịch bản What-If và lập kế hoạch vận hành.",
    version="1.0.0",
    lifespan=lifespan
)

# Cho phép CORS để Streamlit Frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================
class SinglePredictRequest(BaseModel):
    store_id: str = Field(..., example="S01", description="Mã cửa hàng (S01 - S45)")
    date: str = Field(..., example="2026-02-17", description="Ngày dự báo định dạng YYYY-MM-DD")
    is_promo: int = Field(0, ge=0, le=1, example=1, description="Trạng thái khuyến mãi (0: Không, 1: Có)")
    temp_c: Optional[float] = Field(None, example=28.5, description="Nhiệt độ ngày (°C), nếu bỏ trống sẽ tự tra cứu")
    rain_mm: Optional[float] = Field(None, example=0.0, description="Lượng mưa ngày (mm), nếu bỏ trống sẽ tự tra cứu")


class SinglePredictResponse(BaseModel):
    store_id: str
    date: str
    day_of_week: str
    store_type: str
    city: str
    seats: int
    area_m2: float
    is_promo: int
    temp_c: float
    rain_mm: float
    predicted_revenue_m_vnd: float
    component_predictions: Dict[str, float]
    what_if_opposite_promo_pred: float
    promo_uplift_m_vnd: float
    promo_uplift_pct: float
    operational_recommendation: str


class WhatIfRequest(BaseModel):
    store_id: str = Field(..., example="S29", description="Mã cửa hàng (S01 - S45)")
    date: str = Field(..., example="2026-02-10", description="Ngày dự báo (YYYY-MM-DD)")
    base_is_promo: int = Field(0, ge=0, le=1, example=0)
    base_temp_c: Optional[float] = Field(None, example=27.0)
    base_rain_mm: Optional[float] = Field(None, example=0.0)
    new_is_promo: int = Field(1, ge=0, le=1, example=1)
    new_temp_c: Optional[float] = Field(None, example=27.0)
    new_rain_mm: Optional[float] = Field(None, example=35.0)


class WhatIfResponse(BaseModel):
    store_id: str
    date: str
    base_scenario: Dict[str, Any]
    new_scenario: Dict[str, Any]
    revenue_delta_m_vnd: float
    revenue_delta_pct: float
    summary: str


class BatchPredictItem(BaseModel):
    id: Optional[int] = Field(None, example=100000)
    date: str = Field(..., example="2026-01-01")
    store_id: str = Field(..., example="S01")
    is_promo: int = Field(0, ge=0, le=1, example=0)


class BatchPredictResponseItem(BaseModel):
    id: Optional[int]
    date: str
    store_id: str
    is_promo: int
    revenue: float


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/", summary="Root endpoint")
def root():
    return {
        "message": "Chào mừng đến với Cafe Revenue Forecasting API!",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", summary="Kiểm tra trạng thái hệ thống & thông tin mô hình")
def health_check():
    if global_bundle is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được nạp vào bộ nhớ.")
        
    metrics = global_bundle.get("metrics", {})
    return {
        "status": "healthy",
        "service": "Cafe Revenue Forecasting Service",
        "model_version": "1.0.0",
        "algorithm": "Weighted Blend Ensemble (Ridge + LightGBM + CatBoost + XGBoost)",
        "holdout_val_rmsle": metrics.get("holdout_val_rmsle", 0.1218),
        "seasonal_tet_rmsle": metrics.get("seasonal_tet_rmsle", 0.1459),
        "val_mae_m_vnd": metrics.get("val_mae_m_vnd", 2.03),
        "val_r2": metrics.get("val_r2", 0.8950),
        "ensemble_weights": global_bundle.get("weights", {})
    }


@app.get("/stores", summary="Danh sách 45 cửa hàng và thuộc tính tĩnh")
def get_stores(
    city: Optional[str] = Query(None, description="Lọc theo thành phố ('Ha Noi' hoặc 'Ho Chi Minh')"),
    store_type: Optional[str] = Query(None, description="Lọc theo loại vị trí ('mall', 'street', 'office', 'university', 'residential')")
):
    if global_bundle is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được nạp vào bộ nhớ.")
        
    stores_df = global_bundle["stores_lookup"].copy()
    if city:
        stores_df = stores_df[stores_df["city"].str.lower() == city.lower()]
    if store_type:
        stores_df = stores_df[stores_df["store_type"].str.lower() == store_type.lower()]
        
    records = []
    for _, r in stores_df.iterrows():
        records.append({
            "store_id": r["store_id"],
            "city": r["city"],
            "district": r["district"],
            "store_type": r["store_type"],
            "area_m2": float(r["area_m2"]),
            "seats": int(r["seats"]),
            "has_parking": int(r["has_parking"]),
            "open_date": str(r["open_date"])[:10]
        })
    return {"total": len(records), "stores": records}


@app.post("/predict/single", response_model=SinglePredictResponse, summary="Dự báo doanh thu 1 ngày + Phân tích kịch bản")
def predict_single(req: SinglePredictRequest):
    if global_bundle is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")
        
    try:
        res = predict_single_instance(
            global_bundle,
            store_id=req.store_id,
            date_str=req.date,
            is_promo=req.is_promo,
            temp_c=req.temp_c,
            rain_mm=req.rain_mm
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/what-if", response_model=WhatIfResponse, summary="Mô phỏng kịch bản What-If Trước vs Sau khi thay đổi điều kiện")
def predict_what_if(req: WhatIfRequest):
    if global_bundle is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")
        
    try:
        base_res = predict_single_instance(
            global_bundle, store_id=req.store_id, date_str=req.date,
            is_promo=req.base_is_promo, temp_c=req.base_temp_c, rain_mm=req.base_rain_mm
        )
        new_res = predict_single_instance(
            global_bundle, store_id=req.store_id, date_str=req.date,
            is_promo=req.new_is_promo, temp_c=req.new_temp_c, rain_mm=req.new_rain_mm
        )
        
        delta = round(new_res["predicted_revenue_m_vnd"] - base_res["predicted_revenue_m_vnd"], 2)
        base_rev = max(0.1, base_res["predicted_revenue_m_vnd"])
        delta_pct = round((delta / base_rev) * 100, 1)
        
        sign = "+" if delta >= 0 else ""
        summary_msg = (
            f"Kịch bản mới làm thay đổi doanh thu: {sign}{delta} triệu VNĐ ({sign}{delta_pct}%). "
            f"Từ {base_res['predicted_revenue_m_vnd']} lên {new_res['predicted_revenue_m_vnd']} triệu VNĐ."
        )
        
        return {
            "store_id": req.store_id,
            "date": req.date,
            "base_scenario": {
                "is_promo": req.base_is_promo,
                "temp_c": base_res["temp_c"],
                "rain_mm": base_res["rain_mm"],
                "predicted_revenue_m_vnd": base_res["predicted_revenue_m_vnd"]
            },
            "new_scenario": {
                "is_promo": req.new_is_promo,
                "temp_c": new_res["temp_c"],
                "rain_mm": new_res["rain_mm"],
                "predicted_revenue_m_vnd": new_res["predicted_revenue_m_vnd"]
            },
            "revenue_delta_m_vnd": delta,
            "revenue_delta_pct": delta_pct,
            "summary": summary_msg
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=List[BatchPredictResponseItem], summary="Dự báo hàng loạt theo danh sách bản ghi")
def predict_batch(records: List[BatchPredictItem]):
    if global_bundle is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")
    if len(records) == 0:
        return []
        
    try:
        raw_list = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in records]
        input_df = pd.DataFrame(raw_list)
        pred_df = predict_batch_dataframe(global_bundle, input_df)
        
        results = []
        for _, row in pred_df.iterrows():
            results.append({
                "id": int(row["id"]) if "id" in row and pd.notnull(row["id"]) else None,
                "date": str(row["date"])[:10],
                "store_id": row["store_id"],
                "is_promo": int(row["is_promo"]),
                "revenue": float(row["revenue"])
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
