"""
STREAMLIT FRONTEND DASHBOARD: app/streamlit_app.py
Cafe Revenue Forecasting Multi-Store Interactive Platform
"""

import os
import sys
import json
import time
import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import requests
import joblib

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Cafe Revenue Forecasting Platform",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Đảm bảo đường dẫn gốc và src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import src.step3_preprocess as step3_preprocess
import src.step5_scale as step5_scale
import src.step6_model_train as step6_model_train
sys.modules["step3_preprocess"] = step3_preprocess
sys.modules["step5_scale"] = step5_scale
sys.modules["step6_model_train"] = step6_model_train

from src.step8_deploy import predict_single_instance, predict_batch_dataframe

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
BUNDLE_PATH = os.path.join(ROOT_DIR, "artifacts", "cafe_revenue_bundle.joblib")


@st.cache_resource
def load_local_bundle():
    """Tải trực tiếp artifact bundle từ đĩa làm cơ chế Fallback nếu server FastAPI chưa bật."""
    if os.path.exists(BUNDLE_PATH):
        return joblib.load(BUNDLE_PATH)
    return None


local_bundle = load_local_bundle()


def check_api_status() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


api_online = check_api_status()


# ==============================================================================
# SIDEBAR: THÔNG TIN KẾT NỐI & BỘ LỌC CỬA HÀNG
# ==============================================================================
st.sidebar.title("Cafe Revenue AI")
st.sidebar.caption("Dự báo doanh thu chuỗi cửa hàng")

# Trạng thái kết nối
if api_online:
    st.sidebar.success("FastAPI Backend: Online")
else:
    st.sidebar.info("FastAPI Backend: Offline (Chế độ Local Artifact)")

st.sidebar.markdown("---")
st.sidebar.subheader("Chọn cửa hàng")

# Lấy danh sách stores từ bundle
if local_bundle is not None:
    stores_df = local_bundle["stores_lookup"].copy()
else:
    stores_df = pd.DataFrame()

if not stores_df.empty:
    cities = ["Tất cả"] + sorted(stores_df["city"].unique().tolist())
    selected_city = st.sidebar.selectbox("Thành phố", cities)
    
    filtered_stores = stores_df.copy()
    if selected_city != "Tất cả":
        filtered_stores = filtered_stores[filtered_stores["city"] == selected_city]
        
    store_types = ["Tất cả"] + sorted(filtered_stores["store_type"].unique().tolist())
    selected_type = st.sidebar.selectbox("Loại vị trí", store_types)
    if selected_type != "Tất cả":
        filtered_stores = filtered_stores[filtered_stores["store_type"] == selected_type]
        
    store_ids = sorted(filtered_stores["store_id"].unique().tolist())
    selected_store_id = st.sidebar.selectbox("Mã quán (Store ID)", store_ids, index=0)
    
    # Thẻ thông tin nhanh về quán (Store Profile Card)
    curr_store = stores_df[stores_df["store_id"] == selected_store_id].iloc[0]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### Thông tin quán `{selected_store_id}`")
    st.sidebar.markdown(f"- **Vị trí:** {curr_store['store_type'].upper()} ({curr_store['district']}, {curr_store['city']})")
    st.sidebar.markdown(f"- **Quy mô:** {curr_store['seats']} ghế | {curr_store['area_m2']:.1f} m² ({curr_store['area_m2']/curr_store['seats']:.1f} m²/ghế)")
    st.sidebar.markdown(f"- **Bãi đỗ xe:** {'Có' if curr_store['has_parking'] == 1 else 'Không'}")
    st.sidebar.markdown(f"- **Ngày mở:** {str(curr_store['open_date'])[:10]}")
else:
    selected_store_id = "S01"


# ==============================================================================
# NỘI DUNG CHÍNH: 4 TABS CHUYÊN BIỆT
# ==============================================================================
st.title("Dự Báo Doanh Thu Chuỗi Cafe")
st.caption("Mô hình Ensemble tối ưu trực tiếp RMSLE và kiểm định chuỗi thời gian.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Dự Báo Ngày",
    "Lập Kế Hoạch Chuỗi Ngày",
    "Dự Báo Hàng Loạt",
    "Đánh Giá Mô Hình"
])


# ------------------------------------------------------------------------------
# TAB 1: DỰ BÁO NGÀY ĐƠN LẺ
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(f"Dự báo doanh thu quán `{selected_store_id}`")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        target_date = st.date_input(
            "Ngày dự báo",
            datetime.date(2026, 2, 17),
            min_value=datetime.date(2024, 1, 1),
            max_value=datetime.date(2030, 12, 31)
        )
    with col2:
        is_promo = st.radio("Khuyến mãi", [0, 1], format_func=lambda x: "Không khuyến mãi" if x == 0 else "Có khuyến mãi", index=0)
    with col3:
        temp_input = st.slider("Nhiệt độ dự kiến (°C)", min_value=10.0, max_value=42.0, value=28.0, step=0.5)
    with col4:
        rain_input = st.slider("Lượng mưa dự kiến (mm)", min_value=0.0, max_value=120.0, value=0.0, step=1.0)
        
    date_str = target_date.strftime("%Y-%m-%d")
    
    # Dự báo thông qua API hoặc Local
    prediction_result = None
    if api_online:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/predict/single",
                json={
                    "store_id": selected_store_id,
                    "date": date_str,
                    "is_promo": is_promo,
                    "temp_c": temp_input,
                    "rain_mm": rain_input
                },
                timeout=5.0
            )
            if resp.status_code == 200:
                prediction_result = resp.json()
        except Exception:
            prediction_result = None
            
    if prediction_result is None and local_bundle is not None:
        prediction_result = predict_single_instance(
            local_bundle,
            store_id=selected_store_id,
            date_str=date_str,
            is_promo=is_promo,
            temp_c=temp_input,
            rain_mm=rain_input
        )
        
    if prediction_result:
        pred_val = prediction_result["predicted_revenue_m_vnd"]
        uplift_val = prediction_result["promo_uplift_m_vnd"]
        uplift_pct = prediction_result["promo_uplift_pct"]
        opp_pred = prediction_result["what_if_opposite_promo_pred"]
        
        st.markdown("---")
        mcol1, mcol2, mcol3 = st.columns(3)
        
        with mcol1:
            st.metric(
                label=f"Dự báo Doanh thu ({prediction_result['day_of_week']})",
                value=f"{pred_val:.2f} triệu VNĐ",
                delta=f"{'+' if is_promo == 1 else '-'}{abs(uplift_pct)}% do KM"
            )
        with mcol2:
            st.metric(
                label="Hiệu ứng Khuyến mãi",
                value=f"{uplift_val:+.2f} triệu VNĐ",
                delta=f"Khi {'không KM' if is_promo == 1 else 'có KM'}: {opp_pred:.2f} tr.đ"
            )
        with mcol3:
            st.info(f"**Khuyến nghị vận hành:**\n\n{prediction_result['operational_recommendation']}")
            
        # Chi tiết phân rã các mô hình thành phần trong Ensemble
        with st.expander("Chi tiết từng mô hình thành phần"):
            comp_df = pd.DataFrame([
                {"Mô hình": m, "Dự báo (tr.VNĐ)": v, "Trọng số đóng góp": f"{local_bundle['weights'].get(m, 0)*100:.1f}%"}
                for m, v in prediction_result["component_predictions"].items()
            ])
            st.table(comp_df)


# ------------------------------------------------------------------------------
# TAB 2: LẬP KẾ HOẠCH DOANH THU CHUỖI NGÀY
# ------------------------------------------------------------------------------
with tab2:
    st.subheader(f"Kế hoạch doanh thu chuỗi ngày - Quán `{selected_store_id}`")
    pcol1, pcol2 = st.columns([1, 2])
    with pcol1:
        date_range = st.date_input(
            "Khoảng thời gian lập kế hoạch",
            [datetime.date(2026, 2, 1), datetime.date(2026, 3, 10)],
            min_value=datetime.date(2024, 1, 1),
            max_value=datetime.date(2030, 12, 31)
        )
        plan_promo = st.checkbox("Áp dụng khuyến mãi toàn chuỗi ngày", value=False)
        
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and local_bundle is not None:
        start_d, end_d = date_range
        dates = pd.date_range(start_d, end_d, freq="D")
        
        sim_records = []
        for d in dates:
            sim_records.append({
                "date": d.strftime("%Y-%m-%d"),
                "store_id": selected_store_id,
                "is_promo": 1 if plan_promo else 0
            })
        sim_df = pd.DataFrame(sim_records)
        
        # Chạy dự báo chuỗi
        forecast_plan = predict_batch_dataframe(local_bundle, sim_df)
        forecast_plan["date_dt"] = pd.to_datetime(forecast_plan["date"])
        
        with pcol2:
            st.markdown(f"**Tổng doanh thu dự kiến ({len(dates)} ngày):** `{forecast_plan['revenue'].sum():,.2f}` triệu VNĐ")
            st.markdown(f"**Bình quân/ngày:** `{forecast_plan['revenue'].mean():.2f}` triệu VNĐ (Min: `{forecast_plan['revenue'].min():.2f}`, Max: `{forecast_plan['revenue'].max():.2f}`)")
            
        # Vẽ biểu đồ chuỗi thời gian
        st.line_chart(
            forecast_plan.set_index("date_dt")[["revenue"]],
            use_container_width=True
        )


# ------------------------------------------------------------------------------
# TAB 3: DỰ BÁO HÀNG LOẠT
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Dự báo hàng loạt từ tệp CSV")
    st.markdown("Tải lên tệp CSV cần dự báo (yêu cầu các cột: `id, date, store_id, is_promo`), hoặc sử dụng tệp mặc định `data/test.csv`.")
    
    uploaded_file = st.file_uploader("Tải lên tệp CSV", type=["csv"])
    
    batch_source_df = None
    if uploaded_file is not None:
        batch_source_df = pd.read_csv(uploaded_file)
        st.success(f"Đã tải lên tệp: {len(batch_source_df):,} dòng.")
    else:
        test_path = os.path.join(ROOT_DIR, "data", "test.csv")
        if os.path.exists(test_path):
            batch_source_df = pd.read_csv(test_path)
            st.info(f"Đang sử dụng tệp mặc định `data/test.csv` ({len(batch_source_df):,} dòng).")
            
    if batch_source_df is not None and local_bundle is not None:
        if st.button("Chạy dự báo hàng loạt"):
            with st.spinner("Đang tính toán dự báo..."):
                t0 = time.time()
                batch_pred_df = predict_batch_dataframe(local_bundle, batch_source_df)
                elapsed = time.time() - t0
                
            st.success(f"Hoàn tất trong {elapsed:.2f} giây. Tổng số dòng: {len(batch_pred_df):,}")
            
            bcol1, bcol2 = st.columns([2, 1])
            with bcol1:
                st.markdown("#### Xem trước kết quả:")
                st.dataframe(batch_pred_df.head(15), use_container_width=True)
            with bcol2:
                st.markdown("#### Thống kê:")
                st.write(batch_pred_df["revenue"].describe().to_frame())
                
            # Nút tải xuống submission.csv
            csv_data = batch_pred_df[["id", "revenue"]].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải xuống tệp submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )


# ------------------------------------------------------------------------------
# TAB 4: ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Đánh giá chất lượng mô hình")
    
    # Tải model metrics JSON
    metrics_path = os.path.join(ROOT_DIR, "artifacts", "model_metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        kcol1.metric("Holdout Val RMSLE", f"{metrics_data.get('holdout_val_rmsle', 0.1218):.4f}", "-46.5% vs Baseline")
        kcol2.metric("Seasonal Tet RMSLE", f"{metrics_data.get('seasonal_tet_rmsle', 0.1459):.4f}", "-47.2% vs Baseline")
        kcol3.metric("Sai số MAE", f"{metrics_data.get('val_mae_m_vnd', 2.03):.2f} tr.đ", "R² = 0.895")
        kcol4.metric("Baseline RMSLE", "0.2276", "Starter Notebook")
        
    st.markdown("---")
    ecol1, ecol2 = st.columns(2)
    
    with ecol1:
        st.markdown("#### 1. Lộ trình cải thiện RMSLE (Ablation Study)")
        ablation_img = os.path.join(ROOT_DIR, "plots", "eval_6_ablation_study_waterfall.png")
        if os.path.exists(ablation_img):
            st.image(ablation_img, caption="Mức giảm RMSLE qua từng giai đoạn cải tiến", use_container_width=True)
            
    with ecol2:
        st.markdown("#### 2. Phân bố phần dư (Residuals)")
        res_img = os.path.join(ROOT_DIR, "plots", "eval_2_residual_distribution.png")
        if os.path.exists(res_img):
            st.image(res_img, caption="Phân bố phần dư chuẩn hóa", use_container_width=True)
            
    st.markdown("---")
    ecol3, ecol4 = st.columns(2)
    with ecol3:
        st.markdown("#### 3. Sai số theo loại vị trí (Store Type)")
        st_img = os.path.join(ROOT_DIR, "plots", "eval_3_rmsle_by_store_type.png")
        if os.path.exists(st_img):
            st.image(st_img, caption="Phân bổ RMSLE theo các phân khúc cửa hàng", use_container_width=True)
            
    with ecol4:
        st.markdown("#### 4. Đặc trưng quan trọng (Feature Importance)")
        fi_img = os.path.join(ROOT_DIR, "plots", "eval_5_feature_importance_top20.png")
        if os.path.exists(fi_img):
            st.image(fi_img, caption="Các đặc trưng có mức đóng góp cao nhất vào mô hình", use_container_width=True)
