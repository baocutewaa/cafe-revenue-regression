"""
Unit tests for Step 8: DEPLOY, PACKAGING & FASTAPI
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

# Chèn đường dẫn gốc và thư mục src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

import src.step3_preprocess as step3_preprocess
import src.step5_scale as step5_scale
import src.step6_model_train as step6_model_train
sys.modules["step3_preprocess"] = step3_preprocess
sys.modules["step5_scale"] = step5_scale
sys.modules["step6_model_train"] = step6_model_train

from src.step8_deploy import (
    predict_single_instance,
    predict_batch_dataframe,
    DEFAULT_ENSEMBLE_WEIGHTS
)
from fastapi.testclient import TestClient
from app.api_server import app


class TestStep8Deploy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle_path = "artifacts/cafe_revenue_bundle.joblib"
        cls.has_real_bundle = os.path.exists(cls.bundle_path)
        if cls.has_real_bundle:
            cls.bundle = joblib.load(cls.bundle_path)

    def test_submission_file_format(self):
        """Kiểm tra tệp submission.csv đạt 100% chuẩn định dạng Kaggle."""
        sub_path = "submission.csv"
        self.assertTrue(os.path.exists(sub_path), "Tệp submission.csv chưa tồn tại!")
        df_sub = pd.read_csv(sub_path)
        
        # 1. Số dòng đúng 3,869
        self.assertEqual(len(df_sub), 3869)
        # 2. Cột đúng id, revenue
        self.assertEqual(list(df_sub.columns), ["id", "revenue"])
        # 3. Không có giá trị null
        self.assertEqual(df_sub.isnull().sum().sum(), 0)
        # 4. Tất cả doanh thu > 0
        self.assertTrue(np.all(df_sub["revenue"] > 0))
        # 5. Phân phối hợp lý
        self.assertGreater(float(df_sub["revenue"].min()), 1.0)
        self.assertLess(float(df_sub["revenue"].max()), 100.0)

    def test_artifact_bundle_keys(self):
        """Kiểm tra tính toàn vẹn của gói cafe_revenue_bundle.joblib."""
        if not self.has_real_bundle:
            self.skipTest("Bundle chưa được sinh ra trên đĩa.")
            
        required_keys = ["models", "weights", "features", "stores_lookup", "weather_lookup", "holidays_lookup", "store_stats_encoder", "metrics"]
        for k in required_keys:
            self.assertIn(k, self.bundle)
            
        self.assertEqual(len(self.bundle["models"]), 4)
        self.assertIn("LightGBM", self.bundle["models"])
        self.assertIn("CatBoost", self.bundle["models"])
        self.assertIn("XGBoost", self.bundle["models"])
        self.assertIn("Ridge (L2 Linear)", self.bundle["models"])

    def test_predict_single_instance(self):
        """Kiểm tra tính năng dự báo đơn lẻ và kịch bản What-If."""
        if not self.has_real_bundle:
            self.skipTest("Bundle chưa được sinh ra trên đĩa.")
            
        res = predict_single_instance(
            self.bundle,
            store_id="S01",
            date_str="2026-02-17",
            is_promo=0,
            temp_c=28.0,
            rain_mm=0.0
        )
        self.assertEqual(res["store_id"], "S01")
        self.assertGreater(res["predicted_revenue_m_vnd"], 0.0)
        self.assertIn("component_predictions", res)
        self.assertIn("operational_recommendation", res)
        self.assertIn("promo_uplift_m_vnd", res)

    def test_predict_batch_dataframe(self):
        """Kiểm tra dự báo hàng loạt theo lô DataFrame."""
        if not self.has_real_bundle:
            self.skipTest("Bundle chưa được sinh ra trên đĩa.")
            
        sample_batch = pd.DataFrame([
            {"id": 1, "date": "2026-01-01", "store_id": "S01", "is_promo": 0},
            {"id": 2, "date": "2026-01-02", "store_id": "S03", "is_promo": 1},
            {"id": 3, "date": "2026-01-03", "store_id": "S05", "is_promo": 0}
        ])
        res_batch = predict_batch_dataframe(self.bundle, sample_batch)
        self.assertEqual(len(res_batch), 3)
        self.assertIn("revenue", res_batch.columns)
        self.assertTrue(np.all(res_batch["revenue"] > 0))

    def test_fastapi_endpoints(self):
        """Kiểm tra tích hợp toàn bộ các REST endpoints của FastAPI."""
        with TestClient(app) as client:
            # 1. GET /health
            r_health = client.get("/health")
            self.assertEqual(r_health.status_code, 200)
            self.assertEqual(r_health.json()["status"], "healthy")
            
            # 2. GET /stores
            r_stores = client.get("/stores?city=Ha%20Noi")
            self.assertEqual(r_stores.status_code, 200)
            self.assertGreater(r_stores.json()["total"], 0)
            
            # 3. POST /predict/single
            payload = {
                "store_id": "S01",
                "date": "2026-02-17",
                "is_promo": 1,
                "temp_c": 28.0,
                "rain_mm": 0.0
            }
            r_single = client.post("/predict/single", json=payload)
            self.assertEqual(r_single.status_code, 200)
            self.assertGreater(r_single.json()["predicted_revenue_m_vnd"], 0.0)
            
            # 4. POST /predict/what-if
            payload_whatif = {
                "store_id": "S29",
                "date": "2026-02-10",
                "base_is_promo": 0,
                "new_is_promo": 1,
                "base_rain_mm": 0.0,
                "new_rain_mm": 35.0
            }
            r_whatif = client.post("/predict/what-if", json=payload_whatif)
            self.assertEqual(r_whatif.status_code, 200)
            self.assertIn("summary", r_whatif.json())
            
            # 5. POST /predict/batch
            payload_batch = [
                {"id": 100000, "date": "2026-01-01", "store_id": "S01", "is_promo": 0},
                {"id": 100001, "date": "2026-01-01", "store_id": "S03", "is_promo": 1}
            ]
            r_batch = client.post("/predict/batch", json=payload_batch)
            self.assertEqual(r_batch.status_code, 200)
            self.assertEqual(len(r_batch.json()), 2)


if __name__ == "__main__":
    unittest.main()
