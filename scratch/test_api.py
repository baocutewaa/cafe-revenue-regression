import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from app.api_server import app

with TestClient(app) as client:
    # 1. Test /health
    r_health = client.get("/health")
    print("Health response:", r_health.status_code, r_health.json())
    assert r_health.status_code == 200
    
    # 2. Test /stores
    r_stores = client.get("/stores?city=Ha%20Noi")
    print("Stores response:", r_stores.status_code, f"Total stores in HN: {r_stores.json()['total']}")
    assert r_stores.status_code == 200
    assert r_stores.json()["total"] > 0
    
    # 3. Test /predict/single
    payload_single = {
        "store_id": "S01",
        "date": "2026-02-17",
        "is_promo": 1,
        "temp_c": 28.0,
        "rain_mm": 0.0
    }
    r_single = client.post("/predict/single", json=payload_single)
    print("Single predict response:", r_single.status_code, r_single.json()["predicted_revenue_m_vnd"])
    assert r_single.status_code == 200
    assert r_single.json()["predicted_revenue_m_vnd"] > 0
    
    # 4. Test /predict/what-if
    payload_whatif = {
        "store_id": "S29",
        "date": "2026-02-10",
        "base_is_promo": 0,
        "new_is_promo": 1,
        "base_rain_mm": 0.0,
        "new_rain_mm": 35.0
    }
    r_whatif = client.post("/predict/what-if", json=payload_whatif)
    print("What-If response:", r_whatif.status_code, r_whatif.json()["summary"])
    assert r_whatif.status_code == 200
    
    # 5. Test /predict/batch
    payload_batch = [
        {"id": 100000, "date": "2026-01-01", "store_id": "S01", "is_promo": 0},
        {"id": 100001, "date": "2026-01-01", "store_id": "S03", "is_promo": 1}
    ]
    r_batch = client.post("/predict/batch", json=payload_batch)
    print("Batch predict response:", r_batch.status_code, r_batch.json())
    assert r_batch.status_code == 200
    assert len(r_batch.json()) == 2
    
print("\nALL FASTAPI ENDPOINTS PASSED 100%!")
