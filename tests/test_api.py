"""
============================================================
Integration Tests — FastAPI Backend Endpoints
============================================================
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "cpu_percent" in response.json()

def test_patient_crud():
    # Create patient
    patient_data = {
        "first_name": "Test",
        "last_name": "Patient",
        "gender": "Male"
    }
    response = client.post("/patient", json=patient_data)
    assert response.status_code == 201
    created = response.json()
    assert "id" in created
    
    # Get patient
    pid = created["id"]
    get_res = client.get(f"/patient/{pid}")
    assert get_res.status_code == 200
    assert get_res.json()["first_name"] == "Test"

def test_history_empty():
    import uuid
    random_id = uuid.uuid4()
    response = client.get(f"/history/{random_id}")
    assert response.status_code == 200
    assert response.json()["total"] == 0
