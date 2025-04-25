"""
California Housing Price Prediction API Test Suite

Comprehensive test coverage for the FastAPI housing price prediction service.
Validates functionality, edge cases, performance, and error handling.

Test Categories:
- Core Functionality: Model loading, health checks, prediction
- Input Validation: Type checks, value ranges, missing fields
- Edge Cases: Boundary values, extreme inputs, null handling
- Performance: Concurrency, response times
- Error Handling: Invalid endpoints, model failures

Testing Strategy:
- TestClient for FastAPI integration testing
- Parameterized tests for input variations
- Concurrency testing with ThreadPoolExecutor
- Mocking application states for failure scenarios

Author: Glita Jay
Version: 1.1
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app  # Import your FastAPI app
import time
import concurrent.futures
import re

# ============================= Test Model Loading ================================

def test_model_loading():
    """Validate successful model initialization during application startup"""
    with TestClient(app) as client:
        response = client.get("/model/status")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["status"] == "loaded"
        assert "model_type" in status_data
        assert status_data["model_source"] in ["Hugging Face Hub", "Local"]

# ============================== Health Check Test =================================

def test_health_check():
    """Verify system health check endpoint provides operational insights"""
    with TestClient(app) as client:
        # Simulating when the model is loaded
        app.state.model_loaded = True
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "components" in health_data
        assert "model" in health_data["components"]
        assert "memory" in health_data["components"]["model"]
        assert isinstance(health_data["components"]["model"]["memory"], str)

        # Simulating when the model is not loaded
        app.state.model_loaded = False
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "degraded"
        assert "components" in health_data
        assert "model" in health_data["components"]
        assert health_data["components"]["model"]["status"] == "unavailable"
        assert health_data["components"]["model"]["memory"] == "n/a"


# ============================== Prediction Test =====================================

def test_predict_house_price():
    """Validate successful price prediction with valid input"""
    with TestClient(app) as client:
        payload = {
            "MedInc": 3.8479,
            "AveRooms": 5.069,
            "AveOccup": 2.742,
            "Latitude": 34.42,
            "Longitude": -118.32
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        response_data = response.json()
        assert "prediction" in response_data
        assert "units" in response_data
        assert response_data["units"] == "hundred-thousands of USD (scaled)"
        assert "input" in response_data
        assert re.match(r"\d+\.\d+\.\d+", response_data["model_version"])

# ============================= Input Validation Test ==============================

def test_invalid_input():
    """Ensure proper validation for unrealistically high MedInc"""
    with TestClient(app) as client:
        invalid_payload = {
            "MedInc": 200.0,
            "AveRooms": 5.0,
            "AveOccup": 2.5,
            "Latitude": 34.05,
            "Longitude": -118.25
        }
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
        error_list = response.json()["detail"]
        error_msg = error_list[0]["msg"]  # Grab the first error
        assert "Value error, MedInc exceeds realistic threshold for California" in error_msg


# ================================ Test Startup Behavior ============================

def test_app_startup():
    """Validate application initialization sequence"""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["status"] == "operational"
        assert "environment" in status_data

# ============================= Model Failure Scenario Test ==============================

def test_predict_fails_when_model_not_loaded():
    """Simulate and test model loading failure scenario"""
    with TestClient(app) as client:
        client.app.state.model_loaded = False
        client.app.state.model = None
        response = client.post("/predict", json={
            "MedInc": 3.8479,
            "AveRooms": 5.069,
            "AveOccup": 2.742,
            "Latitude": 34.42,
            "Longitude": -118.32
        })
        assert response.status_code == 503
        assert response.json()["error"] == "Service Unavailable"
        assert "Model not initialized" in response.json()["detail"]


# ============================= Response Schema Validation ==============================

def test_predict_response_schema():
    """Validate prediction response structure and data types"""
    with TestClient(app) as client:
        payload = {
            "MedInc": 2.5,
            "AveRooms": 4.2,
            "AveOccup": 3.1,
            "Latitude": 36.7,
            "Longitude": -119.5
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert isinstance(data["prediction"], (int, float))
        assert isinstance(data["units"], str)
        assert isinstance(data["input"], dict)
        assert re.match(r"\d+\.\d+\.\d+", data["model_version"])

# ============================= Missing Fields Validation ==============================

def test_missing_input_fields():
    """Test error handling for incomplete requests"""
    with TestClient(app) as client:
        incomplete_payload = {
            "MedInc": 2.5,
            "AveRooms": 4.2
        }
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422
        assert "detail" in response.json()

# ============================= Extreme Input Handling ==============================

@pytest.mark.parametrize("payload, expected_status", [
    ({"MedInc": -5, "AveRooms": 3, "AveOccup": 2, "Latitude": 34, "Longitude": -118}, 422),
    ({"MedInc": 1e6, "AveRooms": 1e5, "AveOccup": 1e4, "Latitude": 90, "Longitude": 180}, 422),
])
def test_extreme_values(payload, expected_status):
    """Parameterized test for edge case inputs"""
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == expected_status

# ============================= Boundary Value Testing ==============================

@pytest.mark.parametrize("income", [14.99, 15.00])
def test_income_boundary(income):
    """Test validation thresholds around business logic boundaries"""
    with TestClient(app) as client:
        payload = {
            "MedInc": income,
            "AveRooms": 4,
            "AveOccup": 2,
            "Latitude": 35,
            "Longitude": -120
        }
        response = client.post("/predict", json=payload)
        if income > 15:
            assert response.status_code == 422
        else:
            assert response.status_code == 200

# ============================= Concurrency Testing ==============================

def test_concurrent_requests():
    """Validate system behavior under concurrent load"""
    payload = {
        "MedInc": 3.8479,
        "AveRooms": 5.069,
        "AveOccup": 2.742,
        "Latitude": 34.42,
        "Longitude": -118.32
    }
    with TestClient(app) as client:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(client.post, "/predict", json=payload) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                assert response.status_code == 200
                assert "prediction" in response.json()

# ============================= Performance Testing ==============================

def test_predict_response_time():
    """Validate prediction endpoint meets performance requirements"""
    payload = {
        "MedInc": 3.8479,
        "AveRooms": 5.069,
        "AveOccup": 2.742,
        "Latitude": 34.42,
        "Longitude": -118.32
    }
    with TestClient(app) as client:
        start_time = time.time()
        response = client.post("/predict", json=payload)
        end_time = time.time()
        assert response.status_code == 200
        assert (end_time - start_time) < 2.0
