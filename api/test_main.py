import pytest
from fastapi.testclient import TestClient
from main import app, load_model
import logging
import time
import threading
from unittest.mock import Mock

# Create test client
client = TestClient(app)

# Test Home Route
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the House Price Prediction API!"}

# Test Model Info Endpoint
def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    if response.json().get("status") == "Loaded":
        assert response.json() == {
            "model": "House Price Prediction Model",
            "version": "1.0",
            "status": "Loaded",
        }
    else:
        assert "error" in response.json()

# Test Health Check Endpoint
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# Test Prediction Endpoint (Valid Input)
def test_predict_valid():
    sample_input = {
        "MedInc": 3.5,
        "AveRooms": 5.0,
        "AveOccup": 2.0,
        "Latitude": 34.0,
        "Longitude": -118.2
    }
    
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert isinstance(response.json()["predicted_price"], float)

# Test Prediction Endpoint (Invalid Input)
@pytest.mark.parametrize("invalid_input", [
    {"AveRooms": 5.0, "AveOccup": 2.0, "Latitude": 34.0, "Longitude": -118.2},  # Missing MedInc
    {"MedInc": "invalid", "AveRooms": 5.0, "AveOccup": 2.0, "Latitude": 34.0, "Longitude": -118.2},  # String instead of float
    {"MedInc": 3.5, "AveRooms": 5.0, "AveOccup": 2.0, "Latitude": 34.0},  # Missing Longitude
])
def test_predict_invalid(invalid_input):
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

# Test Prediction when Model is Not Loaded
def test_predict_model_not_loaded(monkeypatch):
    mock_load = Mock(side_effect=Exception("Model not loaded"))
    monkeypatch.setattr("main.load_model", mock_load)
    
    sample_input = {
        "MedInc": 3.5,
        "AveRooms": 5.0,
        "AveOccup": 2.0,
        "Latitude": 34.0,
        "Longitude": -118.2
    }
    
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "error" in response.json()

# Test Model Loading Failure
def test_model_loading_failure(monkeypatch):
    mock_load = Mock(side_effect=Exception("Failed to load model"))
    monkeypatch.setattr("main.load_model", mock_load)
    
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "error" in response.json()

# Test Edge Cases for Prediction
@pytest.mark.parametrize("edge_case_input", [
    {"MedInc": 0.0, "AveRooms": 1.0, "AveOccup": 1.0, "Latitude": -90.0, "Longitude": -180.0},
    {"MedInc": 1000000.0, "AveRooms": 100.0, "AveOccup": 100.0, "Latitude": 90.0, "Longitude": 180.0},
])
def test_predict_edge_cases(edge_case_input):
    response = client.post("/predict", json=edge_case_input)
    assert response.status_code == 200
    assert "predicted_price" in response.json()

# Test Logging
def test_logging(caplog):
    sample_input = {
        "MedInc": 3.5,
        "AveRooms": 5.0,
        "AveOccup": 2.0,
        "Latitude": 34.0,
        "Longitude": -118.2
    }
    with caplog.at_level(logging.INFO):
        response = client.post("/predict", json=sample_input)
        assert "Incoming request" in caplog.text
        assert "Response 200" in caplog.text

# Test Concurrency
def test_concurrent_requests():
    def make_request():
        sample_input = {
            "MedInc": 3.5,
            "AveRooms": 5.0,
            "AveOccup": 2.0,
            "Latitude": 34.0,
            "Longitude": -118.2
        }
        response = client.post("/predict", json=sample_input)
        assert response.status_code == 200

    threads = [threading.Thread(target=make_request) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

# Test Response Time
def test_response_time():
    sample_input = {
        "MedInc": 3.5,
        "AveRooms": 5.0,
        "AveOccup": 2.0,
        "Latitude": 34.0,
        "Longitude": -118.2
    }
    start_time = time.time()
    response = client.post("/predict", json=sample_input)
    end_time = time.time()
    assert response.status_code == 200
    assert end_time - start_time < 1.0