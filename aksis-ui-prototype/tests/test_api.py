from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_capabilities():
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "learning_types" in data
    assert "supervised" in data["learning_types"]
    assert "modes" in data
    assert "train" in data["modes"]
    assert "tune" in data["modes"]
    assert "predict" in data["modes"]
    assert "algorithms" in data
    assert "classification" in data["algorithms"]
    assert "regression" in data["algorithms"]
    assert "anomaly_detection" in data["algorithms"]

def test_datasets():
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_experiment_train_mode():
    req = {
        "name": "Test_Train",
        "dataset_id": "ds_class_01",
        "learning_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "algorithm": "xgb_c",
            "preset": "fast"
        }
    }
    response = client.post("/api/v1/experiments", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "configured"
    assert data["name"] == "Test_Train"

def test_create_experiment_tune_mode():
    req = {
        "name": "Test_Tune",
        "dataset_id": "ds_class_01",
        "learning_type": "supervised",
        "task": "classification",
        "mode": "tune",
        "model": {
            "algorithm": "catboost",
            "preset": "strong"
        },
        "tuning": {
            "enabled": True,
            "sampler": "tpe",
            "pruner": "median",
            "trials": 15,
            "scoring": "f1_macro"
        }
    }
    response = client.post("/api/v1/experiments", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "configured"
    assert data["name"] == "Test_Tune"

def test_artifacts_empty():
    response = client.get("/api/v1/artifacts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_inference():
    req = {"artifact_id": "mock_art", "dataset_id": "ds_class_01"}
    response = client.post("/api/v1/inference", json=req)
    assert response.status_code == 200
    data = response.json()
    assert "predictions_preview" in data
