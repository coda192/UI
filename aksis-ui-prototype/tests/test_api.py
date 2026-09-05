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
    
    # Verify algorithm_metadata contract
    assert "algorithm_metadata" in data
    assert data["algorithm_metadata"] is not None
    assert "xgb_c" in data["algorithm_metadata"]
    xgb_meta = data["algorithm_metadata"]["xgb_c"]
    assert "display_name" in xgb_meta and len(xgb_meta["display_name"]) > 0
    assert "description" in xgb_meta and len(xgb_meta["description"]) > 0
    assert "strengths" in xgb_meta and isinstance(xgb_meta["strengths"], list)
    assert "limitations" in xgb_meta and isinstance(xgb_meta["limitations"], list)
    assert "best_for" in xgb_meta and isinstance(xgb_meta["best_for"], list)

def test_datasets():
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Verify optional display_name and description support
    first_ds = data[0]
    assert "display_name" in first_ds
    assert "description" in first_ds
    assert first_ds["display_name"] is not None
    assert first_ds["description"] is not None

def test_get_single_dataset():
    response = client.get("/api/v1/datasets/ds_class_01")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ds_class_01"
    assert data["display_name"] == "Müşteri Kayıp Analizi (Customer Churn)"
    assert "description" in data and len(data["description"]) > 0

def test_create_experiment_default_mode():
    # Omitting 'mode' in payload must default to 'train', not 'local'
    from backend.schemas import ExperimentCreateRequest, ModelConfig
    model_req = ExperimentCreateRequest(
        name="Test_Default_Mode",
        dataset_id="ds_class_01",
        learning_type="supervised",
        task="classification",
        model=ModelConfig(algorithm="xgb_c")
    )
    assert model_req.mode == "train"

    req = {
        "name": "Test_Default_API",
        "dataset_id": "ds_class_01",
        "learning_type": "supervised",
        "task": "classification",
        "model": {
            "algorithm": "xgb_c",
            "preset": "fast"
        }
    }
    response = client.post("/api/v1/experiments", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "configured"
    assert data["name"] == "Test_Default_API"

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
