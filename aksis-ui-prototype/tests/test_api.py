from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import CapabilityResponse, ExperimentCreateRequest, ModelConfig

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

    # Verify parameter_schema contract
    assert "parameter_schema" in data
    assert data["parameter_schema"] is not None
    assert "classification" in data["parameter_schema"]
    assert "xgb_c" in data["parameter_schema"]["classification"]
    xgb_schema = data["parameter_schema"]["classification"]["xgb_c"]
    assert "n_estimators" in xgb_schema
    assert xgb_schema["n_estimators"]["type"] == "int"
    assert "learning_rate" in xgb_schema
    assert xgb_schema["learning_rate"]["type"] == "float"

def test_parameter_schema_optional_contract():
    # Verify CapabilityResponse can be initialized without parameter_schema
    cap = CapabilityResponse(
        learning_types=["supervised"],
        tasks={"supervised": ["classification"]},
        modes=["train"],
        algorithms={"classification": ["xgb_c"]},
        model_presets=["baseline", "custom"],
        preprocessing_strategies={},
        validation_options=["holdout"],
        tuning_options={},
        scoring_options={},
        evaluation_capabilities=[],
        visualization_capabilities=[]
    )
    assert cap.parameter_schema is None

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

def test_create_experiment_custom_preset_with_overrides():
    req = {
        "name": "Test_Custom_XGB",
        "dataset_id": "ds_class_01",
        "learning_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "algorithm": "xgb_c",
            "preset": "custom",
            "overrides": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 8,
                "booster": "gbtree"
            }
        }
    }
    response = client.post("/api/v1/experiments", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "configured"
    assert data["name"] == "Test_Custom_XGB"

def test_create_experiment_non_custom_preset_empty_overrides():
    req = {
        "name": "Test_Fast_Preset",
        "dataset_id": "ds_class_01",
        "learning_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "algorithm": "xgb_c",
            "preset": "fast",
            "overrides": {}
        }
    }
    response = client.post("/api/v1/experiments", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "configured"

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
