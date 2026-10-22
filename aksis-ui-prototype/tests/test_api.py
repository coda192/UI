from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import CapabilityResponse, ExperimentCreateRequest

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


def test_capabilities_real_aksis_provider(monkeypatch):
    import backend.services.aksis_service as aksis_mod
    monkeypatch.setenv("AKSIS_PROVIDER", "aksis")
    mock_caps = {
        "learning_types": ["supervised", "unsupervised"],
        "tasks": ["classification", "regression", "anomaly_detection"],
        "modes": ["train", "tune", "predict"],
        "config_applicability": {
            "supervised_model": {
                "algorithm_types": ["supervised"],
                "supported_tasks": ["classification", "regression"]
            }
        },
        "algorithms": {"classification": ["logreg"]},
        "model_presets": {
            "supervised": ["baseline", "fast"],
            "unsupervised": ["fast", "custom"]
        },
        "preprocessing": {"encoding": ["onehot"]},
        "tuning": {
            "samplers": {"supervised": ["tpe", "random"], "unsupervised": ["random"]},
            "pruners": {"supervised": ["none", "median"], "unsupervised": ["none"]}
        },
        "algorithm_metadata": {
            "logreg": {
                "display_name": "Logistic Regression",
                "description": "Linear classifier",
                "strengths": ["Fast"],
                "limitations": ["Linear boundary only"],
                "best_for": ["Baseline"]
            }
        }
    }
    monkeypatch.setattr(aksis_mod, "aksis_get_capabilities", lambda: mock_caps)
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm_metadata" in data
    assert "logreg" in data["algorithm_metadata"]
    assert data["algorithm_metadata"]["logreg"]["display_name"] == "Logistic Regression"
    assert data["algorithm_metadata"]["logreg"]["strengths"] == ["Fast"]

    # Verify new schema fields
    assert data["tasks"] == ["classification", "regression", "anomaly_detection"]
    assert data["model_presets"]["supervised"] == ["baseline", "fast"]
    assert data["preprocessing"]["encoding"] == ["onehot"]
    assert data["tuning"]["samplers"]["supervised"] == ["tpe", "random"]
    assert data["tuning"]["pruners"]["supervised"] == ["none", "median"]


def test_datasets_real_aksis_provider(monkeypatch):
    import backend.services.aksis_service as aksis_mod
    from types import SimpleNamespace

    monkeypatch.setenv("AKSIS_PROVIDER", "aksis")

    mock_spec1 = SimpleNamespace(
        id="credit_card_fraud",
        display_name="Kredi Kartı Dolandırıcılığı",
        description="Gerçek zamanlı fraud tespiti",
        source="oracle_db",
        local_data=True,
        target="Class",
        columns_to_use=["Time", "Amount", "V1"],
        compatible_tasks=["anomaly_detection"],
        # Sensitive fields that must NOT be exposed:
        host="10.0.0.1",
        port=1521,
        user="oracle_user",
        password="secret_password",
        catalog="PROD",
        schema="FINANCE",
        http_schema="https",
        data_query="SELECT * FROM secret_table"
    )

    test_datasets = [mock_spec1]
    mock_dict = {"credit_card_fraud": mock_spec1}
    monkeypatch.setattr(
        aksis_mod,
        "_get_aksis_dataset_accessors",
        lambda: (lambda: test_datasets, lambda ds_id: mock_dict.get(ds_id))
    )

    # Test GET /api/v1/datasets
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    ds = data[0]
    assert ds["id"] == "credit_card_fraud"
    assert ds["display_name"] == "Kredi Kartı Dolandırıcılığı"
    assert ds["target"] == "Class"
    assert ds["source"] == "oracle_db"
    assert ds["columns_to_use"] == ["Time", "Amount", "V1"]
    assert ds["compatible_tasks"] == ["anomaly_detection"]

    # Verify no sensitive credentials leaked
    for sensitive in ["host", "port", "user", "password", "catalog", "schema", "http_schema", "data_query"]:
        assert sensitive not in ds

    # Test GET /api/v1/datasets/{dataset_id}
    detail_resp = client.get("/api/v1/datasets/credit_card_fraud")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == "credit_card_fraud"
    assert detail["display_name"] == "Kredi Kartı Dolandırıcılığı"
    assert detail["description"] == "Gerçek zamanlı fraud tespiti"
    assert detail["target"] == "Class"
    assert detail["source"] == "oracle_db"
    assert detail["compatible_tasks"] == ["anomaly_detection"]
    for sensitive in ["host", "port", "user", "password", "catalog", "schema", "http_schema", "data_query"]:
        assert sensitive not in detail

    # Test 404 for unknown dataset
    not_found = client.get("/api/v1/datasets/non_existent_dataset")
    assert not_found.status_code == 404


def test_parameter_schema_optional_contract():
    # Verify CapabilityResponse can be initialized without parameter_schema
    cap = CapabilityResponse(
        learning_types=["supervised"],
        tasks=["classification"],
        modes=["train"],
        algorithms={"classification": ["xgb_c"]},
        model_presets={"supervised": ["baseline", "custom"]},
        preprocessing={},
        data_split={},
        tuning={"samplers": {}, "pruners": {}}
    )
    assert cap.parameter_schema is None

def test_datasets():
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Verify DataSpec fields support
    first_ds = data[0]
    assert "id" in first_ds
    assert "display_name" in first_ds
    assert "description" in first_ds
    assert "source" in first_ds
    assert "local_data" in first_ds
    assert "columns_to_use" in first_ds
    assert first_ds["display_name"] is not None
    assert first_ds["description"] is not None

    # Verify no credentials or internal DB connection details leaked
    for ds in data:
        assert "password" not in ds
        assert "user" not in ds
        assert "host" not in ds
        assert "port" not in ds
        assert "data_query" not in ds
        assert "catalog" not in ds
        assert "http_schema" not in ds

def test_get_single_dataset():
    response = client.get("/api/v1/datasets/ds_class_01")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ds_class_01"
    assert data["display_name"] == "Customer Churn Analysis"
    assert data["source"] == "csv_file"
    assert data["local_data"] is True
    assert "Churn" in data["columns_to_use"]
    assert "description" in data and len(data["description"]) > 0

    # Ensure sensitive credentials are not exposed
    assert "password" not in data
    assert "user" not in data
    assert "host" not in data
    assert "data_query" not in data

def test_create_experiment_default_mode():
    model_req = ExperimentCreateRequest(
        project_name="Test_Default_Mode",
        dataset_id="ds_class_01",
        algorithm_type="supervised",
        task="classification",
        model={"name": "xgb_c"}
    )
    assert model_req.mode == "train"

    req = {
        "project_name": "Test_Default_API",
        "dataset_id": "ds_class_01",
        "algorithm_type": "supervised",
        "task": "classification",
        "model": {
            "name": "xgb_c",
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
        "project_name": "Test_Train",
        "dataset_id": "ds_class_01",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "name": "xgb_c",
            "preset": "fast"
        }
    }
    response = client.post("/api/v1/experiments", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "configured"
    assert data["name"] == "Test_Train"

def test_create_experiment_custom_preset_with_parameters():
    req = {
        "project_name": "Test_Custom_XGB",
        "dataset_id": "ds_class_01",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "name": "xgb_c",
            "preset": "custom",
            "parameters": {
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

def test_create_experiment_full_nested_canonical_request():
    req = {
        "project_name": "Test_Full_Nested",
        "dataset_id": "ds_class_01",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "tune",
        "model": {
            "name": "catboost",
            "preset": "strong",
            "parameters": {"depth": 6}
        },
        "preprocess": {
            "handle_missing": True,
            "numeric_imputer": "mean",
            "encoding": True,
            "encode_method": "onehot",
            "scaling": True,
            "scale_method": "standard"
        },
        "data_split": {
            "method": "holdout",
            "parameters": {"test_size": 0.25}
        },
        "tuning": {
            "enabled": True,
            "n_trials": 15,
            "sampler": "tpe",
            "pruner": "median",
            "metric": "f1_macro"
        }
    }
    response = client.post("/api/v1/experiments", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "configured"
    assert data["name"] == "Test_Full_Nested"

def test_transport_a_pydantic_round_trip():
    input_dict = {
        "project_name": "payload_transport_test",
        "dataset_id": "breast_cancer",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "algorithm": "logreg"
        }
    }
    req = ExperimentCreateRequest(**input_dict)
    payload = req.model_dump(exclude_none=True)
    assert payload == input_dict

def test_transport_b_nested_dict_preservation():
    input_dict = {
        "project_name": "nested_preservation_test",
        "dataset_id": "ds_01",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "algorithm": "logreg",
            "preset": "fast",
            "overrides": {
                "C": 1.0
            }
        },
        "preprocess": {
            "some_future_field": "test",
            "nested": {
                "value": 123
            }
        }
    }
    req = ExperimentCreateRequest(**input_dict)
    payload = req.model_dump(exclude_none=True)
    assert payload == input_dict
    assert payload["model"]["overrides"]["C"] == 1.0
    assert payload["preprocess"]["nested"]["value"] == 123

def test_transport_c_optional_section_omission():
    input_dict = {
        "project_name": "omission_test",
        "dataset_id": "ds_01",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {"name": "test_model"}
    }
    req = ExperimentCreateRequest(**input_dict)
    payload = req.model_dump(exclude_none=True)
    assert "preprocess" not in payload
    assert "data_split" not in payload
    assert "tuning" not in payload
    assert "data_analyze" not in payload
    assert "evaluation" not in payload
    assert "visualization" not in payload
    assert payload == input_dict

def test_transport_d_fastapi_post_transport():
    minimal_payload = {
        "project_name": "payload_transport_test",
        "dataset_id": "breast_cancer",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "algorithm": "logreg"
        }
    }
    response = client.post("/api/v1/experiments", json=minimal_payload)
    assert response.status_code == 200
    meta = response.json()
    assert meta["name"] == minimal_payload["project_name"]
    assert meta["status"] == "configured"
    assert "id" in meta

def test_transport_e_service_storage_preservation():
    from backend.api.deps import get_service
    service = get_service()

    payload_sent = {
        "project_name": "payload_transport_test",
        "dataset_id": "breast_cancer",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "algorithm": "logreg"
        }
    }
    response = client.post("/api/v1/experiments", json=payload_sent)
    assert response.status_code == 200
    meta = response.json()
    exp_id = meta["id"]

    # Verify directly from service internal memory
    if hasattr(service, "_experiments"):  # MockAksisService
        stored_req = service._experiments[exp_id]["request"]
    else:  # RealAksisService
        stored_req = service._in_memory_status[exp_id]["request"]

    recovered_payload = stored_req.model_dump(exclude_none=True)
    assert recovered_payload == payload_sent

def test_transport_f_mock_experiment_flow():
    from backend.api.deps import get_service
    service = get_service()

    payload = {
        "project_name": "mock_flow_test",
        "dataset_id": "ds_class_01",
        "algorithm_type": "supervised",
        "task": "classification",
        "mode": "train",
        "model": {
            "name": "CatBoost"
        }
    }
    req = ExperimentCreateRequest(**payload)
    meta = service.create_experiment(req)
    assert meta.name == "mock_flow_test"
    assert meta.status == "configured"

    # List
    experiments = service.list_experiments()
    assert any(e.id == meta.id for e in experiments)

    # Get
    fetched = service.get_experiment(meta.id)
    assert fetched.id == meta.id
    assert fetched.name == "mock_flow_test"
    assert fetched.status == "configured"


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


def test_dataset_profile_schema_and_endpoint(monkeypatch):
    from backend.schemas import ColumnProfile, DatasetProfileResponse

    # 1. Schema direct validation
    col = ColumnProfile(
        name="age",
        detected_type="numeric",
        dtype="float64",
        missing_count=5,
        missing_percentage=2.5
    )
    assert col.name == "age"
    assert col.detected_type == "numeric"
    assert col.dtype == "float64"
    assert col.missing_count == 5
    assert col.missing_percentage == 2.5

    profile = DatasetProfileResponse(
        dataset_id="test_ds",
        row_count=1000,
        column_count=10,
        memory_usage_mb=1.25,
        columns_with_missing=1,
        total_missing_values=5,
        columns=[col]
    )
    assert profile.dataset_id == "test_ds"
    assert profile.row_count == 1000
    assert profile.column_count == 10
    assert profile.memory_usage_mb == 1.25
    assert profile.columns_with_missing == 1
    assert profile.total_missing_values == 5
    assert len(profile.columns) == 1

    # 2. Endpoint default 404 (when analysis not yet run)
    resp_404 = client.get("/api/v1/datasets/test_ds/profile")
    assert resp_404.status_code == 404
    assert "Henüz analiz çalıştırılmadı" in resp_404.json()["detail"]

    # 3. Endpoint success when service provides profile
    from backend.api.deps import get_service
    from backend.main import app
    class MockServiceWithProfile:
        def get_dataset_profile(self, dataset_id: str):
            return profile

    app.dependency_overrides[get_service] = lambda: MockServiceWithProfile()
    try:
        resp_success = client.get("/api/v1/datasets/test_ds/profile")
        assert resp_success.status_code == 200
        res_data = resp_success.json()
        assert res_data["dataset_id"] == "test_ds"
        assert res_data["row_count"] == 1000
        assert res_data["memory_usage_mb"] == 1.25
        assert len(res_data["columns"]) == 1
        assert res_data["columns"][0]["name"] == "age"
    finally:
        app.dependency_overrides.clear()


def test_dataset_info_schema_validation():
    from backend.schemas import DatasetInfoResponse, ColumnInfo, CorrelationInfo

    # Validate schema handles None cells in correlation matrix
    data = {
        "dataset_id": "test_ds",
        "row_count": 100,
        "column_count": 2,
        "missing_value_count": 5,
        "missing_column_count": 1,
        "type_counts": {"numerical": 1, "categorical": 1},
        "columns": [
            {
                "name": "col1",
                "dtype": "int64",
                "primitive_type": "numerical",
                "subtype": "integer",
                "unique_count": 50,
                "missing_count": 0,
                "missing_rate": 0.0,
                "flags": ["numerical"],
            },
            {
                "name": "col2",
                "dtype": "object",
                "primitive_type": "categorical",
                "subtype": "nominal",
                "unique_count": 5,
                "missing_count": 5,
                "missing_rate": 0.05,
                "flags": ["has_missing"],
            },
        ],
        "correlation": {
            "columns": ["col1", "col2"],
            "matrix": [
                [1.0, None],
                [None, 1.0],
            ],
        },
    }
    info = DatasetInfoResponse(**data)
    assert info.dataset_id == "test_ds"
    assert len(info.columns) == 2
    assert info.correlation.matrix[0][1] is None
    assert info.correlation.matrix[1][0] is None
    assert info.correlation.matrix[0][0] == 1.0


def test_get_dataset_info_mock_endpoint():
    response = client.get("/api/v1/datasets/ds_class_01/info")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == "ds_class_01"
    assert data["row_count"] > 0
    assert data["column_count"] > 0
    assert "type_counts" in data
    assert isinstance(data["columns"], list)
    assert len(data["columns"]) == data["column_count"]
    assert "correlation" in data
    assert "matrix" in data["correlation"]
    # Check column info structure
    col0 = data["columns"][0]
    assert "name" in col0
    assert "dtype" in col0
    assert "primitive_type" in col0
    assert "subtype" in col0
    assert "unique_count" in col0
    assert "missing_count" in col0
    assert "missing_rate" in col0
    assert "flags" in col0


def test_dataset_info_histogram_schema_validation():
    from backend.schemas import DatasetInfoResponse, ColumnInfo, HistogramData

    data = {
        "dataset_id": "test_hist_ds",
        "row_count": 100,
        "column_count": 2,
        "missing_value_count": 0,
        "missing_column_count": 0,
        "type_counts": {"numeric": 1, "categorical": 1},
        "columns": [
            {
                "name": "col_num",
                "dtype": "float64",
                "primitive_type": "numeric",
                "subtype": "continuous",
                "unique_count": 50,
                "missing_count": 0,
                "missing_rate": 0.0,
                "statistics": {"skewness": 0.42},
                "histogram": {
                    "counts": [10, 20, 30, 40],
                    "bin_edges": [0.0, 10.0, 20.0, 30.0, 40.0]
                }
            },
            {
                "name": "col_cat",
                "dtype": "object",
                "primitive_type": "categorical",
                "subtype": "nominal",
                "unique_count": 2,
                "missing_count": 0,
                "missing_rate": 0.0,
                "histogram": None
            }
        ]
    }
    info = DatasetInfoResponse(**data)
    assert info.columns[0].histogram is not None
    assert isinstance(info.columns[0].histogram, HistogramData)
    assert info.columns[0].histogram.counts == [10, 20, 30, 40]
    assert len(info.columns[0].histogram.bin_edges) == len(info.columns[0].histogram.counts) + 1
    assert info.columns[1].histogram is None


def test_get_dataset_info_histogram_in_response():
    response = client.get("/api/v1/datasets/ds_class_01/info")
    assert response.status_code == 200
    data = response.json()
    cols = {c["name"]: c for c in data["columns"]}

    assert "Age" in cols
    age_hist = cols["Age"].get("histogram")
    assert age_hist is not None
    assert "counts" in age_hist
    assert "bin_edges" in age_hist
    assert len(age_hist["bin_edges"]) == len(age_hist["counts"]) + 1

    assert "Score" in cols
    assert cols["Score"].get("histogram") is None

    assert "Department" in cols
    assert cols["Department"].get("histogram") is None


def test_get_dataset_info_refresh_param():
    response = client.get("/api/v1/datasets/ds_class_01/info?refresh=true")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == "ds_class_01"


def test_get_dataset_info_not_found():
    response = client.get("/api/v1/datasets/non_existent_dataset/info")
    assert response.status_code == 404


def test_get_dataset_info_real_aksis_provider(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setenv("AKSIS_PROVIDER", "aksis")

    called_args = {}

    def fake_get_dataset_info(dataset_id: str, refresh: bool = False):
        called_args["dataset_id"] = dataset_id
        called_args["refresh"] = refresh
        if dataset_id == "non_existent":
            raise KeyError("not found")
        return {
            "dataset_id": dataset_id,
            "row_count": 500,
            "column_count": 3,
            "missing_value_count": 2,
            "missing_column_count": 1,
            "type_counts": {"numerical": 2, "categorical": 1},
            "columns": [
                {
                    "name": "col_a",
                    "dtype": "float64",
                    "primitive_type": "numerical",
                    "subtype": "continuous",
                    "unique_count": 300,
                    "missing_count": 0,
                    "missing_rate": 0.0,
                    "flags": ["numerical"],
                },
                {
                    "name": "col_b",
                    "dtype": "int64",
                    "primitive_type": "numerical",
                    "subtype": "integer",
                    "unique_count": 10,
                    "missing_count": 2,
                    "missing_rate": 0.004,
                    "flags": ["has_missing"],
                },
                {
                    "name": "col_c",
                    "dtype": "object",
                    "primitive_type": "categorical",
                    "subtype": "nominal",
                    "unique_count": 2,
                    "missing_count": 0,
                    "missing_rate": 0.0,
                    "flags": ["categorical"],
                },
            ],
            "correlation": {
                "columns": ["col_a", "col_b"],
                "matrix": [[1.0, 0.42], [0.42, 1.0]],
            },
        }

    mock_mod = MagicMock()
    mock_mod.get_dataset_info = fake_get_dataset_info
    monkeypatch.setitem(sys.modules, "src", MagicMock())
    monkeypatch.setitem(sys.modules, "src.core", MagicMock())
    monkeypatch.setitem(sys.modules, "src.core.data_processing", MagicMock())
    monkeypatch.setitem(sys.modules, "src.core.data_processing.data_info", mock_mod)

    # Test GET /api/v1/datasets/ds_real_01/info
    resp = client.get("/api/v1/datasets/ds_real_01/info")
    assert resp.status_code == 200
    assert called_args == {"dataset_id": "ds_real_01", "refresh": False}
    data = resp.json()
    assert data["dataset_id"] == "ds_real_01"
    assert data["row_count"] == 500

    # Test refresh=true query parameter
    resp_refresh = client.get("/api/v1/datasets/ds_real_01/info?refresh=true")
    assert resp_refresh.status_code == 200
    assert called_args == {"dataset_id": "ds_real_01", "refresh": True}

    # Test 404 for unknown dataset in real aksis provider
    resp_404 = client.get("/api/v1/datasets/non_existent/info")
    assert resp_404.status_code == 404


def test_dataset_info_statistics_schema():
    from backend.schemas import ColumnInfo

    # 1. Full statistics
    col_full = ColumnInfo(
        name="test_num",
        dtype="float64",
        primitive_type="numeric",
        unique_count=100,
        missing_count=0,
        missing_rate=0.0,
        flags=[],
        statistics={
            "count": 100.0,
            "mean": 35.2,
            "std": 8.4,
            "min": 18.0,
            "25%": 29.0,
            "50%": 34.0,
            "75%": 41.0,
            "max": 67.0,
            "skewness": 0.65,
        },
    )
    assert col_full.statistics is not None
    assert col_full.statistics["count"] == 100.0
    assert col_full.statistics["25%"] == 29.0
    assert col_full.statistics["skewness"] == 0.65

    # 2. Undefined skewness
    col_undefined_skew = ColumnInfo(
        name="test_undef",
        dtype="int64",
        primitive_type="numeric",
        unique_count=10,
        missing_count=0,
        missing_rate=0.0,
        flags=[],
        statistics={
            "count": 50.0,
            "mean": 10.0,
            "std": 2.0,
            "min": 5.0,
            "25%": 8.0,
            "50%": 10.0,
            "75%": 12.0,
            "max": 15.0,
            "skewness": None,
        },
    )
    assert col_undefined_skew.statistics["skewness"] is None

    # 3. Missing statistics
    col_no_stats = ColumnInfo(
        name="test_no_stats",
        dtype="object",
        primitive_type="categorical",
        unique_count=5,
        missing_count=0,
        missing_rate=0.0,
        flags=[],
        statistics=None,
    )
    assert col_no_stats.statistics is None

    # 4. Constant column
    col_const = ColumnInfo(
        name="test_const",
        dtype="float64",
        primitive_type="numeric",
        unique_count=1,
        missing_count=0,
        missing_rate=0.0,
        flags=["constant"],
        statistics={
            "count": 100.0,
            "mean": 42.0,
            "std": 0.0,
            "min": 42.0,
            "25%": 42.0,
            "50%": 42.0,
            "75%": 42.0,
            "max": 42.0,
            "skewness": None,
        },
    )
    assert col_const.statistics["min"] == col_const.statistics["max"] == 42.0
    assert col_const.statistics["std"] == 0.0


def test_mock_endpoint_returns_numeric_statistics():
    response = client.get("/api/v1/datasets/ds_class_01/info")
    assert response.status_code == 200
    data = response.json()
    columns = {c["name"]: c for c in data["columns"]}

    # Age: complete statistics
    assert "Age" in columns
    age_stats = columns["Age"]["statistics"]
    assert age_stats is not None
    assert age_stats["count"] == 988.0
    assert age_stats["mean"] == 38.5
    assert age_stats["std"] == 10.2
    assert age_stats["min"] == 18.0
    assert age_stats["25%"] == 30.0
    assert age_stats["50%"] == 37.0
    assert age_stats["75%"] == 46.0
    assert age_stats["max"] == 65.0
    assert age_stats["skewness"] == 0.35

    # Department: categorical without statistics
    assert "Department" in columns
    assert columns["Department"]["statistics"] is None

    # Tenure: repeated quartile values and undefined skewness
    assert "Tenure" in columns
    tenure_stats = columns["Tenure"]["statistics"]
    assert tenure_stats is not None
    assert tenure_stats["50%"] == tenure_stats["75%"] == 3.0
    assert tenure_stats["skewness"] is None

    # ConstantCol: constant column
    assert "ConstantCol" in columns
    const_stats = columns["ConstantCol"]["statistics"]
    assert const_stats is not None
    assert const_stats["min"] == const_stats["max"] == 42.0
    assert const_stats["std"] == 0.0

    # Score: unprofiled numeric without statistics
    assert "Score" in columns
    assert columns["Score"]["statistics"] is None


def test_top_categories_schema():
    from backend.schemas import ColumnInfo, TopCategory

    col = ColumnInfo(
        name="Dept",
        dtype="object",
        primitive_type="categorical",
        unique_count=3,
        missing_count=0,
        missing_rate=0.0,
        flags=[],
        valid_count=100,
        other_count=5,
        top_categories=[
            TopCategory(value="IT", count=60),
            TopCategory(value="HR", count=35),
        ],
    )
    assert col.valid_count == 100
    assert col.other_count == 5
    assert len(col.top_categories) == 2
    assert col.top_categories[0].value == "IT"
    assert col.top_categories[0].count == 60


def test_mock_endpoint_returns_top_categories():
    response = client.get("/api/v1/datasets/ds_class_01/info")
    assert response.status_code == 200
    data = response.json()
    columns = {c["name"]: c for c in data["columns"]}

    assert "Department" in columns
    dept = columns["Department"]
    assert dept["valid_count"] == 1000
    assert dept["other_count"] == 50
    assert isinstance(dept["top_categories"], list)
    assert len(dept["top_categories"]) == 5
    assert dept["top_categories"][0]["value"] == "Mühendislik"
    assert dept["top_categories"][0]["count"] == 420

    assert "Education" in columns
    edu = columns["Education"]
    assert edu["valid_count"] == 1000
    assert edu["other_count"] == 0
    assert len(edu["top_categories"]) == 4
    assert edu["top_categories"][0]["value"] == "Lisans"




