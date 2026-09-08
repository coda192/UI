import os
from backend.services.factory import get_aksis_service
from backend.services.mock_service import MockAksisService
from backend.services.aksis_service import RealAksisService
from backend.schemas import ExperimentCreateRequest, ModelConfig

def test_service_factory_mock():
    os.environ["AKSIS_PROVIDER"] = "mock"
    service = get_aksis_service()
    assert isinstance(service, MockAksisService)

def test_service_factory_aksis():
    os.environ["AKSIS_PROVIDER"] = "aksis"
    service = get_aksis_service()
    assert isinstance(service, RealAksisService)

def test_mock_service_flow():
    service = MockAksisService()
    
    # Create
    req = ExperimentCreateRequest(
        name="Test",
        dataset_id="ds_class_01",
        learning_type="supervised",
        task="classification",
        model=ModelConfig(algorithm="CatBoost")
    )
    meta = service.create_experiment(req)
    assert meta.status == "configured"
    
    # Results should fail if not completed
    try:
        service.get_experiment_results(meta.id)
        assert False, "Should have raised exception"
    except ValueError:
        pass
        
    # (Running is async, so we'd need to mock sleep for a full test, but creation works)

def test_mock_service_anomaly_detection_results():
    service = MockAksisService()
    
    # Test Unlabeled Anomaly
    req_unlabeled = ExperimentCreateRequest(
        name="Unlabeled Anomaly Test",
        dataset_id="ds_anom_unlabeled_01",
        learning_type="unsupervised",
        task="anomaly_detection",
        model=ModelConfig(algorithm="Isolation Forest")
    )
    meta_unlabeled = service.create_experiment(req_unlabeled)
    # Directly complete metadata for test assertions
    meta_unlabeled.status = "completed"
    
    res_unlabeled = service.get_experiment_results(meta_unlabeled.id)
    assert res_unlabeled.has_ground_truth is False
    assert res_unlabeled.metrics.classification_metrics is None
    assert res_unlabeled.metrics.anomaly_metrics == {}
    assert "top_anomalies" in res_unlabeled.tables
    
    # Test Labeled Anomaly
    req_labeled = ExperimentCreateRequest(
        name="Labeled Anomaly Test",
        dataset_id="ds_anom_labeled_01",
        learning_type="unsupervised",
        task="anomaly_detection",
        model=ModelConfig(algorithm="Isolation Forest")
    )
    meta_labeled = service.create_experiment(req_labeled)
    meta_labeled.status = "completed"
    
def test_real_aksis_service_dataspec_mapping_and_security():
    from backend.services.aksis_service import RealAksisService, _DATASET_INDEX
    from types import SimpleNamespace
    
    # Simulate a raw DataSpec object with sensitive backend fields
    raw_spec = SimpleNamespace(
        id="credit_card_fraud",
        display_name="Kredi Kartı Dolandırıcılığı",
        description="Gerçek zamanlı fraud tespiti veri seti",
        source="oracle_db",
        local_data=True,
        target="Class",
        columns_to_use=["Time", "Amount", "V1", "V2"],
        # Sensitive fields that MUST be excluded:
        host="192.168.1.100",
        port=1521,
        user="db_admin",
        password="SuperSecretPassword123!",
        catalog="FINANCE",
        schema="TRANSACTIONS",
        http_schema="https",
        data_query="SELECT * FROM secret_transactions_table"
    )
    
    metadata = RealAksisService._map_dataspec_to_metadata(raw_spec)
    
    # Verify UI-safe fields
    assert metadata.id == "credit_card_fraud"
    assert metadata.display_name == "Kredi Kartı Dolandırıcılığı"
    assert metadata.description == "Gerçek zamanlı fraud tespiti veri seti"
    assert metadata.source == "oracle_db"
    assert metadata.local_data is True
    assert metadata.target == "Class"
    assert metadata.columns_to_use == ["Time", "Amount", "V1", "V2"]
    
    # Verify sensitive fields are NOT in the schema
    meta_dict = metadata.model_dump()
    for sensitive in ["host", "port", "user", "password", "catalog", "schema", "http_schema", "data_query"]:
        assert sensitive not in meta_dict, f"Sensitive field '{sensitive}' must not be exposed!"

def test_real_aksis_service_dataset_index_registry(monkeypatch):
    import backend.services.aksis_service as aksis_mod
    from backend.services.aksis_service import RealAksisService
    from types import SimpleNamespace
    
    mock_spec1 = SimpleNamespace(
        id="reg_housing",
        display_name="Konut Fiyatları",
        description="Ev fiyat tahmini",
        source="csv_file",
        local_data=True,
        target="SalePrice",
        columns_to_use=["LotArea", "YearBuilt"]
    )
    mock_spec2 = SimpleNamespace(
        id="cls_churn",
        display_name=None, # Missing display_name fallback test
        description=None,  # Missing description test
        source="db",
        local_data=False,
        target="Churn",
        columns_to_use=None
    )
    
    # Populate _DATASET_INDEX
    test_index = {
        "reg_housing": mock_spec1,
        "cls_churn": mock_spec2
    }
    monkeypatch.setattr(aksis_mod, "_DATASET_INDEX", test_index)
    
    service = RealAksisService()
    datasets = service.list_datasets()
    
    assert len(datasets) == 2
    ds_ids = [d.id for d in datasets]
    assert "reg_housing" in ds_ids
    assert "cls_churn" in ds_ids
    
    # Verify fallback for missing display_name
    churn_meta = service.get_dataset("cls_churn")
    assert churn_meta.id == "cls_churn"
    assert churn_meta.name == "cls_churn"  # fallback to id
    assert churn_meta.display_name is None
    assert churn_meta.description is None
    
    # Verify retrieval of reg_housing
    housing_meta = service.get_dataset("reg_housing")
    assert housing_meta.id == "reg_housing"
    assert housing_meta.display_name == "Konut Fiyatları"
    assert housing_meta.target == "SalePrice"

