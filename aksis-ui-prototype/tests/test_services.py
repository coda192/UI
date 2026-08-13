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
    
    res_labeled = service.get_experiment_results(meta_labeled.id)
    assert res_labeled.has_ground_truth is True
    assert res_labeled.metrics.anomaly_metrics.get("f1") is not None

