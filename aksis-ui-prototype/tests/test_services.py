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
