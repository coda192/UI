import pytest
import os
from backend.api.deps import get_service

@pytest.fixture(autouse=True)
def reset_service_cache():
    # Ensure test environment defaults to mock provider and clears cached service
    os.environ["AKSIS_PROVIDER"] = "mock"
    get_service.cache_clear()
    yield
    get_service.cache_clear()
