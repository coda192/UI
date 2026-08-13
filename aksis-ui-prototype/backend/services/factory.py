import os
from .base import AksisService
from .mock_service import MockAksisService
from .aksis_service import RealAksisService

def get_aksis_service() -> AksisService:
    provider = os.getenv("AKSIS_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockAksisService()
    elif provider == "aksis":
        return RealAksisService()
    else:
        raise ValueError(f"Unknown AKSIS_PROVIDER: {provider}")
