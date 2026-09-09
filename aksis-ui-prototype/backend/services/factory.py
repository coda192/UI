import os
import logging
from .base import AksisService
from .mock_service import MockAksisService
from .aksis_service import RealAksisService

logger = logging.getLogger("backend.services.factory")

def get_aksis_service() -> AksisService:
    provider = os.getenv("AKSIS_PROVIDER", "mock").lower()
    if provider == "mock":
        logger.info("AKSIS provider: MockAksisService")
        return MockAksisService()
    elif provider == "aksis":
        logger.info("AKSIS provider: RealAksisService")
        print("AKSIS provider: RealAksisService")
        return RealAksisService()
    else:
        raise ValueError(f"Unknown AKSIS_PROVIDER: {provider}")
