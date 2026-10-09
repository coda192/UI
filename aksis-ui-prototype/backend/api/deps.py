from fastapi import Depends

from functools import lru_cache
from backend.services.factory import get_aksis_service
from backend.services.base import AksisService

@lru_cache(maxsize=1)
def get_service() -> AksisService:
    return get_aksis_service()
