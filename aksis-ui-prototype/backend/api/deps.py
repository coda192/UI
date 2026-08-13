from fastapi import Depends
from backend.services.factory import get_aksis_service
from backend.services.base import AksisService

def get_service() -> AksisService:
    return get_aksis_service()
