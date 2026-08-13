from fastapi import APIRouter, Depends
from backend.schemas import CapabilityResponse
from backend.services.base import AksisService
from backend.api.deps import get_service

router = APIRouter()

@router.get("/", response_model=CapabilityResponse)
def get_capabilities(service: AksisService = Depends(get_service)):
    return service.get_capabilities()
