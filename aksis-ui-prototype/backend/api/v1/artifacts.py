from fastapi import APIRouter, Depends
from typing import List
from backend.schemas import ArtifactMetadata
from backend.services.base import AksisService
from backend.api.deps import get_service

router = APIRouter()

@router.get("/", response_model=List[ArtifactMetadata])
def list_artifacts(service: AksisService = Depends(get_service)):
    return service.list_artifacts()
