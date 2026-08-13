from fastapi import APIRouter, Depends
from backend.schemas import InferenceRequest, InferenceResponse
from backend.services.base import AksisService
from backend.api.deps import get_service

router = APIRouter()

@router.post("/", response_model=InferenceResponse)
def run_inference(req: InferenceRequest, service: AksisService = Depends(get_service)):
    return service.run_inference(req)
