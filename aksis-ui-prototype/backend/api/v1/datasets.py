from fastapi import APIRouter, Depends, HTTPException
from typing import List
from backend.schemas import DatasetMetadata
from backend.services.base import AksisService
from backend.api.deps import get_service

router = APIRouter()

@router.get("/", response_model=List[DatasetMetadata])
def list_datasets(service: AksisService = Depends(get_service)):
    return service.list_datasets()

@router.get("/{dataset_id}", response_model=DatasetMetadata)
def get_dataset(dataset_id: str, service: AksisService = Depends(get_service)):
    try:
        return service.get_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
