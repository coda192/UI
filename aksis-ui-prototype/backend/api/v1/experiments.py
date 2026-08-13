from fastapi import APIRouter, Depends, HTTPException
from typing import List
from backend.schemas import (
    ExperimentCreateRequest,
    ExperimentMetadata,
    ExperimentResultResponse
)
from backend.services.base import AksisService
from backend.api.deps import get_service

router = APIRouter()

@router.post("/", response_model=ExperimentMetadata)
def create_experiment(req: ExperimentCreateRequest, service: AksisService = Depends(get_service)):
    return service.create_experiment(req)

@router.get("/", response_model=List[ExperimentMetadata])
def list_experiments(service: AksisService = Depends(get_service)):
    return service.list_experiments()

@router.get("/{experiment_id}", response_model=ExperimentMetadata)
def get_experiment(experiment_id: str, service: AksisService = Depends(get_service)):
    try:
        return service.get_experiment(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{experiment_id}/run")
def run_experiment(experiment_id: str, service: AksisService = Depends(get_service)):
    try:
        service.run_experiment(experiment_id)
        return {"message": "Experiment execution started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{experiment_id}/results", response_model=ExperimentResultResponse)
def get_experiment_results(experiment_id: str, service: AksisService = Depends(get_service)):
    try:
        return service.get_experiment_results(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
