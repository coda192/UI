from fastapi import APIRouter
from backend.api.v1 import capabilities, datasets, experiments, artifacts, inference

api_router = APIRouter()
api_router.include_router(capabilities.router, prefix="/capabilities", tags=["capabilities"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(artifacts.router, prefix="/artifacts", tags=["artifacts"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
