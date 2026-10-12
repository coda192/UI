from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentCreateRequest(BaseModel):
    project_name: str
    dataset_id: str
    algorithm_type: str
    task: str
    mode: str = "train"

    model: Optional[Dict[str, Any]] = None
    data_analyze: Optional[Dict[str, Any]] = None
    preprocess: Optional[Dict[str, Any]] = None
    data_split: Optional[Dict[str, Any]] = None
    tuning: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None


class ExperimentMetadata(BaseModel):
    id: str
    name: str
    status: str  # "configured", "running", "completed", "failed"
    created_at: datetime
