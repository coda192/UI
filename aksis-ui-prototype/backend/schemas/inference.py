from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class InferenceRequest(BaseModel):
    artifact_id: str
    dataset_id: str

class InferenceResponse(BaseModel):
    status: str
    predictions_preview: List[Dict[str, Any]]
    total_predictions: int
