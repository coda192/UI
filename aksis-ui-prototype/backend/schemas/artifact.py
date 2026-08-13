from pydantic import BaseModel
from datetime import datetime

class ArtifactMetadata(BaseModel):
    id: str
    name: str
    type: str # "model", "preprocessor", "plot"
    experiment_id: str
    created_at: datetime
