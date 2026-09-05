from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ModelConfig(BaseModel):
    algorithm: str
    preset: Optional[str] = None
    overrides: Dict[str, Any] = {}

class PreprocessingConfig(BaseModel):
    missing_value: Optional[str] = None
    encoding: Optional[str] = None
    scaling: Optional[str] = None

class ValidationConfig(BaseModel):
    strategy: str # e.g. holdout, kfold
    test_size: Optional[float] = 0.2
    folds: Optional[int] = 5

class TuningConfig(BaseModel):
    enabled: bool = False
    sampler: Optional[str] = None
    pruner: Optional[str] = None
    trials: Optional[int] = 10
    scoring: Optional[str] = None
    search_space_options: Dict[str, Any] = {}

class EvaluationConfig(BaseModel):
    metrics: List[str] = []

class VisualizationConfig(BaseModel):
    types: List[str] = []

class ExperimentCreateRequest(BaseModel):
    name: str
    dataset_id: str
    learning_type: str
    task: str
    mode: str = "train"
    model: ModelConfig
    preprocessing: Optional[PreprocessingConfig] = None
    validation: Optional[ValidationConfig] = None
    tuning: Optional[TuningConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    visualization: Optional[VisualizationConfig] = None

class ExperimentMetadata(BaseModel):
    id: str
    name: str
    status: str # "configured", "running", "completed", "failed"
    created_at: datetime
