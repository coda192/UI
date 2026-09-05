from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AlgorithmMetadata(BaseModel):
    display_name: str
    description: str
    strengths: List[str]
    limitations: List[str]
    best_for: List[str]

class CapabilityResponse(BaseModel):
    learning_types: List[str]
    tasks: Dict[str, List[str]] # e.g. "supervised": ["classification", "regression"]
    modes: List[str] # e.g. ["train", "tune"]
    algorithms: Dict[str, List[str]] # e.g. "classification": ["logreg", "catboost"]
    model_presets: List[str]
    preprocessing_strategies: Dict[str, List[str]] # e.g. "missing_value": ["mean", "median", "drop"]
    validation_options: List[str] # e.g. "kfold", "holdout"
    tuning_options: Dict[str, List[str]] # e.g. "sampler": ["tpe", "random"]
    scoring_options: Dict[str, List[str]] # e.g. "classification": ["f1", "accuracy"]
    evaluation_capabilities: List[str]
    visualization_capabilities: List[str]
    algorithm_metadata: Optional[Dict[str, AlgorithmMetadata]] = None
