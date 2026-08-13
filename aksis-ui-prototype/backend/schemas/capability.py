from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class CapabilityResponse(BaseModel):
    learning_types: List[str]
    tasks: Dict[str, List[str]] # e.g. "supervised": ["classification", "regression"]
    modes: List[str] # e.g. ["local", "distributed"]
    algorithms: Dict[str, List[str]] # e.g. "classification": ["Random Forest", "CatBoost"]
    model_presets: List[str]
    preprocessing_strategies: Dict[str, List[str]] # e.g. "missing_value": ["mean", "median", "drop"]
    validation_options: List[str] # e.g. "kfold", "holdout"
    tuning_options: Dict[str, List[str]] # e.g. "sampler": ["tpe", "random"]
    scoring_options: Dict[str, List[str]] # e.g. "classification": ["f1", "accuracy"]
    evaluation_capabilities: List[str]
    visualization_capabilities: List[str]
