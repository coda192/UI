from pydantic import BaseModel,Field
from typing import List, Dict, Any, Optional

class AlgorithmMetadata(BaseModel):
    display_name: str
    description: str
    strengths: List[str]
    limitations: List[str]
    best_for: List[str]

class ConfigApplicability(BaseModel):
    algorithm_types: List[str] = Field(default_factory=list)
    supported_tasks: List[str] = Field(default_factory=list)

class TuningCapabilities(BaseModel):
    samplers: Dict[str, List[str]] = Field(default_factory=dict)
    pruners: Dict[str, List[str]] = Field(default_factory=dict)
    aggregation: Dict[str, List[str]] = Field(default_factory=dict)

    space_presets: Dict[str, List[str]] = Field(default_factory=dict)
    objectives: Dict[str, List[str]] = Field(default_factory=dict)

class CapabilityResponse(BaseModel):
    learning_types: List[str]
    tasks: Dict[str, List[str]] # e.g. "supervised": ["classification", "regression"]
    modes: List[str] # e.g. ["train", "tune", "predict"]
    algorithms: Dict[str, List[str]] # e.g. "classification": ["logreg", "catboost"]
    model_presets: List[str]
    preprocessing_strategies: Dict[str, List[str]] # e.g. "missing_value": ["mean", "median", "drop"]
    validation_options: List[str] # e.g. "kfold", "holdout"
    tuning_options: Dict[str, List[str]] # e.g. "sampler": ["tpe", "random"]
    scoring_options: Dict[str, List[str]] # e.g. "classification": ["f1", "accuracy"]
    evaluation_capabilities: List[str]
    visualization_capabilities: List[str]
    algorithm_metadata: Optional[Dict[str, AlgorithmMetadata]] = None
    parameter_schema: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None

class CapabilityResponse(BaseModel):

    learning_types: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    modes: List[str] = Field(default_factory=list)

    config_applicability: Dict[str, ConfigApplicability] = Field(
        default_factory=dict
    )

    algorithms: Dict[str, List[str]] = Field(default_factory=dict)

    algorithm_metadata: Optional[
        Dict[str, AlgorithmMetadata]
    ] = None

    model_presets: Dict[str, List[str]] = Field(
        default_factory=dict
    )

    tuning: TuningCapabilities = Field(
        default_factory=TuningCapabilities
    )

    preprocessing: Dict[str, List[str]] = Field(
        default_factory=dict
    )

    data_split: Dict[str, Any] = Field(default_factory=dict)
    data_analyze: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    visualization: Dict[str, Any] = Field(default_factory=dict)

    parameter_schema: Optional[
        Dict[str, Dict[str, Dict[str, Any]]]
    ] = None