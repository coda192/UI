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

    preprocessing: Dict[str, Any] = Field(
        default_factory=dict
    )

    data_split: Dict[str, Any] = Field(default_factory=dict)
    data_analyze: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    visualization: Dict[str, Any] = Field(default_factory=dict)

    parameter_schema: Optional[
        Dict[str, Dict[str, Dict[str, Any]]]
    ] = None