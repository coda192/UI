from abc import ABC, abstractmethod
from typing import List

from backend.schemas import (
    CapabilityResponse,
    DatasetMetadata,
    ExperimentCreateRequest,
    ExperimentMetadata,
    ExperimentResultResponse,
    ArtifactMetadata,
    InferenceRequest,
    InferenceResponse
)

class AksisService(ABC):
    @abstractmethod
    def get_capabilities(self) -> CapabilityResponse:
        pass
        
    @abstractmethod
    def list_datasets(self) -> List[DatasetMetadata]:
        pass
        
    @abstractmethod
    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        pass
        
    @abstractmethod
    def create_experiment(self, req: ExperimentCreateRequest) -> ExperimentMetadata:
        pass
        
    @abstractmethod
    def list_experiments(self) -> List[ExperimentMetadata]:
        pass
        
    @abstractmethod
    def get_experiment(self, experiment_id: str) -> ExperimentMetadata:
        pass
        
    @abstractmethod
    def run_experiment(self, experiment_id: str) -> None:
        pass
        
    @abstractmethod
    def get_experiment_results(self, experiment_id: str) -> ExperimentResultResponse:
        pass
        
    @abstractmethod
    def list_artifacts(self) -> List[ArtifactMetadata]:
        pass
        
    @abstractmethod
    def run_inference(self, req: InferenceRequest) -> InferenceResponse:
        pass
