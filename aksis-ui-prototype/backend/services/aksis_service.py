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
from backend.services.base import AksisService

class RealAksisService(AksisService):
    """
    Adapter layer for the real AKSIS framework.
    All methods here contain AKSIS_INTEGRATION_POINT markers.
    """
    
    def get_capabilities(self) -> CapabilityResponse:
        # AKSIS_INTEGRATION_POINT
        # Query real AKSIS registries for supported algorithms/tasks.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def list_datasets(self) -> List[DatasetMetadata]:
        # AKSIS_INTEGRATION_POINT
        # Fetch a list of datasets from AKSIS Data Catalog/DataSpec.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        # AKSIS_INTEGRATION_POINT
        # Fetch metadata for a specific dataset from AKSIS.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def create_experiment(self, req: ExperimentCreateRequest) -> ExperimentMetadata:
        # AKSIS_INTEGRATION_POINT
        # Translate the API ExperimentCreateRequest into the real AKSIS ExperimentConfig.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def list_experiments(self) -> List[ExperimentMetadata]:
        # AKSIS_INTEGRATION_POINT
        # List previous experiments tracked by AKSIS.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def get_experiment(self, experiment_id: str) -> ExperimentMetadata:
        # AKSIS_INTEGRATION_POINT
        # Query AKSIS job/run status to get experiment state.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def run_experiment(self, experiment_id: str) -> None:
        # AKSIS_INTEGRATION_POINT
        # Fire the existing AKSIS execution pipeline (non-blocking).
        # E.g. call aksis.runner.run_experiment_async()
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def get_experiment_results(self, experiment_id: str) -> ExperimentResultResponse:
        # AKSIS_INTEGRATION_POINT
        # Fetch typed ML evaluation outputs from wherever AKSIS stores them.
        # Translate to ExperimentResultResponse API schema.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def list_artifacts(self) -> List[ArtifactMetadata]:
        # AKSIS_INTEGRATION_POINT
        # (Optional) Expose trained models and experiment outputs.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
        
    def run_inference(self, req: InferenceRequest) -> InferenceResponse:
        # AKSIS_INTEGRATION_POINT
        # (Optional) Run batch inference on a dataset using an existing model.
        raise NotImplementedError("Real AKSIS integration has not been configured.")
