from typing import List, Dict
import uuid
from datetime import datetime, timezone
import threading
import time

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
from backend.demo import CAPABILITIES, DATASETS, generate_mock_result

class MockAksisService(AksisService):
    def __init__(self):
        self._experiments: Dict[str, dict] = {}
        # Store full request and metadata in memory
    
    def get_capabilities(self) -> CapabilityResponse:
        return CAPABILITIES
        
    def list_datasets(self) -> List[DatasetMetadata]:
        return DATASETS
        
    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        for ds in DATASETS:
            if ds.id == dataset_id:
                return ds
        raise ValueError(f"Dataset {dataset_id} not found")
        
    def create_experiment(self, req: ExperimentCreateRequest) -> ExperimentMetadata:
        exp_id = f"exp_{uuid.uuid4().hex[:8]}"
        meta = ExperimentMetadata(
            id=exp_id,
            name=req.name,
            status="configured",
            created_at=datetime.now(timezone.utc)
        )
        self._experiments[exp_id] = {
            "metadata": meta,
            "request": req
        }
        return meta
        
    def list_experiments(self) -> List[ExperimentMetadata]:
        return [exp["metadata"] for exp in self._experiments.values()]
        
    def get_experiment(self, experiment_id: str) -> ExperimentMetadata:
        if experiment_id not in self._experiments:
            raise ValueError("Experiment not found")
        return self._experiments[experiment_id]["metadata"]
        
    def run_experiment(self, experiment_id: str) -> None:
        if experiment_id not in self._experiments:
            raise ValueError("Experiment not found")
            
        exp_data = self._experiments[experiment_id]
        meta = exp_data["metadata"]
        
        if meta.status != "configured":
            raise ValueError("Experiment already running or completed")
            
        # Update status to running
        meta.status = "running"
        
        # Simulate asynchronous execution (fire and forget thread)
        def _simulate_run():
            time.sleep(3) # Simulate a short delay
            meta.status = "completed"
            
        threading.Thread(target=_simulate_run, daemon=True).start()
        
    def get_experiment_results(self, experiment_id: str) -> ExperimentResultResponse:
        if experiment_id not in self._experiments:
            raise ValueError("Experiment not found")
            
        exp_data = self._experiments[experiment_id]
        meta = exp_data["metadata"]
        req = exp_data["request"]
        
        if meta.status != "completed":
            raise ValueError(f"Experiment is not completed (status: {meta.status})")
            
        ds = self.get_dataset(req.dataset_id)
        has_ground_truth = ds.target is not None
            
        return generate_mock_result(
            experiment_id=experiment_id,
            task=req.task,
            algorithm=req.model.algorithm,
            has_ground_truth=has_ground_truth
        )
        
    def list_artifacts(self) -> List[ArtifactMetadata]:
        artifacts = []
        for exp_id, exp_data in self._experiments.items():
            if exp_data["metadata"].status == "completed":
                artifacts.append(
                    ArtifactMetadata(
                        id=f"model_{exp_id}",
                        name=f"Model for {exp_data['metadata'].name}",
                        type="model",
                        experiment_id=exp_id,
                        created_at=exp_data["metadata"].created_at
                    )
                )
        return artifacts
        
    def run_inference(self, req: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            status="completed",
            predictions_preview=[{"id": 1, "pred": 0.5}],
            total_predictions=1
        )
