from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class VisualizationData(BaseModel):
    type: str
    title: str
    data: Dict[str, Any] = {} # Flexible inner data for plotly/charts
    html_content: Optional[str] = None # Raw interactive Plotly HTML string
    file_path: Optional[str] = None # Relative path in outputs directory
    metadata: Dict[str, Any] = {}

class ArtifactReference(BaseModel):
    artifact_id: str
    type: str # e.g. "model", "scaler", "report"
    name: str

class MetricsData(BaseModel):
    # A structured envelope. Different tasks will populate different fields or use the generic dict fallback if needed.
    # However, keeping it strongly typed where possible is better.
    classification_metrics: Optional[Dict[str, float]] = None # F1, Accuracy, etc.
    regression_metrics: Optional[Dict[str, float]] = None # RMSE, MAE, etc.
    anomaly_metrics: Optional[Dict[str, Any]] = None # total_samples, anomaly_rate, etc.
    other: Optional[Dict[str, Any]] = None

class ExperimentResultResponse(BaseModel):
    experiment_id: str
    task: str
    status: str
    algorithm: str
    has_ground_truth: bool
    summary: Dict[str, Any] = {}
    metrics: MetricsData = MetricsData()
    visualizations: List[VisualizationData] = []
    tables: Dict[str, List[Dict[str, Any]]] = {} # e.g. "anomalous_records": [...]
    artifacts: List[ArtifactReference] = []
    error: Optional[str] = None
