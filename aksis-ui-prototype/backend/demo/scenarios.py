from typing import Optional
from backend.schemas import (
    ExperimentResultResponse,
    MetricsData,
    VisualizationData,
    ArtifactReference
)

def generate_mock_result(
    experiment_id: str,
    task: str,
    algorithm: str,
    has_ground_truth: bool
) -> ExperimentResultResponse:
    metrics = MetricsData()
    visualizations = []
    summary = {}
    tables = {}
    artifacts = [
        ArtifactReference(artifact_id=f"model_{experiment_id}", type="model", name=f"{algorithm} Model")
    ]
    
    if task == "classification":
        metrics.classification_metrics = {
            "accuracy": 0.92,
            "f1": 0.90,
            "precision": 0.91,
            "recall": 0.89
        }
        visualizations.append(
            VisualizationData(
                type="feature_importance",
                title="Top Features",
                data={"features": ["Age", "Income", "Usage"], "importance": [0.4, 0.35, 0.25]}
            )
        )
        tables["confusion_matrix"] = [
            {"actual": "Positive", "predicted_positive": 890, "predicted_negative": 110},
            {"actual": "Negative", "predicted_positive": 50, "predicted_negative": 8950}
        ]
        
    elif task == "regression":
        metrics.regression_metrics = {
            "rmse": 12.5,
            "mae": 8.3,
            "r2": 0.85
        }
        visualizations.append(
            VisualizationData(
                type="residual_plot",
                title="Residuals",
                data={"predicted": [10, 20, 30], "residuals": [1.1, -0.5, 0.8]}
            )
        )
        
    elif task == "anomaly_detection":
        summary = {
            "total_samples": 50000,
            "normal_samples": 49500,
            "detected_anomalies": 500,
            "anomaly_rate": 0.01
        }
        if has_ground_truth:
            # Labeled anomaly
            metrics.anomaly_metrics = {
                "f1": 0.85,
                "precision": 0.90,
                "recall": 0.81
            }
            tables["confusion_matrix"] = [
                {"actual": "Anomaly", "predicted_anomaly": 405, "predicted_normal": 95},
                {"actual": "Normal", "predicted_anomaly": 45, "predicted_normal": 49455}
            ]
        else:
            # Unlabeled anomaly MUST NOT expose accuracy/f1
            metrics.anomaly_metrics = {}
            
        visualizations.append(
            VisualizationData(
                type="anomaly_score_distribution",
                title="Anomaly Score Distribution",
                data={"bins": [0.1, 0.5, 0.9], "counts": [40000, 9000, 1000]}
            )
        )
        tables["top_anomalies"] = [
            {"id": "A123", "prediction": -1, "anomaly_score": 0.99},
            {"id": "B456", "prediction": -1, "anomaly_score": 0.98},
            {"id": "C789", "prediction": 1, "anomaly_score": 0.12}
        ]

    return ExperimentResultResponse(
        experiment_id=experiment_id,
        task=task,
        status="completed",
        algorithm=algorithm,
        has_ground_truth=has_ground_truth,
        summary=summary,
        metrics=metrics,
        visualizations=visualizations,
        tables=tables,
        artifacts=artifacts
    )
