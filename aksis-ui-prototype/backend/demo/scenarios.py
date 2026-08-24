import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from backend.schemas import (
    ExperimentResultResponse,
    MetricsData,
    VisualizationData,
    ArtifactReference
)

def _create_feature_importance_plot() -> str:
    features = ["MonthlyCharge", "Age", "ContractLength", "TotalCharges", "SupportCalls"]
    importance = [0.38, 0.27, 0.18, 0.11, 0.06]
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis'
        )
    ))
    fig.update_layout(
        title="Nitelik Önem Düzeyleri (Feature Importance)",
        xaxis_title="Önem Skoru",
        yaxis_title="Nitelik",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380
    )
    return fig.to_html(include_plotlyjs='cdn', full_html=False)

def _create_residual_plot() -> str:
    predicted = [120000, 180000, 240000, 310000, 420000, 500000, 620000]
    actual = [118000, 185000, 232000, 315000, 410000, 512000, 608000]
    residuals = [a - p for a, p in zip(actual, predicted)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predicted,
        y=residuals,
        mode='markers',
        marker=dict(size=10, color='#3B82F6', opacity=0.8),
        name="Artıklar (Residuals)"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444")
    fig.update_layout(
        title="Artık Değer Grafiği (Residual Plot)",
        xaxis_title="Tahmin Edilen Değer (Predicted)",
        yaxis_title="Hata / Artık (Actual - Predicted)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380
    )
    return fig.to_html(include_plotlyjs='cdn', full_html=False)

def _create_anomaly_score_plot() -> str:
    import numpy as np
    np.random.seed(42)
    scores = np.concatenate([
        np.random.normal(loc=0.15, scale=0.08, size=4000),
        np.random.normal(loc=0.75, scale=0.12, size=200)
    ])
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=50,
        marker=dict(color='#6366F1'),
        name="Skor Dağılımı"
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#EF4444", annotation_text="Anomali Eşik Değeri (0.50)")
    fig.update_layout(
        title="Anomali Skor Dağılımı (Anomaly Score Distribution)",
        xaxis_title="Anomali Skoru [0 - 1]",
        yaxis_title="Kayıt Sayısı",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380
    )
    return fig.to_html(include_plotlyjs='cdn', full_html=False)

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
        ArtifactReference(artifact_id=f"model_{experiment_id}", type="model", name=f"{algorithm} Modeli")
    ]
    
    if task == "classification":
        metrics.classification_metrics = {
            "accuracy": 0.924,
            "f1": 0.908,
            "precision": 0.915,
            "recall": 0.892
        }
        visualizations.append(
            VisualizationData(
                type="feature_importance",
                title="En Önemli Nitelikler (Feature Importance)",
                data={"features": ["MonthlyCharge", "Age", "ContractLength"], "importance": [0.38, 0.27, 0.18]},
                html_content=_create_feature_importance_plot()
            )
        )
        tables["confusion_matrix"] = [
            {"Gerçek Sınıf (Actual)": "Churn (Pozitif)", "Tahmin Churn": 892, "Tahmin Normal": 108},
            {"Gerçek Sınıf (Actual)": "Normal (Negatif)", "Tahmin Churn": 48, "Tahmin Normal": 8952}
        ]
        
    elif task == "regression":
        metrics.regression_metrics = {
            "rmse": 12450.0,
            "mae": 8320.0,
            "r2": 0.885
        }
        visualizations.append(
            VisualizationData(
                type="residual_plot",
                title="Hata Dağılımı (Residual Plot)",
                data={"predicted": [120000, 180000, 240000], "residuals": [-2000, 5000, -8000]},
                html_content=_create_residual_plot()
            )
        )
        
    elif task == "anomaly_detection":
        summary = {
            "total_samples": 50000,
            "normal_samples": 49500,
            "detected_anomalies": 500,
            "anomaly_rate": "1.00%"
        }
        if has_ground_truth:
            # Labeled anomaly
            metrics.anomaly_metrics = {
                "f1": 0.854,
                "precision": 0.902,
                "recall": 0.811
            }
            tables["confusion_matrix"] = [
                {"Gerçek Durum (Actual)": "Anomali", "Tahmin Anomali": 405, "Tahmin Normal": 95},
                {"Gerçek Durum (Actual)": "Normal", "Tahmin Anomali": 45, "Tahmin Normal": 49455}
            ]
        else:
            # Unlabeled anomaly MUST NOT expose accuracy/f1
            metrics.anomaly_metrics = {}
            
        visualizations.append(
            VisualizationData(
                type="anomaly_score_distribution",
                title="Anomali Skor Dağılımı (Anomaly Score Distribution)",
                data={"bins": [0.1, 0.5, 0.9], "counts": [40000, 9000, 1000]},
                html_content=_create_anomaly_score_plot()
            )
        )
        tables["top_anomalies"] = [
            {"Log / Kayıt ID": "LOG_A1092", "Tahmin": "Anomali (-1)", "Anomali Skoru": 0.992},
            {"Log / Kayıt ID": "LOG_B8831", "Tahmin": "Anomali (-1)", "Anomali Skoru": 0.984},
            {"Log / Kayıt ID": "LOG_C4402", "Tahmin": "Anomali (-1)", "Anomali Skoru": 0.961},
            {"Log / Kayıt ID": "LOG_D1198", "Tahmin": "Normal (1)", "Anomali Skoru": 0.120}
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
