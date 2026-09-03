import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.services.base import AksisService
from backend.schemas import (
    CapabilityResponse,
    DatasetMetadata,
    ExperimentCreateRequest,
    ExperimentMetadata,
    ExperimentResultResponse,
    VisualizationData,
    MetricsData,
    ArtifactMetadata,
    InferenceRequest,
    InferenceResponse
)

# ==============================================================================
# AKSIS İTHALAT (IMPORT) NOKTASI
# Özel / Şirket bilgisayarında aşağıdaki importların başındaki yorumları kaldırın:
# ==============================================================================
# try:
#     from aksis.config import ExperimentConfig, ModelConfig, PreprocessConfig, TuningConfig, ValidationConfig
#     from aksis.runner import run_experiment as aksis_run_experiment
#     from aksis.registry import model_registry, task_registry
#     from aksis.data import dataset_catalog
#     from aksis.inference import predict_batch
#     AKSIS_AVAILABLE = True
# except ImportError:
#     AKSIS_AVAILABLE = False
# ==============================================================================


class RealAksisService(AksisService):
    """
    Kurumsal AKSIS Çerçevesi için Minimal ve Güvenli Adaptör Katmanı.
    Bu sınıf doğrudan 'src/' altındaki kütüphane fonksiyonlarıyla haberleşir.
    """

    def __init__(self):
        # Deney durumlarını ve konfigürasyonlarını bellekte takip eden sözlük
        self._in_memory_status: Dict[str, dict] = {}

    def get_capabilities(self) -> CapabilityResponse:
        """
        PRIORITY 2: AKSIS bünyesinde kayıtlı algoritmaları, görevleri ve stratejileri döner.
        """
        # AKSIS_INTEGRATION_POINT: model_registry ve task_registry entegrasyonu
        # Örnek:
        # tasks = task_registry.get_all()
        # algorithms = model_registry.get_all_by_task()
        return CapabilityResponse(
            learning_types=["supervised", "unsupervised"],
            tasks={
                "supervised": ["classification", "regression"],
                "unsupervised": ["anomaly_detection"]
            },
            modes=["local"],
            algorithms={
                "classification": ["Logistic Regression", "Random Forest", "HistGradientBoosting", "SVC", "KNN", "CatBoost", "XGBoost"],
                "regression": ["Ridge", "SVR", "Random Forest", "HistGradientBoosting", "XGBoost"],
                "anomaly_detection": ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Elliptic Envelope"]
            },
            model_presets=["fast", "accurate", "interpretable"],
            preprocessing_strategies={
                "missing_value": ["mean", "median", "most_frequent", "drop"],
                "encoding": ["onehot", "label", "target"],
                "scaling": ["standard", "minmax", "robust"]
            },
            validation_options=["holdout", "kfold", "stratified_kfold"],
            tuning_options={
                "sampler": ["tpe", "random", "grid"],
                "pruner": ["median", "hyperband"]
            },
            scoring_options={
                "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
                "regression": ["rmse", "mae", "r2"],
                "anomaly_detection": ["f1", "precision", "recall"]
            },
            evaluation_capabilities=["confusion_matrix", "feature_importance", "residuals", "anomaly_distribution"],
            visualization_capabilities=["roc_curve", "pr_curve", "residual_plot", "feature_importance_plot", "anomaly_score_histogram"]
        )

    def list_datasets(self) -> List[DatasetMetadata]:
        """
        PRIORITY 5: Kayıtlı veri setlerini ve üst verilerini (schema/target/shape) listeler.
        """
        # AKSIS_INTEGRATION_POINT: dataset_catalog.list_specs()
        # specs = dataset_catalog.list_specs()
        # return [DatasetMetadata(...) for s in specs]
        raise NotImplementedError("RealAksisService.list_datasets henüz AKSIS veri kataloğuna bağlanmadı.")

    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        """
        PRIORITY 5: Tek bir veri setinin detaylı meta verilerini çeker.
        """
        # AKSIS_INTEGRATION_POINT: dataset_catalog.get_spec(dataset_id)
        raise NotImplementedError("RealAksisService.get_dataset henüz AKSIS veri kataloğuna bağlanmadı.")

    def create_experiment(self, req: ExperimentCreateRequest) -> ExperimentMetadata:
        """
        PRIORITY 1 & 2 & 3 & 4: API istek gövdesini alır ve deney kaydını başlatır.
        """
        exp_id = f"exp_{req.name}_{int(time.time())}"
        self._in_memory_status[exp_id] = {
            "status": "configured",
            "request": req,
            "error": None
        }
        return ExperimentMetadata(
            id=exp_id,
            name=req.name,
            status="configured",
            created_at=datetime.now(timezone.utc)
        )

    def list_experiments(self) -> List[ExperimentMetadata]:
        return [
            ExperimentMetadata(
                id=exp_id,
                name=data["request"].name,
                status=data["status"],
                created_at=datetime.now(timezone.utc)
            )
            for exp_id, data in self._in_memory_status.items()
        ]

    def get_experiment(self, experiment_id: str) -> ExperimentMetadata:
        if experiment_id not in self._in_memory_status:
            raise ValueError(f"Experiment {experiment_id} bulunamadı.")
        data = self._in_memory_status[experiment_id]
        return ExperimentMetadata(
            id=experiment_id,
            name=data["request"].name,
            status=data["status"],
            created_at=datetime.now(timezone.utc)
        )

    def run_experiment(self, experiment_id: str) -> None:
        """
        PRIORITY 6: Asenkron (non-blocking) olarak AKSIS run_experiment fonksiyonunu tetikler.
        """
        if experiment_id not in self._in_memory_status:
            raise ValueError(f"Experiment {experiment_id} bulunamadı.")

        req: ExperimentCreateRequest = self._in_memory_status[experiment_id]["request"]
        self._in_memory_status[experiment_id]["status"] = "running"

        def _execute():
            try:
                # AKSIS_INTEGRATION_POINT: ExperimentConfig nesnesini oluştur ve koşucuyu çağır
                # exp_config = ExperimentConfig(
                #     experiment_id=experiment_id,
                #     dataset_id=req.dataset_id,
                #     task=req.task,
                #     mode=req.mode,
                #     model_config=ModelConfig(**req.model.model_dump()),
                #     preprocess_config=PreprocessConfig(**req.preprocessing.model_dump()) if req.preprocessing else None,
                #     tuning_config=TuningConfig(**req.tuning.model_dump()) if req.tuning and req.tuning.enabled else None,
                #     validation_config=ValidationConfig(**req.validation.model_dump()) if req.validation else None
                # )
                # aksis_run_experiment(exp_config)
                
                self._in_memory_status[experiment_id]["status"] = "completed"
            except Exception as e:
                self._in_memory_status[experiment_id]["status"] = "failed"
                self._in_memory_status[experiment_id]["error"] = str(e)

        threading.Thread(target=_execute, daemon=True).start()

    def get_experiment_results(self, experiment_id: str) -> ExperimentResultResponse:
        """
        PRIORITY 7: outputs/{experiment_id}/ klasöründeki metrics.json ve *.html grafiklerini okur.
        """
        if experiment_id not in self._in_memory_status:
            raise ValueError(f"Experiment {experiment_id} bulunamadı.")

        exp_data = self._in_memory_status[experiment_id]
        if exp_data["status"] != "completed":
            raise ValueError(f"Deney henüz tamamlanmadı (Mevcut Durum: {exp_data['status']})")

        output_dir = os.path.join("outputs", experiment_id)
        metrics_file = os.path.join(output_dir, "metrics.json")
        
        metrics_dict = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)

        visualizations: List[VisualizationData] = []
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                if fname.endswith(".html"):
                    with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                        visualizations.append(
                            VisualizationData(
                                type=fname.replace(".html", ""),
                                title=fname.replace(".html", "").replace("_", " ").title(),
                                html_content=f.read()
                            )
                        )

        req = exp_data["request"]
        return ExperimentResultResponse(
            experiment_id=experiment_id,
            task=req.task,
            status="completed",
            algorithm=req.model.algorithm,
            has_ground_truth=True,
            metrics=MetricsData(**metrics_dict) if metrics_dict else MetricsData(),
            visualizations=visualizations
        )

    def list_artifacts(self) -> List[ArtifactMetadata]:
        """
        PRIORITY 8: Eğitilmiş model dosyalarını listeler.
        """
        artifacts = []
        for exp_id, data in self._in_memory_status.items():
            if data["status"] == "completed":
                artifacts.append(
                    ArtifactMetadata(
                        id=f"model_{exp_id}",
                        name=f"{data['request'].model.algorithm} Modeli ({exp_id})",
                        type="model",
                        experiment_id=exp_id,
                        created_at=datetime.now(timezone.utc)
                    )
                )
        return artifacts

    def run_inference(self, req: InferenceRequest) -> InferenceResponse:
        """
        PRIORITY 8: Eğitilmiş model ile hedef veri seti üzerinde toplu çıkarım yapar.
        """
        # AKSIS_INTEGRATION_POINT: predict_batch(model_path, dataset_id)
        raise NotImplementedError("RealAksisService.run_inference henüz AKSIS tahmin motoruna bağlanmadı.")
