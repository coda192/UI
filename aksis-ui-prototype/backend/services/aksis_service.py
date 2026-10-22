import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.services.base import AksisService
from backend.schemas import (
    CapabilityResponse,
    DatasetMetadata,
    DatasetInfoResponse,
    ExperimentCreateRequest,
    ExperimentMetadata,
    ExperimentResultResponse,
    VisualizationData,
    MetricsData,
    ArtifactMetadata,
    InferenceRequest,
    InferenceResponse
)

logger = logging.getLogger("backend.services.aksis_service")

# ==============================================================================
# AKSIS CORE CAPABILITIES IMPORT
# ==============================================================================
try:
    from src.core.capabilities import get_capabilities as aksis_get_capabilities
    AKSIS_CORE_AVAILABLE = True
    AKSIS_CORE_IMPORT_ERROR = None
except ImportError as e:
    aksis_get_capabilities = None
    AKSIS_CORE_AVAILABLE = False
    AKSIS_CORE_IMPORT_ERROR = str(e)


def _get_aksis_dataset_accessors():
    try:
        from src.core.registries.data_registry import (
            get_all_datasets,
            get_dataset,
        )
    except ImportError as exc:
        raise RuntimeError(
            "AKSIS dataset registry is unavailable. "
            "Real provider requires the AKSIS src package."
        ) from exc

    return get_all_datasets, get_dataset


def _get_aksis_runner_accessors():
    try:
        from src.core.builders.experiment_config_builder import build_experiment_config
        from runner import run_experiment as aksis_run_experiment
    except ImportError as exc:
        raise RuntimeError(
            "AKSIS experiment runner is unavailable. "
            "Real experiment execution requires the AKSIS src package."
        ) from exc

    return build_experiment_config, aksis_run_experiment



class RealAksisService(AksisService):
    """
    Kurumsal AKSIS Çerçevesi için Minimal ve Güvenli Adaptör Katmanı.
    Bu sınıf doğrudan 'src/' altındaki kütüphane fonksiyonlarıyla ve registry ile haberleşir.
    """

    def __init__(self):
        # Deney durumlarını ve konfigürasyonlarını bellekte takip eden sözlük
        self._in_memory_status: Dict[str, dict] = {}

    def get_capabilities(self) -> CapabilityResponse:

        if aksis_get_capabilities is None:
            err_msg = (
                "AKSIS yetenek kütüphanesi "
                "(src.core.capabilities) bulunamadı veya yüklenemedi: "
                f"{AKSIS_CORE_IMPORT_ERROR}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        try:
            raw = aksis_get_capabilities()
        except Exception as exc:
            err_msg = (
                f"aksis_get_capabilities() çağrısı başarısız oldu: {exc}"
            )
            logger.error(err_msg, exc_info=True)
            raise RuntimeError(err_msg) from exc

        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif hasattr(raw, "dict"):
            raw = raw.dict()
        elif not isinstance(raw, dict):
            raw = dict(raw)

        try:
            return CapabilityResponse(**raw)
        except Exception as exc:
            err_msg = (
                "AKSIS capability çıktısı "
                "CapabilityResponse sözleşmesiyle uyumlu değil: "
                f"{exc}"
            )
            logger.error(err_msg, exc_info=True)
            raise RuntimeError(err_msg) from exc

    def _map_dataspec_to_metadata(self, spec: Any) -> DatasetMetadata:
        """
        AKSIS DataSpec nesnesini DatasetMetadata şemasına dönüştürür.
        Yalnızca şu alanları eşler:
          id, display_name, description, source, local_data, target, columns_to_use
        Çalışma zamanı/analiz alanları boş/None olarak ayarlanır:
          row_count=None, column_count=None, columns=[], identifier_columns=[], compatible_tasks=[]
        """
        if isinstance(spec, dict):
            spec_id = spec.get("id", "")
            display_name = spec.get("display_name")
            description = spec.get("description")
            source = spec.get("source")
            local_data = spec.get("local_data")
            target = spec.get("target")
            columns_to_use = spec.get("columns_to_use")
            compatible_tasks = spec.get("compatible_tasks", ())
        else:
            spec_id = getattr(spec, "id", "")
            display_name = getattr(spec, "display_name", None)
            description = getattr(spec, "description", None)
            source = getattr(spec, "source", None)
            local_data = getattr(spec, "local_data", None)
            target = getattr(spec, "target", None)
            columns_to_use = getattr(spec, "columns_to_use", None)
            compatible_tasks = getattr(spec, "compatible_tasks", ())

        if hasattr(source, "value"):
            source = source.value
        elif source is not None:
            source = str(source)

        if isinstance(compatible_tasks, (list, tuple, set)):
            norm_compatible_tasks = [str(t) for t in compatible_tasks]
        elif compatible_tasks:
            norm_compatible_tasks = [str(compatible_tasks)]
        else:
            norm_compatible_tasks = []

        return DatasetMetadata(
            id=spec_id,
            name=display_name or spec_id,
            display_name=display_name,
            description=description,
            source=source,
            local_data=local_data,
            target=target,
            columns_to_use=columns_to_use,
            row_count=None,
            column_count=None,
            columns=[],
            identifier_columns=[],
            compatible_tasks=norm_compatible_tasks,
        )

    def list_datasets(self) -> List[DatasetMetadata]:
        get_all_datasets, _ = _get_aksis_dataset_accessors()

        return [
            self._map_dataspec_to_metadata(spec)
            for spec in get_all_datasets()
        ]

    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        _, get_aksis_dataset = _get_aksis_dataset_accessors()

        try:
            spec = get_aksis_dataset(dataset_id)
        except (KeyError, IndexError):
            spec = None

        if spec is None:
            raise ValueError(f"Dataset '{dataset_id}' bulunamadı.")

        return self._map_dataspec_to_metadata(spec)

    def get_dataset_info(
        self,
        dataset_id: str,
        refresh: bool = False,
    ) -> DatasetInfoResponse:
        try:
            from src.core.data_processing.data_info import (
                get_dataset_info as aksis_get_dataset_info,
            )
        except ImportError as exc:
            raise RuntimeError(
                "AKSIS dataset info is unavailable. "
                "Real provider requires the AKSIS src package."
            ) from exc

        try:
            result = aksis_get_dataset_info(
                dataset_id,
                refresh=refresh,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Dataset '{dataset_id}' bulunamadı.") from exc

        if result is None:
            raise ValueError(f"Dataset '{dataset_id}' bulunamadı.")

        return DatasetInfoResponse(**result)

    def create_experiment(self, req: ExperimentCreateRequest) -> ExperimentMetadata:
        """
        API istek gövdesini alır ve deney kaydını başlatır.
        """
        exp_id = f"exp_{req.project_name}_{int(time.time())}"
        self._in_memory_status[exp_id] = {
            "status": "configured",
            "request": req,
            "error": None
        }
        return ExperimentMetadata(
            id=exp_id,
            name=req.project_name,
            status="configured",
            created_at=datetime.now(timezone.utc)
        )

    def list_experiments(self) -> List[ExperimentMetadata]:
        return [
            ExperimentMetadata(
                id=exp_id,
                name=data["request"].project_name,
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
            name=data["request"].project_name,
            status=data["status"],
            created_at=datetime.now(timezone.utc)
        )

    def run_experiment(self, experiment_id: str) -> None:
        """
        Asenkron (non-blocking) olarak AKSIS run_experiment fonksiyonunu tetikler.
        """
        if experiment_id not in self._in_memory_status:
            raise ValueError(f"Experiment {experiment_id} bulunamadı.")

        req: ExperimentCreateRequest = self._in_memory_status[experiment_id]["request"]
        self._in_memory_status[experiment_id]["status"] = "running"

        def _execute():
            try:
                build_experiment_config, aksis_run_experiment = _get_aksis_runner_accessors()
                payload = req.model_dump(exclude_none=True)
                exp_config = build_experiment_config(payload)
                aksis_run_experiment(exp_config)
                self._in_memory_status[experiment_id]["status"] = "completed"
            except Exception as exc:
                self._in_memory_status[experiment_id]["status"] = "failed"
                self._in_memory_status[experiment_id]["error"] = str(exc)
                logger.exception(f"Experiment execution failed for {experiment_id}: {exc}")

        threading.Thread(target=_execute, daemon=True).start()

    def get_experiment_results(self, experiment_id: str) -> ExperimentResultResponse:
        """
        outputs/{experiment_id}/ klasöründeki metrics.json ve *.html grafiklerini okur.
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
        algorithm_name = ""
        if isinstance(req.model, dict):
            algorithm_name = req.model.get("name") or req.model.get("algorithm") or ""
        elif req.model is not None:
            algorithm_name = getattr(req.model, "name", "") or getattr(req.model, "algorithm", "")

        return ExperimentResultResponse(
            experiment_id=experiment_id,
            task=req.task,
            status="completed",
            algorithm=algorithm_name,
            has_ground_truth=True,
            metrics=MetricsData(**metrics_dict) if metrics_dict else MetricsData(),
            visualizations=visualizations
        )

    def list_artifacts(self) -> List[ArtifactMetadata]:
        """
        Eğitilmiş model dosyalarını listeler.
        """
        artifacts = []
        for exp_id, data in self._in_memory_status.items():
            if data["status"] == "completed":
                req_obj = data["request"]
                algo_name = ""
                if isinstance(req_obj.model, dict):
                    algo_name = req_obj.model.get("name") or req_obj.model.get("algorithm") or ""
                elif req_obj.model is not None:
                    algo_name = getattr(req_obj.model, "name", "") or getattr(req_obj.model, "algorithm", "")
                artifacts.append(
                    ArtifactMetadata(
                        id=f"model_{exp_id}",
                        name=f"{algo_name} Modeli ({exp_id})",
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
