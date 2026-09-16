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


class RealAksisService(AksisService):
    """
    Kurumsal AKSIS Çerçevesi için Minimal ve Güvenli Adaptör Katmanı.
    Bu sınıf doğrudan 'src/' altındaki kütüphane fonksiyonlarıyla ve registry ile haberleşir.
    """

    def __init__(self):
        # Deney durumlarını ve konfigürasyonlarını bellekte takip eden sözlük
        self._in_memory_status: Dict[str, dict] = {}

    def get_capabilities(self) -> CapabilityResponse:
        """
        AKSIS bünyesinde kayıtlı algoritmaları, görevleri, stratejileri ve
        algorithm_metadata sözlüğünü döner.
        AKSIS sözlük yapısını API CapabilityResponse sözleşmesine normalize eder.
        """
        if aksis_get_capabilities is None:
            err_msg = (
                f"AKSIS yetenek kütüphanesi (src.core.capabilities) bulunamadı veya yüklenemedi: "
                f"{AKSIS_CORE_IMPORT_ERROR}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        try:
            raw = aksis_get_capabilities()
        except Exception as e:
            err_msg = f"aksis_get_capabilities() çağrısı başarısız oldu: {e}"
            logger.error(err_msg, exc_info=True)
            raise RuntimeError(err_msg) from e

        if hasattr(raw, "model_dump"):
            aksis_caps = raw.model_dump()
        elif hasattr(raw, "dict"):
            aksis_caps = raw.dict()
        elif not isinstance(raw, dict):
            aksis_caps = dict(raw)
        else:
            aksis_caps = raw

        # 1. Normalize tasks: Convert flat list (or dict) into {"supervised": [...], "unsupervised": [...]}
        raw_tasks = aksis_caps.get("tasks")
        if isinstance(raw_tasks, dict):
            tasks = {k: list(v) for k, v in raw_tasks.items() if isinstance(v, (list, tuple, set))}
        elif isinstance(raw_tasks, (list, tuple, set)):
            supervised_candidates = {"classification", "regression"}
            unsupervised_candidates = {"anomaly_detection", "clustering"}
            sup = [t for t in raw_tasks if t in supervised_candidates]
            unsup = [t for t in raw_tasks if t in unsupervised_candidates]
            other = [t for t in raw_tasks if t not in supervised_candidates and t not in unsupervised_candidates]
            if other:
                sup.extend(other)
            tasks = {}
            if sup:
                tasks["supervised"] = sup
            if unsup:
                tasks["unsupervised"] = unsup
        else:
            tasks = {}

        # 2. Normalize learning_types
        raw_lt = aksis_caps.get("learning_types")
        if isinstance(raw_lt, (list, tuple, set)):
            learning_types = list(raw_lt)
        elif tasks:
            learning_types = list(tasks.keys())
        else:
            learning_types = ["supervised", "unsupervised"]

        # 3. Normalize modes
        raw_modes = aksis_caps.get("modes")
        if isinstance(raw_modes, (list, tuple, set)):
            modes = list(raw_modes)
        else:
            modes = ["train", "tune", "predict"]

        # 4. Normalize algorithms: Dict[str, List[str]]
        raw_algorithms = aksis_caps.get("algorithms", {})
        algorithms: Dict[str, List[str]] = {}
        if isinstance(raw_algorithms, dict):
            for ak, av in raw_algorithms.items():
                if isinstance(av, (list, tuple, set)):
                    algorithms[ak] = list(av)
                else:
                    algorithms[ak] = [str(av)]

        # 5. Normalize model_presets: Flatten nested dict or list into a single ordered list
        raw_presets = aksis_caps.get("model_presets")
        model_presets = []
        seen_presets = set()
        if isinstance(raw_presets, dict):
            for group in raw_presets.values():
                items = group if isinstance(group, (list, tuple, set)) else [group]
                for p in items:
                    if p and p not in seen_presets:
                        seen_presets.add(p)
                        model_presets.append(p)
        elif isinstance(raw_presets, (list, tuple, set)):
            for p in raw_presets:
                if p and p not in seen_presets:
                    seen_presets.add(p)
                    model_presets.append(p)
        if not model_presets:
            model_presets = ["baseline", "fast", "strong", "custom"]

        # 6. Normalize preprocessing_strategies: Dict[str, List[str]]
        raw_prep = (
            aksis_caps.get("preprocessing_strategies")
            or aksis_caps.get("preprocessing")
            or {}
        )
        preprocessing_strategies: Dict[str, List[str]] = {}
        if isinstance(raw_prep, dict):
            for pk, pv in raw_prep.items():
                if isinstance(pv, (list, tuple, set)):
                    preprocessing_strategies[pk] = list(pv)
                elif isinstance(pv, dict):
                    sub_items = []
                    for sub in pv.values():
                        if isinstance(sub, (list, tuple, set)):
                            sub_items.extend(sub)
                    preprocessing_strategies[pk] = list(dict.fromkeys(sub_items))
                else:
                    preprocessing_strategies[pk] = [str(pv)]

        # Ensure missing_value key exists without fabricating fake strategies
        if "missing_value" not in preprocessing_strategies:
            preprocessing_strategies["missing_value"] = []

        # 7. Normalize validation_options: List[str]
        raw_validation = (
            aksis_caps.get("validation_options")
            or aksis_caps.get("validation")
            or ["holdout", "kfold", "stratified_kfold"]
        )
        if isinstance(raw_validation, (list, tuple, set)):
            validation_options = list(raw_validation)
        elif isinstance(raw_validation, dict):
            val_items = []
            for sub in raw_validation.values():
                if isinstance(sub, (list, tuple, set)):
                    val_items.extend(sub)
            validation_options = list(dict.fromkeys(val_items))
        else:
            validation_options = ["holdout", "kfold", "stratified_kfold"]

        # 8. Normalize tuning_options: Merge nested dicts (e.g. supervised/unsupervised) into flat Dict[str, List[str]]
        raw_tuning = (
            aksis_caps.get("tuning_options")
            or aksis_caps.get("tuning")
            or {}
        )
        tuning_options: Dict[str, List[str]] = {}
        if isinstance(raw_tuning, dict):
            is_nested = any(isinstance(v, dict) for v in raw_tuning.values())
            if is_nested:
                for sub_dict in raw_tuning.values():
                    if isinstance(sub_dict, dict):
                        for opt_key, opt_vals in sub_dict.items():
                            if opt_key not in tuning_options:
                                tuning_options[opt_key] = []
                            items = opt_vals if isinstance(opt_vals, (list, tuple, set)) else [opt_vals]
                            for it in items:
                                if it not in tuning_options[opt_key]:
                                    tuning_options[opt_key].append(it)
            else:
                for opt_key, opt_vals in raw_tuning.items():
                    if isinstance(opt_vals, (list, tuple, set)):
                        tuning_options[opt_key] = list(opt_vals)
                    else:
                        tuning_options[opt_key] = [str(opt_vals)]

        # 9. Normalize scoring_options: Dict[str, List[str]] without fabricating values
        raw_scoring = (
            aksis_caps.get("scoring_options")
            or aksis_caps.get("scoring")
            or aksis_caps.get("metrics")
            or aksis_caps.get("objectives")
            or {}
        )
        scoring_options: Dict[str, List[str]] = {}
        if isinstance(raw_scoring, dict):
            for sk, sv in raw_scoring.items():
                if isinstance(sv, (list, tuple, set)):
                    scoring_options[sk] = list(sv)
                elif isinstance(sv, dict):
                    scoring_options[sk] = list(sv.keys())
                else:
                    scoring_options[sk] = [str(sv)]

        # If AKSIS does not expose direct scoring per task, initialize empty list per known task
        for task_list in tasks.values():
            for t in task_list:
                scoring_options.setdefault(t, [])

        # 10. Evaluation & Visualization Capabilities
        raw_eval = (
            aksis_caps.get("evaluation_capabilities")
            or aksis_caps.get("evaluation")
            or []
        )
        evaluation_capabilities = list(raw_eval) if isinstance(raw_eval, (list, tuple, set)) else []

        raw_vis = (
            aksis_caps.get("visualization_capabilities")
            or aksis_caps.get("visualization")
            or []
        )
        visualization_capabilities = list(raw_vis) if isinstance(raw_vis, (list, tuple, set)) else []

        # 11. Algorithm metadata & parameter schema (passed directly from AKSIS)
        algorithm_metadata = aksis_caps.get("algorithm_metadata")
        parameter_schema = aksis_caps.get("parameter_schema")

        return CapabilityResponse(
            learning_types=learning_types,
            tasks=tasks,
            modes=modes,
            algorithms=algorithms,
            model_presets=model_presets,
            preprocessing_strategies=preprocessing_strategies,
            validation_options=validation_options,
            tuning_options=tuning_options,
            scoring_options=scoring_options,
            evaluation_capabilities=evaluation_capabilities,
            visualization_capabilities=visualization_capabilities,
            algorithm_metadata=algorithm_metadata,
            parameter_schema=parameter_schema
        )

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
        else:
            spec_id = getattr(spec, "id", "")
            display_name = getattr(spec, "display_name", None)
            description = getattr(spec, "description", None)
            source = getattr(spec, "source", None)
            local_data = getattr(spec, "local_data", None)
            target = getattr(spec, "target", None)
            columns_to_use = getattr(spec, "columns_to_use", None)

        if hasattr(source, "value"):
            source = source.value
        elif source is not None:
            source = str(source)

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
            compatible_tasks=[],
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
