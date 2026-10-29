"""
AKSIS Platformu - Model Rehberi ve Karar Destek Yardımcıları (API-Driven)
Bu modül sabit model bilgisi barındırmaz; tüm meta verileri API capabilities yanıtından dinamik okur.
"""

from typing import Dict, Any, Optional

def get_algorithm_metadata(algorithm_key: str, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    API capabilities yanıtından verilen algoritmanın meta verilerini döner.
    Eğer meta veri sağlanmamışsa, ham anahtarı ve nötr varsayılan açıklamayı döner.
    """
    fallback = {
        "display_name": algorithm_key or "",
        "description": "Model açıklaması sağlanmamış.",
        "strengths": [],
        "limitations": [],
        "best_for": []
    }
    
    if not algorithm_key or not capabilities or not isinstance(capabilities, dict):
        return fallback
        
    algo_meta_dict = capabilities.get("algorithm_metadata") or {}
    if algorithm_key in algo_meta_dict:
        meta = algo_meta_dict[algorithm_key]
        if hasattr(meta, "model_dump"):
            return meta.model_dump()
        elif isinstance(meta, dict):
            return meta
            
    return fallback

def get_algorithm_display_name(algorithm_key: str, capabilities: Optional[Dict[str, Any]] = None) -> str:
    """
    Algoritmanın API tarafından sağlanan display_name değerini döner.
    Yoksa ham algoritma anahtarını (ID) döner.
    """
    if not algorithm_key:
        return ""
    meta = get_algorithm_metadata(algorithm_key, capabilities)
    return meta.get("display_name") or algorithm_key


def is_config_applicable(
    capabilities: Optional[Dict[str, Any]],
    config_name: str,
    algorithm_type: str,
    task: str,
) -> bool:
    """
    Belirli bir konfigürasyon bloğunun (ör. 'data_split', 'supervised_tuning', 'preprocess'),
    mevcut algorithm_type ve task için AKSIS config_applicability meta verisine göre
    uygulanabilir olup olmadığını doğrular.

    Hem algorithm_types hem de supported_tasks listesinde eşleşme olmalıdır.
    """
    if not capabilities or not isinstance(capabilities, dict):
        return False

    applicability = capabilities.get("config_applicability") or {}
    config_meta = applicability.get(config_name)

    if not config_meta:
        return False

    if hasattr(config_meta, "model_dump"):
        config_meta = config_meta.model_dump()
    elif hasattr(config_meta, "dict"):
        config_meta = config_meta.dict()
    elif not isinstance(config_meta, dict):
        config_meta = dict(config_meta)

    allowed_algo_types = config_meta.get("algorithm_types") or []
    supported_tasks = config_meta.get("supported_tasks") or []

    if algorithm_type not in allowed_algo_types:
        return False

    if task not in supported_tasks:
        return False

    return True


def resolve_task_learning_types(
    capabilities: Optional[Dict[str, Any]],
    candidate_tasks: Optional[list]
) -> Dict[str, str]:
    """
    Veri setinin uyumlu görevleri (candidate_tasks) için hangi learning_type'ların
    uygun olduğunu AKSIS capabilities['config_applicability'] üzerinden dinamik olarak çözümler.
    
    Her bir learning_type için '{learning_type}_model' uygunluk girdisini inceler:
    - task in config_applicability['{lt}_model']['supported_tasks']
    - lt in config_applicability['{lt}_model']['algorithm_types']

    Dönen sözlük: {task: learning_type}
    """
    if not capabilities or not isinstance(capabilities, dict) or not candidate_tasks:
        return {}

    learning_types = capabilities.get("learning_types") or []
    task_to_lt: Dict[str, str] = {}

    for t in candidate_tasks:
        for lt in learning_types:
            config_key = f"{lt}_model"
            if is_config_applicable(capabilities, config_key, lt, t):
                task_to_lt[t] = lt
                break

    return task_to_lt


# ==============================================================================
# CAPABILITY DIAGNOSTICS & VALIDATION UTILITY (STABLE ERROR CODES)
# ==============================================================================

from dataclasses import dataclass
from typing import Sequence, Union


@dataclass(frozen=True)
class CapabilityDiagnostic:
    code: str
    message: str
    path: Sequence[Union[str, int]]
    is_valid: bool = False

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __bool__(self) -> bool:
        return self.is_valid


def _format_path(path: Sequence[Union[str, int]]) -> str:
    if not path:
        return "capabilities"
    res = "capabilities"
    for p in path:
        if isinstance(p, int):
            res += f"[{p}]"
        else:
            res += f'["{p}"]'
    return res


# Stable error code mappings for known paths / categories
KNOWN_ERROR_CODES: Dict[tuple, Dict[str, str]] = {
    # Core
    ("learning_types",): {
        "missing": "CAP-CORE-001",
        "empty": "CAP-CORE-001",
        "invalid": "CAP-CORE-001"
    },
    ("tasks",): {
        "missing": "CAP-CORE-002",
        "empty": "CAP-CORE-002",
        "invalid": "CAP-CORE-002"
    },
    ("modes",): {
        "missing": "CAP-CORE-003",
        "empty": "CAP-CORE-003",
        "invalid": "CAP-CORE-003"
    },
    ("algorithms",): {
        "missing": "CAP-CORE-004",
        "empty": "CAP-CORE-004",
        "invalid": "CAP-CORE-004"
    },
    # Config Applicability
    ("config_applicability",): {
        "missing": "CAP-APP-001",
        "empty": "CAP-APP-001",
        "invalid": "CAP-APP-001"
    },
    # Model
    ("model_presets",): {
        "missing": "CAP-MODEL-001",
        "empty": "CAP-MODEL-001",
        "invalid": "CAP-MODEL-001"
    },
    # Tuning
    ("tuning",): {
        "missing": "CAP-TUNE-001",
        "empty": "CAP-TUNE-001",
        "invalid": "CAP-TUNE-001"
    },
    ("tuning", "samplers"): {
        "missing": "CAP-TUNE-002",
        "empty": "CAP-TUNE-002",
        "invalid": "CAP-TUNE-002"
    },
    ("tuning", "pruners"): {
        "missing": "CAP-TUNE-003",
        "empty": "CAP-TUNE-003",
        "invalid": "CAP-TUNE-003"
    },
    # Preprocessing
    ("preprocessing",): {
        "missing": "CAP-PREP-001",
        "empty": "CAP-PREP-001",
        "invalid": "CAP-PREP-001"
    },
    # Data Split
    ("data_split",): {
        "missing": "CAP-DS-001",
        "empty": "CAP-DS-001",
        "invalid": "CAP-DS-001"
    },
    ("data_split", "methods"): {
        "missing": "CAP-DS-002",
        "empty": "CAP-DS-002",
        "invalid": "CAP-DS-002"
    },
    ("data_split", "parameters"): {
        "missing": "CAP-DS-003",
        "empty": "CAP-DS-003",
        "invalid": "CAP-DS-003"
    },
    # Data Analyze
    ("data_analyze",): {
        "missing": "CAP-AN-001",
        "empty": "CAP-AN-001",
        "invalid": "CAP-AN-001"
    },
    # Evaluation
    ("evaluation",): {
        "missing": "CAP-EVAL-001",
        "empty": "CAP-EVAL-001",
        "invalid": "CAP-EVAL-001"
    },
    # Visualization
    ("visualization",): {
        "missing": "CAP-VIS-001",
        "empty": "CAP-VIS-001",
        "invalid": "CAP-VIS-001"
    },
}


def _infer_error_code(path: Sequence[Union[str, int]], reason: str, custom_code: Optional[str] = None) -> str:
    if custom_code:
        return custom_code

    tuple_path = tuple(path)
    if tuple_path in KNOWN_ERROR_CODES:
        return KNOWN_ERROR_CODES[tuple_path].get(reason, "CAP-GEN-001")

    # Match based on top-level root
    if path:
        root = str(path[0])
        if root == "data_split":
            return "CAP-DS-001"
        elif root == "tuning":
            return "CAP-TUNE-001"
        elif root == "model_presets":
            return "CAP-MODEL-001"
        elif root == "preprocessing":
            return "CAP-PREP-001"
        elif root == "config_applicability":
            return "CAP-APP-001"
        elif root == "data_analyze":
            return "CAP-AN-001"
        elif root == "evaluation":
            return "CAP-EVAL-001"
        elif root == "visualization":
            return "CAP-VIS-001"

    return "CAP-GEN-001"


def validate_capability_path(
    capabilities: Optional[Dict[str, Any]],
    path: Sequence[Union[str, int]],
    required: bool = True,
    allow_empty: bool = False,
    expected_type: Optional[type] = None,
    error_code: Optional[str] = None,
) -> CapabilityDiagnostic:
    """
    capabilities sözlüğünde verilen yolun (path) varlığını, tipini ve boşluk durumunu doğrular.
    Geçerliyse is_valid=True olan bir CapabilityDiagnostic döner.
    Geçersizse stabil bir error code ve okunabilir hata mesajı döner.
    """
    path_seq = tuple(path)
    path_str = _format_path(path_seq)

    if capabilities is None or not isinstance(capabilities, dict):
        code = error_code or _infer_error_code(path_seq, "missing")
        return CapabilityDiagnostic(
            code=code,
            message=f"Capabilities payload is missing or invalid: {path_str} cannot be resolved.",
            path=path_seq,
            is_valid=False
        )

    curr: Any = capabilities
    for idx, segment in enumerate(path_seq):
        # Dict traversal
        if isinstance(curr, dict):
            if segment not in curr:
                if not required:
                    return CapabilityDiagnostic(
                        code="CAP-OK",
                        message=f"Optional path {path_str} is absent.",
                        path=path_seq,
                        is_valid=True
                    )
                code = error_code or _infer_error_code(path_seq[:idx + 1], "missing")
                return CapabilityDiagnostic(
                    code=code,
                    message=f"Capability section is missing: {path_str}.",
                    path=path_seq,
                    is_valid=False
                )
            curr = curr[segment]
        # Object traversal with model_dump or attributes
        elif hasattr(curr, "__dict__") or hasattr(curr, "model_dump"):
            dumped = curr.model_dump() if hasattr(curr, "model_dump") else (curr.dict() if hasattr(curr, "dict") else dict(curr))
            if segment not in dumped:
                if not required:
                    return CapabilityDiagnostic(
                        code="CAP-OK",
                        message=f"Optional path {path_str} is absent.",
                        path=path_seq,
                        is_valid=True
                    )
                code = error_code or _infer_error_code(path_seq[:idx + 1], "missing")
                return CapabilityDiagnostic(
                    code=code,
                    message=f"Capability section is missing: {path_str}.",
                    path=path_seq,
                    is_valid=False
                )
            curr = dumped[segment]
        elif isinstance(curr, (list, tuple)):
            if isinstance(segment, int) and 0 <= segment < len(curr):
                curr = curr[segment]
            else:
                code = error_code or _infer_error_code(path_seq[:idx + 1], "missing")
                return CapabilityDiagnostic(
                    code=code,
                    message=f"Index {segment} out of range in {path_str}.",
                    path=path_seq,
                    is_valid=False
                )
        else:
            code = error_code or _infer_error_code(path_seq[:idx + 1], "invalid")
            return CapabilityDiagnostic(
                code=code,
                message=f"Cannot traverse into non-container type {type(curr).__name__} at {path_str}.",
                path=path_seq,
                is_valid=False
            )

    # Type check if specified
    if expected_type is not None and not isinstance(curr, expected_type):
        code = error_code or _infer_error_code(path_seq, "invalid")
        return CapabilityDiagnostic(
            code=code,
            message=f"Capability path {path_str} expected type {expected_type.__name__}, got {type(curr).__name__}.",
            path=path_seq,
            is_valid=False
        )

    # Empty check
    if not allow_empty:
        is_empty = False
        if curr is None:
            is_empty = True
        elif isinstance(curr, (list, dict, tuple, set, str)) and len(curr) == 0:
            is_empty = True

        if is_empty:
            code = error_code or _infer_error_code(path_seq, "empty")
            return CapabilityDiagnostic(
                code=code,
                message=f"Capability is missing or empty: {path_str}.",
                path=path_seq,
                is_valid=False
            )

    return CapabilityDiagnostic(
        code="CAP-OK",
        message=f"{path_str} is valid.",
        path=path_seq,
        is_valid=True
    )


def diagnose_config_applicability(
    capabilities: Optional[Dict[str, Any]],
    config_name: str
) -> CapabilityDiagnostic:
    """
    Belirli bir config_applicability girdisini detaylı doğrular:
    - CAP-APP-001: config_applicability eksik/geçersiz
    - CAP-APP-002: İstenen config_applicability girdisi eksik
    - CAP-APP-003: algorithm_types eksik veya boş
    - CAP-APP-004: supported_tasks eksik veya boş
    """
    root_diag = validate_capability_path(
        capabilities,
        path=("config_applicability",),
        required=True,
        allow_empty=False,
        error_code="CAP-APP-001"
    )
    if not root_diag.is_valid:
        return root_diag

    app_dict = capabilities.get("config_applicability", {})
    if config_name not in app_dict:
        return CapabilityDiagnostic(
            code="CAP-APP-002",
            message=f"Requested config applicability entry is missing: capabilities[\"config_applicability\"][\"{config_name}\"].",
            path=("config_applicability", config_name),
            is_valid=False
        )

    entry = app_dict[config_name]
    if hasattr(entry, "model_dump"):
        entry = entry.model_dump()
    elif hasattr(entry, "dict"):
        entry = entry.dict()
    elif not isinstance(entry, dict):
        entry = dict(entry)

    algo_types = entry.get("algorithm_types")
    if not algo_types:
        return CapabilityDiagnostic(
            code="CAP-APP-003",
            message=f"algorithm_types missing or empty for applicability entry '{config_name}'.",
            path=("config_applicability", config_name, "algorithm_types"),
            is_valid=False
        )

    tasks = entry.get("supported_tasks")
    if not tasks:
        return CapabilityDiagnostic(
            code="CAP-APP-004",
            message=f"supported_tasks missing or empty for applicability entry '{config_name}'.",
            path=("config_applicability", config_name, "supported_tasks"),
            is_valid=False
        )

    return CapabilityDiagnostic(
        code="CAP-OK",
        message=f"Applicability entry '{config_name}' is valid.",
        path=("config_applicability", config_name),
        is_valid=True
    )


def diagnose_model_presets(
    capabilities: Optional[Dict[str, Any]],
    learning_type: Optional[str] = None
) -> CapabilityDiagnostic:
    """
    Model presets yapısını doğrular:
    - CAP-MODEL-001: model_presets eksik veya boş
    - CAP-MODEL-002: Verilen learning_type için model presets eksik veya boş
    """
    root_diag = validate_capability_path(
        capabilities,
        path=("model_presets",),
        required=True,
        allow_empty=False,
        error_code="CAP-MODEL-001"
    )
    if not root_diag.is_valid:
        return root_diag

    if learning_type:
        presets_dict = capabilities.get("model_presets", {})
        if isinstance(presets_dict, dict):
            lt_presets = presets_dict.get(learning_type)
            if not lt_presets:
                return CapabilityDiagnostic(
                    code="CAP-MODEL-002",
                    message=f"Model presets missing or empty for selected learning_type '{learning_type}': capabilities[\"model_presets\"][\"{learning_type}\"].",
                    path=("model_presets", learning_type),
                    is_valid=False
                )
        elif isinstance(presets_dict, list):
            if not presets_dict:
                return CapabilityDiagnostic(
                    code="CAP-MODEL-002",
                    message="Model presets list is empty.",
                    path=("model_presets",),
                    is_valid=False
                )

    return CapabilityDiagnostic(
        code="CAP-OK",
        message="Model presets are valid.",
        path=("model_presets",) if not learning_type else ("model_presets", learning_type),
        is_valid=True
    )



