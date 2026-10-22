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


