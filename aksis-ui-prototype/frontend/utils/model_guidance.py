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

