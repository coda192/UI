import requests
import os
from typing import Dict, Any, List

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

class AksisClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        
    def get_health(self) -> bool:
        try:
            # Health check is at the root
            health_url = self.base_url.replace("/api/v1", "/health")
            resp = requests.get(health_url, timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/capabilities")
        resp.raise_for_status()
        return resp.json()

    def get_datasets(self) -> List[Dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/datasets")
        resp.raise_for_status()
        return resp.json()
        
    def create_experiment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/experiments", json=payload)
        resp.raise_for_status()
        return resp.json()
        
    def run_experiment(self, experiment_id: str):
        resp = requests.post(f"{self.base_url}/experiments/{experiment_id}/run")
        resp.raise_for_status()
        
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/experiments/{experiment_id}")
        resp.raise_for_status()
        return resp.json()
        
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/experiments/{experiment_id}/results")
        resp.raise_for_status()
        return resp.json()
