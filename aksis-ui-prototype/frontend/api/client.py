import requests
import os
from typing import Dict, Any, List

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

class AksisAPIError(Exception):
    pass

class AksisClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code >= 400:
                try:
                    error_detail = resp.json().get("detail", resp.text)
                except ValueError:
                    error_detail = resp.text
                raise AksisAPIError(f"API Error ({resp.status_code}): {error_detail}")
            
            # Not all endpoints return JSON
            if resp.content:
                return resp.json()
            return None
        except requests.RequestException as e:
            raise AksisAPIError(f"Connection failed: {str(e)}")

    def get_health(self) -> bool:
        try:
            health_url = self.base_url.replace("/api/v1", "/health")
            resp = requests.get(health_url, timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        return self._request("GET", "/capabilities")

    def get_datasets(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/datasets")
        
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/datasets/{dataset_id}")
        
    def create_experiment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/experiments", json=payload)
        
    def list_experiments(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/experiments")
        
    def run_experiment(self, experiment_id: str):
        return self._request("POST", f"/experiments/{experiment_id}/run")
        
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/experiments/{experiment_id}")
        
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/experiments/{experiment_id}/results")
        
    def list_artifacts(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/artifacts")
        
    def run_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/inference", json=payload)
