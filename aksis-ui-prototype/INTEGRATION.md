# AKSIS Integration Checklist

This document details exactly what must be connected on the company computer to make the prototype work with the real AKSIS framework.
Search the codebase for `AKSIS_INTEGRATION_POINT` to locate these placeholders.
Note: Phase 2 extends the UI to cover Artifacts and Inference, the placeholders for these are also included.

All markers are located in `backend/services/aksis_service.py`.

## ☑ get_capabilities
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.get_capabilities`
- **Purpose:** Provide the UI with supported learning types, tasks, algorithms, tuning options, etc.
- **Expected Output Schema:** `CapabilityResponse`
- **Likely AKSIS component:** Configuration registries (e.g. model registry, task registry).
- **Pseudo-code:**
  ```python
  algorithms = aksis.registry.get_supported_algorithms()
  return CapabilityResponse(algorithms=algorithms, ...)
  ```

## ☑ list_datasets
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.list_datasets`
- **Purpose:** Fetch a list of datasets available for training.
- **Expected Output Schema:** `List[DatasetMetadata]`
- **Likely AKSIS component:** Dataset registry / DataSpec catalog.
- **Pseudo-code:**
  ```python
  datasets = aksis.data.catalog.list()
  return [DatasetMetadata(id=d.id, name=d.name, ...) for d in datasets]
  ```

## ☑ get_dataset
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.get_dataset`
- **Purpose:** Fetch metadata for a specific dataset.
- **Expected Output Schema:** `DatasetMetadata`
- **Likely AKSIS component:** Dataset registry / DataSpec.

## ☑ create_experiment
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.create_experiment`
- **Purpose:** Translate the API `ExperimentCreateRequest` into the real AKSIS `ExperimentConfig`.
- **Input Schema:** `ExperimentCreateRequest`
- **Expected Output Schema:** `ExperimentMetadata`
- **Likely AKSIS component:** `ExperimentConfig` factory.
- **Pseudo-code:**
  ```python
  aksis_config = ExperimentConfig(
      dataset_id=req.dataset_id,
      algorithm=req.model.algorithm,
      # mapping overrides etc...
  )
  exp_id = aksis.store.save_config(aksis_config)
  return ExperimentMetadata(id=exp_id, status="configured")
  ```

## ☑ run_experiment
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.run_experiment`
- **Purpose:** Fire the existing AKSIS execution pipeline. Must return immediately (non-blocking).
- **Input Schema:** `experiment_id`
- **Likely AKSIS component:** `run_experiment()` runner.
- **Pseudo-code:**
  ```python
  config = aksis.store.get_config(experiment_id)
  # Launch asynchronously (e.g. via thread, local worker, or job submission)
  aksis.runner.run_experiment_async(config)
  ```

## ☑ get_experiment_results
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.get_experiment_results`
- **Purpose:** Fetch typed ML evaluation outputs from disk (`outputs/{experiment_id}/`). Reads `metrics.json` and scans all interactive Plotly `*.html` files.
- **Input Schema:** `experiment_id`
- **Expected Output Schema:** `ExperimentResultResponse`
- **Likely AKSIS component:** Output directory scanner.
- **Pseudo-code:**
  ```python
  output_dir = os.path.join("outputs", experiment_id)
  
  # 1. Metrikleri JSON'dan oku
  with open(os.path.join(output_dir, "metrics.json")) as f:
      metrics = json.load(f)
      
  # 2. *.html Plotly grafiklerini tara ve oku (UI'da interaktif açılır)
  visualizations = []
  for fname in os.listdir(output_dir):
      if fname.endswith(".html"):
          with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
              visualizations.append(VisualizationData(
                  type=fname.replace(".html", ""),
                  title=fname.replace(".html", "").replace("_", " ").title(),
                  html_content=f.read()
              ))
              
  return ExperimentResultResponse(
      experiment_id=experiment_id,
      status="completed",
      metrics=MetricsData(**metrics),
      visualizations=visualizations
  )
  ```

## ☑ list_artifacts (Optional)
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.list_artifacts`
- **Purpose:** Expose trained models and experiment outputs (minimal implementation for inference).
- **Expected Output Schema:** `List[ArtifactMetadata]`
- **Likely AKSIS component:** However models/outputs are stored (e.g. output directories).

## ☑ run_inference (Optional)
- **File Path:** `backend/services/aksis_service.py`
- **Class/Function:** `RealAksisService.run_inference`
- **Purpose:** Run batch inference on a dataset using an existing model artifact.
- **Input Schema:** `InferenceRequest`
- **Expected Output Schema:** `InferenceResponse`
- **Likely AKSIS component:** AKSIS inference pipeline.
