import os
from backend.services.factory import get_aksis_service
from backend.services.mock_service import MockAksisService
from backend.services.aksis_service import RealAksisService
from backend.schemas import ExperimentCreateRequest

def test_service_factory_mock():
    os.environ["AKSIS_PROVIDER"] = "mock"
    service = get_aksis_service()
    assert isinstance(service, MockAksisService)

def test_service_factory_aksis():
    os.environ["AKSIS_PROVIDER"] = "aksis"
    service = get_aksis_service()
    assert isinstance(service, RealAksisService)

def test_mock_service_flow():
    service = MockAksisService()
    
    # Create
    req = ExperimentCreateRequest(
        project_name="Test",
        dataset_id="ds_class_01",
        algorithm_type="supervised",
        task="classification",
        model={"name": "CatBoost"}
    )
    meta = service.create_experiment(req)
    assert meta.status == "configured"
    assert meta.name == "Test"
    
    # Results should fail if not completed
    try:
        service.get_experiment_results(meta.id)
        assert False, "Should have raised exception"
    except ValueError:
        pass
        
    # (Running is async, so we'd need to mock sleep for a full test, but creation works)

def test_mock_service_anomaly_detection_results():
    service = MockAksisService()
    
    # Test Unlabeled Anomaly
    req_unlabeled = ExperimentCreateRequest(
        project_name="Unlabeled Anomaly Test",
        dataset_id="ds_anom_unlabeled_01",
        algorithm_type="unsupervised",
        task="anomaly_detection",
        model={"name": "Isolation Forest"}
    )
    meta_unlabeled = service.create_experiment(req_unlabeled)
    assert meta_unlabeled.name == "Unlabeled Anomaly Test"
    # Directly complete metadata for test assertions
    meta_unlabeled.status = "completed"
    
    res_unlabeled = service.get_experiment_results(meta_unlabeled.id)
    assert res_unlabeled.has_ground_truth is False
    assert res_unlabeled.metrics.classification_metrics is None
    assert res_unlabeled.metrics.anomaly_metrics == {}
    assert "top_anomalies" in res_unlabeled.tables
    
    # Test Labeled Anomaly
    req_labeled = ExperimentCreateRequest(
        project_name="Labeled Anomaly Test",
        dataset_id="ds_anom_labeled_01",
        algorithm_type="unsupervised",
        task="anomaly_detection",
        model={"name": "Isolation Forest"}
    )
    meta_labeled = service.create_experiment(req_labeled)
    assert meta_labeled.name == "Labeled Anomaly Test"
    meta_labeled.status = "completed"

def test_real_aksis_service_execution_without_src_fails():
    import time
    service = RealAksisService()
    req = ExperimentCreateRequest(
        project_name="Real Execution Test",
        dataset_id="ds_class_01",
        algorithm_type="supervised",
        task="classification",
        model={"name": "xgb_c"}
    )
    meta = service.create_experiment(req)
    assert meta.status == "configured"
    assert meta.name == "Real Execution Test"

    # Attempt execution without AKSIS src available
    service.run_experiment(meta.id)
    time.sleep(0.1)

    exp_meta = service.get_experiment(meta.id)
    assert exp_meta.status == "failed"
    assert "AKSIS experiment runner is unavailable" in service._in_memory_status[meta.id]["error"]

    
def test_real_aksis_service_dataspec_mapping_and_security():
    from backend.services.aksis_service import RealAksisService
    from types import SimpleNamespace
    
    # Simulate a raw DataSpec object with sensitive backend fields
    raw_spec = SimpleNamespace(
        id="credit_card_fraud",
        display_name="Kredi Kartı Dolandırıcılığı",
        description="Gerçek zamanlı fraud tespiti veri seti",
        source="oracle_db",
        local_data=True,
        target="Class",
        columns_to_use=["Time", "Amount", "V1", "V2"],
        compatible_tasks=["anomaly_detection"],
        # Sensitive fields that MUST be excluded:
        host="192.168.1.100",
        port=1521,
        user="db_admin",
        password="SuperSecretPassword123!",
        catalog="FINANCE",
        schema="TRANSACTIONS",
        http_schema="https",
        data_query="SELECT * FROM secret_transactions_table"
    )
    
    service = RealAksisService()
    metadata = service._map_dataspec_to_metadata(raw_spec)
    
    # Verify UI-safe fields
    assert metadata.id == "credit_card_fraud"
    assert metadata.display_name == "Kredi Kartı Dolandırıcılığı"
    assert metadata.description == "Gerçek zamanlı fraud tespiti veri seti"
    assert metadata.source == "oracle_db"
    assert metadata.local_data is True
    assert metadata.target == "Class"
    assert metadata.columns_to_use == ["Time", "Amount", "V1", "V2"]
    assert metadata.compatible_tasks == ["anomaly_detection"]
    
    # Verify non-DataSpec fields are set to empty/None
    assert metadata.row_count is None
    assert metadata.column_count is None
    assert metadata.columns == []
    assert metadata.identifier_columns == []
    
    # Verify sensitive fields are NOT in the schema
    meta_dict = metadata.model_dump()
    for sensitive in ["host", "port", "user", "password", "catalog", "schema", "http_schema", "data_query"]:
        assert sensitive not in meta_dict, f"Sensitive field '{sensitive}' must not be exposed!"

def test_real_aksis_service_dataset_registry_accessors(monkeypatch):
    import backend.services.aksis_service as aksis_mod
    from backend.services.aksis_service import RealAksisService
    from types import SimpleNamespace
    
    mock_spec1 = SimpleNamespace(
        id="reg_housing",
        display_name="Konut Fiyatları",
        description="Ev fiyat tahmini",
        source="csv_file",
        local_data=True,
        target="SalePrice",
        columns_to_use=["LotArea", "YearBuilt"]
    )
    mock_spec2 = SimpleNamespace(
        id="cls_churn",
        display_name=None, # Missing display_name fallback test
        description=None,  # Missing description test
        source="db",
        local_data=False,
        target="Churn",
        columns_to_use=None
    )
    
    test_datasets = [mock_spec1, mock_spec2]
    mock_dict = {s.id: s for s in test_datasets}

    monkeypatch.setattr(
        aksis_mod,
        "_get_aksis_dataset_accessors",
        lambda: (lambda: test_datasets, lambda ds_id: mock_dict.get(ds_id))
    )
    
    service = RealAksisService()
    datasets = service.list_datasets()
    
    assert len(datasets) == 2
    ds_ids = [d.id for d in datasets]
    assert "reg_housing" in ds_ids
    assert "cls_churn" in ds_ids
    
    # Verify fallback for missing display_name
    churn_meta = service.get_dataset("cls_churn")
    assert churn_meta.id == "cls_churn"
    assert churn_meta.name == "cls_churn"  # fallback to id
    assert churn_meta.display_name is None
    assert churn_meta.description is None
    
    # Verify retrieval of reg_housing
    housing_meta = service.get_dataset("reg_housing")
    assert housing_meta.id == "reg_housing"
    assert housing_meta.display_name == "Konut Fiyatları"
    assert housing_meta.target == "SalePrice"

    # Verify not found raises ValueError
    try:
        service.get_dataset("non_existent")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_real_aksis_service_datasets_missing_lib():
    import pytest
    from backend.services.aksis_service import RealAksisService

    service = RealAksisService()
    # In this environment without AKSIS src, calling accessors raises RuntimeError
    with pytest.raises(RuntimeError, match="AKSIS dataset registry is unavailable"):
        service.list_datasets()

    with pytest.raises(RuntimeError, match="AKSIS dataset registry is unavailable"):
        service.get_dataset("reg_housing")


def test_real_aksis_service_get_capabilities_mapping(monkeypatch):
    import backend.services.aksis_service as aksis_mod
    from backend.services.aksis_service import RealAksisService

    mock_caps = {
        "learning_types": ["supervised", "unsupervised"],
        "tasks": ["classification", "regression"],
        "modes": ["train", "tune"],
        "config_applicability": {
            "supervised_model": {
                "algorithm_types": ["supervised"],
                "supported_tasks": ["classification", "regression"]
            }
        },
        "algorithms": {
            "classification": ["logreg", "random_forest_c"],
            "regression": ["ridge"]
        },
        "model_presets": {
            "supervised": ["baseline", "fast"]
        },
        "preprocessing": {
            "missing_value": ["mean", "median"],
            "encoding": ["onehot"]
        },
        "tuning": {
            "samplers": {"supervised": ["tpe"]},
            "pruners": {"supervised": ["none"]}
        },
        "algorithm_metadata": {
            "logreg": {
                "display_name": "Lojistik Regresyon",
                "description": "Doğrusal sınıflandırma modeli",
                "strengths": ["Hızlı", "Açıklanabilir"],
                "limitations": ["Doğrusal olmayan ilişkileri yakalayamaz"],
                "best_for": ["Baseline modeller"]
            }
        },
        "parameter_schema": {
            "logreg": {
                "C": {"type": "float", "default": 1.0}
            }
        }
    }

    monkeypatch.setattr(aksis_mod, "aksis_get_capabilities", lambda: mock_caps)

    service = RealAksisService()
    res = service.get_capabilities()

    assert res.learning_types == ["supervised", "unsupervised"]
    assert res.tasks == ["classification", "regression"]
    assert res.algorithms["classification"] == ["logreg", "random_forest_c"]
    assert res.model_presets["supervised"] == ["baseline", "fast"]
    assert res.preprocessing["missing_value"] == ["mean", "median"]
    assert res.tuning.samplers["supervised"] == ["tpe"]
    assert res.algorithm_metadata is not None
    assert "logreg" in res.algorithm_metadata
    assert res.algorithm_metadata["logreg"].display_name == "Lojistik Regresyon"
    assert res.algorithm_metadata["logreg"].strengths == ["Hızlı", "Açıklanabilir"]
    assert res.parameter_schema["logreg"]["C"]["default"] == 1.0


def test_real_aksis_service_get_capabilities_missing_lib(monkeypatch):
    import pytest
    import backend.services.aksis_service as aksis_mod
    from backend.services.aksis_service import RealAksisService

    monkeypatch.setattr(aksis_mod, "aksis_get_capabilities", None)

    service = RealAksisService()
    with pytest.raises(RuntimeError, match="AKSIS.*kütüphanesi"):
        service.get_capabilities()


def test_real_aksis_service_get_capabilities_real_shape_normalization(monkeypatch):
    """
    Test direct loading of AKSIS capability structures into CapabilityResponse:
    - Tasks is a flat list
    - Dict model_presets by learning_type
    - Preprocessing dict
    - Nested tuning (samplers, pruners, etc.)
    - Algorithm metadata preserved
    """
    import backend.services.aksis_service as aksis_mod
    from backend.services.aksis_service import RealAksisService

    real_aksis_raw_caps = {
        "learning_types": ["supervised", "unsupervised"],
        "tasks": ["classification", "regression", "anomaly_detection"],  # Flat list from AKSIS
        "modes": ["train", "tune", "predict"],
        "config_applicability": {
            "supervised_model": {
                "algorithm_types": ["supervised"],
                "supported_tasks": ["classification", "regression"]
            },
            "unsupervised_model": {
                "algorithm_types": ["unsupervised"],
                "supported_tasks": ["anomaly_detection"]
            }
        },
        "algorithms": {
            "classification": ["logreg", "random_forest_c"],
            "regression": ["ridge"],
            "anomaly_detection": ["isolation_forest"]
        },
        "model_presets": {
            "supervised": ["baseline", "fast", "strong"],
            "unsupervised": ["fast", "custom"]
        },
        "preprocessing": {
            "encoding": ["onehot", "ordinal"],
            "scaling": ["standard", "minmax"]
        },
        "tuning": {
            "samplers": {
                "supervised": ["tpe", "random"],
                "unsupervised": ["random"]
            },
            "pruners": {
                "supervised": ["none", "median", "sha"],
                "unsupervised": ["none"]
            }
        },
        "algorithm_metadata": {
            "logreg": {
                "display_name": "Lojistik Regresyon",
                "description": "Doğrusal sınıflandırma",
                "strengths": ["Hızlı"],
                "limitations": ["Doğrusal"],
                "best_for": ["Baseline"]
            }
        }
    }

    monkeypatch.setattr(aksis_mod, "aksis_get_capabilities", lambda: real_aksis_raw_caps)

    service = RealAksisService()
    res = service.get_capabilities()

    # 1. Tasks is flat list
    assert isinstance(res.tasks, list)
    assert res.tasks == ["classification", "regression", "anomaly_detection"]

    # 2. Model presets dict
    assert isinstance(res.model_presets, dict)
    assert res.model_presets["supervised"] == ["baseline", "fast", "strong"]
    assert res.model_presets["unsupervised"] == ["fast", "custom"]

    # 3. Preprocessing preserves raw keys
    assert res.preprocessing["encoding"] == ["onehot", "ordinal"]
    assert res.preprocessing["scaling"] == ["standard", "minmax"]

    # 4. Tuning samplers and pruners
    assert res.tuning.samplers["supervised"] == ["tpe", "random"]
    assert res.tuning.pruners["supervised"] == ["none", "median", "sha"]

    # 6. Algorithm metadata preserved
    assert res.algorithm_metadata is not None
    assert "logreg" in res.algorithm_metadata
    assert res.algorithm_metadata["logreg"].display_name == "Lojistik Regresyon"


def test_mock_service_get_dataset_info():
    import pytest
    from backend.services.mock_service import MockAksisService
    from backend.schemas import DatasetInfoResponse

    service = MockAksisService()
    info = service.get_dataset_info("ds_class_01")
    assert isinstance(info, DatasetInfoResponse)
    assert info.dataset_id == "ds_class_01"
    assert info.row_count > 0
    assert info.column_count > 0
    assert len(info.columns) == info.column_count
    assert info.correlation is not None
    assert len(info.correlation.matrix) == len(info.correlation.columns)

    with pytest.raises(ValueError, match="not found|bulunamadı"):
        service.get_dataset_info("non_existent_dataset")


def test_real_aksis_service_get_dataset_info(monkeypatch):
    import sys
    import pytest
    from unittest.mock import MagicMock
    from backend.services.aksis_service import RealAksisService
    from backend.schemas import DatasetInfoResponse

    service = RealAksisService()

    # 1. When src package is unavailable (ImportError) -> raises RuntimeError
    monkeypatch.delitem(sys.modules, "src.core.data_processing.data_info", raising=False)
    monkeypatch.delitem(sys.modules, "src.core.data_processing", raising=False)
    monkeypatch.delitem(sys.modules, "src.core", raising=False)
    monkeypatch.delitem(sys.modules, "src", raising=False)

    with pytest.raises(RuntimeError, match="requires the AKSIS src package"):
        service.get_dataset_info("test_ds")

    # 2. When src package is available
    call_records = {}

    def fake_get_dataset_info(dataset_id: str, refresh: bool = False):
        call_records["dataset_id"] = dataset_id
        call_records["refresh"] = refresh
        if dataset_id == "missing_id":
            raise KeyError("not found")
        if dataset_id == "none_id":
            return None
        return {
            "dataset_id": dataset_id,
            "row_count": 1000,
            "column_count": 2,
            "missing_value_count": 10,
            "missing_column_count": 1,
            "type_counts": {"numerical": 2},
            "columns": [
                {
                    "name": "feat1",
                    "dtype": "float64",
                    "primitive_type": "numerical",
                    "subtype": "continuous",
                    "unique_count": 800,
                    "missing_count": 0,
                    "missing_rate": 0.0,
                    "flags": ["numerical"],
                },
                {
                    "name": "feat2",
                    "dtype": "int64",
                    "primitive_type": "numerical",
                    "subtype": "integer",
                    "unique_count": 100,
                    "missing_count": 10,
                    "missing_rate": 0.01,
                    "flags": ["has_missing"],
                },
            ],
            "correlation": {
                "columns": ["feat1", "feat2"],
                "matrix": [[1.0, None], [None, 1.0]],
            },
        }

    mock_mod = MagicMock()
    mock_mod.get_dataset_info = fake_get_dataset_info
    monkeypatch.setitem(sys.modules, "src", MagicMock())
    monkeypatch.setitem(sys.modules, "src.core", MagicMock())
    monkeypatch.setitem(sys.modules, "src.core.data_processing", MagicMock())
    monkeypatch.setitem(sys.modules, "src.core.data_processing.data_info", mock_mod)

    # Call with refresh=True
    info = service.get_dataset_info("ds_real", refresh=True)
    assert isinstance(info, DatasetInfoResponse)
    assert call_records == {"dataset_id": "ds_real", "refresh": True}
    assert info.dataset_id == "ds_real"
    assert info.row_count == 1000
    assert info.correlation.matrix[0][1] is None

    # Unknown dataset raises ValueError
    with pytest.raises(ValueError, match="bulunamadı"):
        service.get_dataset_info("missing_id")

    # None return raises ValueError
    with pytest.raises(ValueError, match="bulunamadı"):
        service.get_dataset_info("none_id")


def test_mock_service_get_dataset_info_statistics():
    from backend.services.mock_service import MockAksisService

    service = MockAksisService()
    info = service.get_dataset_info("ds_class_01")

    cols = {c.name: c for c in info.columns}
    assert "Age" in cols
    assert cols["Age"].statistics is not None
    assert cols["Age"].statistics["count"] == 988.0
    assert cols["Age"].statistics["skewness"] == 0.35

    assert "Tenure" in cols
    assert cols["Tenure"].statistics is not None
    assert cols["Tenure"].statistics["50%"] == cols["Tenure"].statistics["75%"] == 3.0
    assert cols["Tenure"].statistics["skewness"] is None

    assert "ConstantCol" in cols
    assert cols["ConstantCol"].statistics is not None
    assert cols["ConstantCol"].statistics["min"] == cols["ConstantCol"].statistics["max"] == 42.0

    assert "Department" in cols
    assert cols["Department"].statistics is None


def test_dataset_info_no_numeric_columns():
    from backend.schemas import DatasetInfoResponse, ColumnInfo, CorrelationInfo

    # Dataset with only categorical / text columns (0 numeric columns)
    info = DatasetInfoResponse(
        dataset_id="ds_categorical_only",
        row_count=200,
        column_count=2,
        missing_value_count=0,
        missing_column_count=0,
        type_counts={"categorical": 2},
        columns=[
            ColumnInfo(
                name="CategoryA",
                dtype="object",
                primitive_type="categorical",
                subtype="nominal",
                unique_count=4,
                missing_count=0,
                missing_rate=0.0,
                flags=[],
                statistics=None
            ),
            ColumnInfo(
                name="CategoryB",
                dtype="object",
                primitive_type="categorical",
                subtype="ordinal",
                unique_count=3,
                missing_count=0,
                missing_rate=0.0,
                flags=[],
                statistics=None
            ),
        ],
        correlation=CorrelationInfo(columns=[], matrix=[])
    )

    numeric_cols = [
        c for c in info.columns
        if c.primitive_type in ["numeric", "numerical"]
    ]
    assert len(numeric_cols) == 0
    assert len(info.columns) == 2


def test_mock_service_top_categories():
    from backend.services.mock_service import MockAksisService

    service = MockAksisService()
    info = service.get_dataset_info("ds_class_01")
    dept = next(c for c in info.columns if c.name == "Department")
    assert dept.top_categories is not None
    assert len(dept.top_categories) == 5
    assert dept.top_categories[0].value == "Mühendislik"
    assert dept.valid_count == 1000
    assert dept.other_count == 50


def test_mock_service_histogram():
    from backend.services.mock_service import MockAksisService
    from backend.schemas import HistogramData

    service = MockAksisService()
    info = service.get_dataset_info("ds_class_01")
    cols = {c.name: c for c in info.columns}

    # Age has histogram
    age = cols["Age"]
    assert isinstance(age.histogram, HistogramData)
    assert len(age.histogram.counts) > 0
    assert len(age.histogram.bin_edges) == len(age.histogram.counts) + 1
    assert age.statistics["skewness"] == 0.35

    # Income has histogram and skewness
    income = cols["Income"]
    assert isinstance(income.histogram, HistogramData)
    assert len(income.histogram.bin_edges) == len(income.histogram.counts) + 1
    assert income.statistics["skewness"] == 1.25

    # Score has None histogram and None statistics
    score = cols["Score"]
    assert score.histogram is None
    assert score.statistics is None

    # Categorical Department has None histogram
    dept = cols["Department"]
    assert dept.histogram is None


def test_real_aksis_service_dataspec_compatible_tasks_types():
    from backend.services.aksis_service import RealAksisService
    from types import SimpleNamespace

    service = RealAksisService()

    # 1. Classification dataset
    spec_cls = SimpleNamespace(
        id="breast_cancer",
        display_name="Breast Cancer",
        source="scikit_learn",
        target="target",
        compatible_tasks=["classification"]
    )
    meta_cls = service._map_dataspec_to_metadata(spec_cls)
    assert meta_cls.compatible_tasks == ["classification"]

    # 2. Regression dataset
    spec_reg = SimpleNamespace(
        id="california_housing",
        display_name="California Housing",
        source="scikit_learn",
        target="MedHouseVal",
        compatible_tasks=["regression"]
    )
    meta_reg = service._map_dataspec_to_metadata(spec_reg)
    assert meta_reg.compatible_tasks == ["regression"]

    # 3. Anomaly detection dataset
    spec_anom = SimpleNamespace(
        id="kddcup99",
        display_name="KDDCup99",
        source="csv_file",
        target=None,
        compatible_tasks=["anomaly_detection"]
    )
    meta_anom = service._map_dataspec_to_metadata(spec_anom)
    assert meta_anom.compatible_tasks == ["anomaly_detection"]

    # 4. Dict format
    spec_dict = {
        "id": "dict_ds",
        "display_name": "Dict Dataset",
        "compatible_tasks": ("classification",)
    }
    meta_dict = service._map_dataspec_to_metadata(spec_dict)
    assert meta_dict.compatible_tasks == ["classification"]

    # 5. Missing / None compatible_tasks defaults to empty list
    spec_none = SimpleNamespace(id="no_tasks")
    meta_none = service._map_dataspec_to_metadata(spec_none)
    assert meta_none.compatible_tasks == []


def test_dataset_to_learning_type_and_task_resolution_logic():
    """Verify the capability-driven resolution algorithm used in frontend."""
    caps_tasks_dict = {
        "supervised": ["classification", "regression"],
        "unsupervised": ["anomaly_detection"]
    }

    # Case A: Unique classification dataset
    compat_a = ["classification"]
    task_to_lt_a = {t: lt for lt, t_list in caps_tasks_dict.items() for t in t_list if t in compat_a}
    assert list(task_to_lt_a.keys()) == ["classification"]
    assert list(dict.fromkeys(task_to_lt_a.values())) == ["supervised"]

    # Case B: Unique anomaly detection dataset
    compat_b = ["anomaly_detection"]
    task_to_lt_b = {t: lt for lt, t_list in caps_tasks_dict.items() for t in t_list if t in compat_b}
    assert list(task_to_lt_b.keys()) == ["anomaly_detection"]
    assert list(dict.fromkeys(task_to_lt_b.values())) == ["unsupervised"]

    # Case C: Dataset supporting both classification and regression (multi-task)
    compat_c = ["classification", "regression"]
    task_to_lt_c = {t: lt for lt, t_list in caps_tasks_dict.items() for t in t_list if t in compat_c}
    assert set(task_to_lt_c.keys()) == {"classification", "regression"}
    assert list(dict.fromkeys(task_to_lt_c.values())) == ["supervised"]

    # Case D: Incompatible / unknown task metadata
    compat_d = ["unsupported_future_task"]
    task_to_lt_d = {t: lt for lt, t_list in caps_tasks_dict.items() for t in t_list if t in compat_d}
    assert len(task_to_lt_d) == 0







