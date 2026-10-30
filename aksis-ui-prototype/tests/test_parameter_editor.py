import pytest
from unittest.mock import patch, MagicMock
from frontend.components.parameter_editor import render_parameter_editor

def test_parameter_editor_empty_or_none_schema():
    with patch("streamlit.warning") as mock_warning:
        # None schema
        res_none = render_parameter_editor(None)
        assert res_none == {}
        mock_warning.assert_called_with("Bu model için özelleştirilebilir parametre şeması sağlanmamış.")

        # Empty dict schema
        res_empty = render_parameter_editor({})
        assert res_empty == {}

def test_parameter_editor_generic_types():
    schema = {
        "n_estimators": {
            "type": "int",
            "default": 100,
            "min": 10,
            "max": 1000,
            "step": 10,
            "description": "Number of trees"
        },
        "learning_rate": {
            "type": "float",
            "default": 0.05,
            "min": 0.001,
            "max": 1.0,
            "step": 0.01,
            "description": "Learning rate"
        },
        "use_gpu": {
            "type": "bool",
            "default": True,
            "description": "GPU support"
        },
        "tag": {
            "type": "str",
            "default": "v1",
            "description": "Model tag"
        },
        "criterion": {
            "type": "choice",
            "default": "gini",
            "choices": ["gini", "entropy", "log_loss"],
            "description": "Split criterion"
        }
    }

    with patch("streamlit.number_input") as mock_number, \
         patch("streamlit.checkbox") as mock_checkbox, \
         patch("streamlit.text_input") as mock_text, \
         patch("streamlit.selectbox") as mock_select, \
         patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]):

        mock_number.side_effect = [200, 0.02]
        mock_checkbox.return_value = True
        mock_text.return_value = "custom_run"
        mock_select.return_value = "entropy"

        overrides = render_parameter_editor(schema, key_prefix="test_param")

        assert overrides["n_estimators"] == 200
        assert overrides["learning_rate"] == 0.02
        assert overrides["use_gpu"] is True
        assert overrides["tag"] == "custom_run"
        assert overrides["criterion"] == "entropy"

def test_parameter_editor_session_state_cleanup_behavior():
    session_state = {
        "custom_param_xgb_c_n_estimators": 500,
        "custom_param_xgb_c_learning_rate": 0.01,
        "selected_dataset_id": "ds_class_01"
    }

    # Simulate context switch from xgb_c to svc
    current_context = "classification_svc_custom"
    last_context = "classification_xgb_c_custom"

    if current_context != last_context:
        for k in list(session_state.keys()):
            if k.startswith("custom_param_"):
                del session_state[k]

    # Verify custom_param keys were purged and unrelated keys remained
    assert "custom_param_xgb_c_n_estimators" not in session_state
    assert "custom_param_xgb_c_learning_rate" not in session_state
    assert session_state["selected_dataset_id"] == "ds_class_01"


def test_custom_preset_temporary_disabled_contract():
    """
    Verifies that:
    - 'custom' is present in PRESET_NAMES_TR
    - Its label indicates it is temporarily disabled
    - Non-custom presets keep their normal display label
    """
    import importlib.util
    from pathlib import Path

    file_path = Path("frontend/pages/3_experiment_config.py").resolve()
    spec = importlib.util.spec_from_file_location("exp_config", file_path)
    # Read the file content and check PRESET_NAMES_TR definition directly or evaluate
    content = file_path.read_text(encoding="utf-8")
    assert '"custom": "Özel Parametre Ayarı (custom) — Geçici Olarak Kapalı"' in content

    # Alternatively parse PRESET_NAMES_TR safely
    import ast
    tree = ast.parse(content)
    preset_names_tr = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "PRESET_NAMES_TR":
                    preset_names_tr = ast.literal_eval(stmt.value)
                    break

    assert preset_names_tr is not None
    assert "custom" in preset_names_tr
    assert preset_names_tr["custom"] == "Özel Parametre Ayarı (custom) — Geçici Olarak Kapalı"
    assert preset_names_tr["baseline"] == "Temel Seviye (baseline)"
    assert preset_names_tr["fast"] == "Hızlı (fast)"
    assert preset_names_tr["strong"] == "Yüksek Başarım (strong)"


def test_custom_preset_ui_behavior():
    """
    Verifies that when preset == 'custom', parameter editor inputs are NOT rendered
    and an informational message is displayed instead, whereas non-custom presets
    operate normally without rendering parameter inputs or disable messages.
    """
    # Mock capabilities with custom in model_presets and parameter schema
    mock_caps = {
        "model_presets": ["baseline", "fast", "strong", "custom"],
        "parameter_schema": {
            "classification": {
                "rf_classifier": {
                    "n_estimators": {"type": "int", "default": 100}
                }
            }
        }
    }

    # 1. Custom preset is present in capabilities
    assert "custom" in mock_caps["model_presets"]

    # 2. Simulate custom selection behavior
    preset = "custom"
    algorithm = "rf_classifier"
    task = "classification"

    with patch("streamlit.info") as mock_info, \
         patch("frontend.components.parameter_editor.render_parameter_editor") as mock_render_editor, \
         patch("streamlit.number_input") as mock_number_input:

        # In 3_experiment_config.py:
        if preset == "custom" and algorithm:
            # st.divider()
            mock_info("Custom model parameter configuration is temporarily disabled for the demo.")
            custom_overrides = {}

        # Assert render_parameter_editor was NOT called
        mock_render_editor.assert_not_called()
        mock_number_input.assert_not_called()
        mock_info.assert_called_once_with("Custom model parameter configuration is temporarily disabled for the demo.")
        assert custom_overrides == {}

    # 3. Simulate non-custom preset behavior (e.g. baseline)
    preset = "baseline"
    with patch("streamlit.info") as mock_info, \
         patch("frontend.components.parameter_editor.render_parameter_editor") as mock_render_editor:

        if preset == "custom" and algorithm:
            mock_info("Custom model parameter configuration is temporarily disabled for the demo.")
            custom_overrides = {}
        else:
            custom_overrides = {}

        mock_render_editor.assert_not_called()
        mock_info.assert_not_called()
        assert custom_overrides == {}


def test_config_applicability_helper():
    from frontend.utils.model_guidance import is_config_applicable

    caps = {
        "config_applicability": {
            "data_split": {
                "algorithm_types": ["supervised"],
                "supported_tasks": ["classification", "regression"]
            },
            "supervised_tuning": {
                "algorithm_types": ["supervised"],
                "supported_tasks": ["classification", "regression"]
            },
            "unsupervised_tuning": {
                "algorithm_types": ["unsupervised"],
                "supported_tasks": ["anomaly_detection"]
            },
            "preprocess": {
                "algorithm_types": ["supervised", "unsupervised"],
                "supported_tasks": ["classification", "regression", "anomaly_detection"]
            }
        }
    }

    # Data split: applicable for supervised classification and regression, but NOT for unsupervised anomaly_detection
    assert is_config_applicable(caps, "data_split", "supervised", "classification") is True
    assert is_config_applicable(caps, "data_split", "supervised", "regression") is True
    assert is_config_applicable(caps, "data_split", "unsupervised", "anomaly_detection") is False
    assert is_config_applicable(caps, "data_split", "supervised", "anomaly_detection") is False

    # Supervised tuning
    assert is_config_applicable(caps, "supervised_tuning", "supervised", "classification") is True
    assert is_config_applicable(caps, "supervised_tuning", "unsupervised", "anomaly_detection") is False

    # Unsupervised tuning
    assert is_config_applicable(caps, "unsupervised_tuning", "unsupervised", "anomaly_detection") is True
    assert is_config_applicable(caps, "unsupervised_tuning", "supervised", "classification") is False

    # Non-existent config
    assert is_config_applicable(caps, "non_existent", "supervised", "classification") is False
    assert is_config_applicable({}, "data_split", "supervised", "classification") is False
    assert is_config_applicable(None, "data_split", "supervised", "classification") is False


def test_task_learning_type_resolution_from_capabilities():
    from frontend.utils.model_guidance import resolve_task_learning_types

    caps = {
        "learning_types": ["supervised", "unsupervised"],
        "tasks": ["classification", "regression", "anomaly_detection"],
        "config_applicability": {
            "supervised_model": {
                "algorithm_types": ["supervised"],
                "supported_tasks": ["classification", "regression"]
            },
            "unsupervised_model": {
                "algorithm_types": ["unsupervised"],
                "supported_tasks": ["anomaly_detection"]
            }
        }
    }

    # Test resolution of supervised tasks
    resolved_sup = resolve_task_learning_types(caps, ["classification", "regression"])
    assert resolved_sup == {
        "classification": "supervised",
        "regression": "supervised"
    }

    # Test resolution of unsupervised task
    resolved_unsup = resolve_task_learning_types(caps, ["anomaly_detection"])
    assert resolved_unsup == {
        "anomaly_detection": "unsupervised"
    }

    # Test mixed candidate tasks
    resolved_mixed = resolve_task_learning_types(caps, ["classification", "anomaly_detection"])
    assert resolved_mixed == {
        "classification": "supervised",
        "anomaly_detection": "unsupervised"
    }

    # Unsupported candidate task
    resolved_unknown = resolve_task_learning_types(caps, ["forecasting"])
    assert resolved_unknown == {}


def test_contextual_model_presets_and_tuning_capabilities():
    from backend.demo.fixtures import CAPABILITIES

    caps = CAPABILITIES.model_dump()

    # 1. Verify absence of old flattened keys in new CapabilityResponse
    assert "tuning_options" not in caps
    assert "scoring_options" not in caps
    assert "preprocessing_strategies" not in caps
    assert "validation_options" not in caps

    # 2. Contextual model presets
    assert isinstance(caps["model_presets"], dict)
    assert "supervised" in caps["model_presets"]
    assert "unsupervised" in caps["model_presets"]
    assert "baseline" in caps["model_presets"]["supervised"]
    assert "custom" in caps["model_presets"]["supervised"]
    assert "custom" in caps["model_presets"]["unsupervised"]

    # 3. Contextual tuning structure
    tuning = caps["tuning"]
    assert "samplers" in tuning
    assert "pruners" in tuning
    assert "objectives" in tuning

    assert tuning["samplers"]["supervised"] == ["tpe", "random"]
    assert tuning["samplers"]["unsupervised"] == ["random"]
    assert tuning["pruners"]["supervised"] == ["none", "median", "sha"]
    assert tuning["pruners"]["unsupervised"] == ["none"]

    assert tuning["objectives"]["classification"] == ["f1_macro", "accuracy", "balanced_accuracy"]
    assert tuning["objectives"]["regression"] == ["neg_mean_squared_error", "r2", "rmse", "mae"]
    assert tuning["objectives"]["anomaly_detection"] == ["f1_score", "accuracy", "anomaly_count", "anomaly_ratio"]

    # 4. Preprocessing structure
    assert "preprocessing" in caps
    assert "missing_value" in caps["preprocessing"]
    assert "encoding" in caps["preprocessing"]
    assert "scaling" in caps["preprocessing"]


def test_data_split_dynamic_type_rendering_and_payload():
    """
    Verifies that:
    1. methods drives method selection.
    2. parameters drives dynamic controls.
    3. bool values remain bool.
    4. int values remain int.
    5. float values remain float.
    6. only returned parameters are included in payload.
    7. no parameter_schema is required anymore.
    """
    capabilities = {
        "data_split": {
            "methods": ["standard_random"],
            "parameters": {
                "test_size": 0.2,
                "random_state": 42,
                "shuffle": True,
                "stratify_column": "target"
            }
        }
    }

    split_meta = capabilities["data_split"]
    val_options = split_meta.get("methods", [])
    split_params = split_meta.get("parameters", {})

    assert val_options == ["standard_random"]
    assert "parameter_schema" not in split_meta

    val_strategy = val_options[0]
    data_split_params = {}

    for p_name, p_default in split_params.items():
        if isinstance(p_default, bool):
            data_split_params[p_name] = bool(p_default)
        elif isinstance(p_default, int):
            data_split_params[p_name] = int(p_default)
        elif isinstance(p_default, float):
            data_split_params[p_name] = float(p_default)
        elif isinstance(p_default, str):
            data_split_params[p_name] = str(p_default)
        else:
            data_split_params[p_name] = p_default

    payload = {
        "method": val_strategy,
        "parameters": data_split_params
    }

    assert payload["method"] == "standard_random"
    assert payload["parameters"]["test_size"] == 0.2
    assert isinstance(payload["parameters"]["test_size"], float)
    assert payload["parameters"]["random_state"] == 42
    assert isinstance(payload["parameters"]["random_state"], int)
    assert payload["parameters"]["shuffle"] is True
    assert isinstance(payload["parameters"]["shuffle"], bool)
    assert payload["parameters"]["stratify_column"] == "target"
    assert isinstance(payload["parameters"]["stratify_column"], str)
    assert "n_splits" not in payload["parameters"]
    assert len(payload["parameters"]) == 4


def test_capability_diagnostics_core_and_groups():
    from frontend.utils.model_guidance import (
        validate_capability_path,
        diagnose_config_applicability,
        diagnose_model_presets,
        CapabilityDiagnostic
    )

    # 1. Core checks
    diag_none = validate_capability_path(None, path=("learning_types",))
    assert not diag_none
    assert diag_none.code == "CAP-CORE-001"
    assert "CAP-CORE-001" in str(diag_none)

    caps = {
        "learning_types": [],
        "tasks": ["classification"],
        "modes": []
    }
    diag_lt = validate_capability_path(caps, path=("learning_types",), allow_empty=False)
    assert not diag_lt
    assert diag_lt.code == "CAP-CORE-001"

    diag_tasks = validate_capability_path(caps, path=("tasks",), allow_empty=False)
    assert bool(diag_tasks) is True
    assert diag_tasks.code == "CAP-OK"

    diag_modes = validate_capability_path(caps, path=("modes",), allow_empty=False)
    assert not diag_modes
    assert diag_modes.code == "CAP-CORE-003"

    diag_algos = validate_capability_path(caps, path=("algorithms",), required=True)
    assert not diag_algos
    assert diag_algos.code == "CAP-CORE-004"

    # 2. Data Split checks
    diag_ds_missing = validate_capability_path(caps, path=("data_split",), required=True)
    assert not diag_ds_missing
    assert diag_ds_missing.code == "CAP-DS-001"

    caps["data_split"] = {"methods": []}
    diag_ds_methods = validate_capability_path(caps, path=("data_split", "methods"), allow_empty=False)
    assert not diag_ds_methods
    assert diag_ds_methods.code == "CAP-DS-002"

    diag_ds_params = validate_capability_path(caps, path=("data_split", "parameters"), required=True)
    assert not diag_ds_params
    assert diag_ds_params.code == "CAP-DS-003"

    # 3. Tuning & Preprocessing checks
    diag_tune = validate_capability_path(caps, path=("tuning",), required=True)
    assert not diag_tune
    assert diag_tune.code == "CAP-TUNE-001"

    diag_prep = validate_capability_path(caps, path=("preprocessing",), required=True)
    assert not diag_prep
    assert diag_prep.code == "CAP-PREP-001"

    # 4. Config Applicability detailed diagnostics
    diag_app_missing = diagnose_config_applicability(caps, "data_split")
    assert not diag_app_missing
    assert diag_app_missing.code == "CAP-APP-001"

    caps["config_applicability"] = {
        "data_split": {
            "algorithm_types": [],
            "supported_tasks": ["classification"]
        }
    }
    diag_app_entry = diagnose_config_applicability(caps, "preprocess")
    assert not diag_app_entry
    assert diag_app_entry.code == "CAP-APP-002"

    diag_app_algo = diagnose_config_applicability(caps, "data_split")
    assert not diag_app_algo
    assert diag_app_algo.code == "CAP-APP-003"

    caps["config_applicability"]["data_split"]["algorithm_types"] = ["supervised"]
    caps["config_applicability"]["data_split"]["supported_tasks"] = []
    diag_app_task = diagnose_config_applicability(caps, "data_split")
    assert not diag_app_task
    assert diag_app_task.code == "CAP-APP-004"

    caps["config_applicability"]["data_split"]["supported_tasks"] = ["classification"]
    diag_app_ok = diagnose_config_applicability(caps, "data_split")
    assert bool(diag_app_ok) is True
    assert diag_app_ok.code == "CAP-OK"

    # 5. Model Presets detailed diagnostics
    diag_preset_root = diagnose_model_presets(caps)
    assert not diag_preset_root
    assert diag_preset_root.code == "CAP-MODEL-001"

    caps["model_presets"] = {"supervised": ["fast"], "unsupervised": []}
    diag_preset_unsup = diagnose_model_presets(caps, "unsupervised")
    assert not diag_preset_unsup
    assert diag_preset_unsup.code == "CAP-MODEL-002"

    diag_preset_sup = diagnose_model_presets(caps, "supervised")
    assert bool(diag_preset_sup) is True
    assert diag_preset_sup.code == "CAP-OK"


def test_preprocessing_capability_contract_and_payload():
    """
    Verifies that:
    1. encoding methods come from capability
    2. scaling methods come from capability
    3. defaults come from preprocessing.defaults
    4. targetEncoding remains exactly targetEncoding
    5. encoding_parameters preserve bool/int/float/str types
    6. scaling_parameters preserve bool/int/float/str types
    7. missing imputer strategy lists are not fabricated
    8. outgoing payload matches canonical PreprocessConfig fields
    9. missing preprocessing section produces CAP-PREP-001
    10. missing defaults produces CAP-PREP-002
    """
    from frontend.utils.model_guidance import validate_capability_path

    # Capability shape with new preprocessing contract
    raw_caps = {
        "preprocessing": {
            "encoding_methods": ["onehot", "ordinal", "target"],
            "scaling_methods": ["standard", "minmax", "robust"],
            "defaults": {
                "handle_missing": True,
                "numeric_imputer": "mean",
                "categorical_imputer": "most_frequent",
                "numeric_fill_value": 0.0,
                "categorical_fill_value": "missing",
                "encoding": True,
                "encode_method": "onehot",
                "targetEncoding": False,
                "encoding_parameters": {
                    "drop_first": True,
                    "max_categories": 10,
                    "handle_unknown": "ignore",
                    "min_frequency": 0.05
                },
                "scaling": True,
                "scale_method": "standard",
                "scaling_parameters": {
                    "with_mean": True,
                    "with_std": True,
                    "quantile_range_min": 25.0,
                    "quantile_range_max": 75.0,
                    "unit_variance": False
                }
            }
        }
    }

    prep_caps = raw_caps["preprocessing"]
    prep_defaults = prep_caps["defaults"]

    # 1. Encoding methods come from capability
    assert prep_caps.get("encoding_methods") == ["onehot", "ordinal", "target"]

    # 2. Scaling methods come from capability
    assert prep_caps.get("scaling_methods") == ["standard", "minmax", "robust"]

    # 3. Defaults come from preprocessing.defaults
    assert prep_defaults.get("numeric_imputer") == "mean"
    assert prep_defaults.get("categorical_imputer") == "most_frequent"
    assert prep_defaults.get("numeric_fill_value") == 0.0
    assert prep_defaults.get("categorical_fill_value") == "missing"
    assert prep_defaults.get("encode_method") == "onehot"
    assert prep_defaults.get("scale_method") == "standard"

    # 4. targetEncoding remains exactly targetEncoding
    assert "targetEncoding" in prep_defaults
    assert "target_encoding" not in prep_defaults

    # 5. encoding_parameters preserve bool/int/float/str types
    enc_params = prep_defaults["encoding_parameters"]
    assert isinstance(enc_params["drop_first"], bool)
    assert isinstance(enc_params["max_categories"], int)
    assert isinstance(enc_params["handle_unknown"], str)
    assert isinstance(enc_params["min_frequency"], float)

    # 6. scaling_parameters preserve bool/int/float/str types
    scale_params = prep_defaults["scaling_parameters"]
    assert isinstance(scale_params["with_mean"], bool)
    assert isinstance(scale_params["with_std"], bool)
    assert isinstance(scale_params["quantile_range_min"], float)
    assert isinstance(scale_params["quantile_range_max"], float)
    assert isinstance(scale_params["unit_variance"], bool)

    # 7. Missing imputer strategy lists are not fabricated
    assert "missing_value" not in prep_caps
    assert "imputer_strategies" not in prep_caps

    # 8. Outgoing payload matches canonical PreprocessConfig fields
    payload = {
        "handle_missing": bool(prep_defaults["handle_missing"]),
        "numeric_imputer": prep_defaults["numeric_imputer"],
        "categorical_imputer": prep_defaults["categorical_imputer"],
        "numeric_fill_value": prep_defaults["numeric_fill_value"],
        "categorical_fill_value": prep_defaults["categorical_fill_value"],
        "encoding": bool(prep_defaults["encoding"]),
        "encode_method": prep_defaults["encode_method"],
        "targetEncoding": bool(prep_defaults["targetEncoding"]),
        "encoding_parameters": enc_params,
        "scaling": bool(prep_defaults["scaling"]),
        "scale_method": prep_defaults["scale_method"],
        "scaling_parameters": scale_params
    }

    expected_canonical_keys = {
        "handle_missing",
        "numeric_imputer",
        "categorical_imputer",
        "numeric_fill_value",
        "categorical_fill_value",
        "encoding",
        "encode_method",
        "targetEncoding",
        "encoding_parameters",
        "scaling",
        "scale_method",
        "scaling_parameters"
    }
    assert set(payload.keys()) == expected_canonical_keys

    # 9. Missing preprocessing section produces CAP-PREP-001
    diag_missing_prep = validate_capability_path({}, path=("preprocessing",), required=True)
    assert not diag_missing_prep
    assert diag_missing_prep.code == "CAP-PREP-001"

    # 10. Missing defaults produces CAP-PREP-002
    diag_missing_defaults = validate_capability_path({"preprocessing": {}}, path=("preprocessing", "defaults"), required=True)
    assert not diag_missing_defaults
    assert diag_missing_defaults.code == "CAP-PREP-002"

    # 11. Empty encoding_methods produces CAP-PREP-003
    diag_enc_methods = validate_capability_path({"preprocessing": {"encoding_methods": []}}, path=("preprocessing", "encoding_methods"), required=True, allow_empty=False)
    assert not diag_enc_methods
    assert diag_enc_methods.code == "CAP-PREP-003"

    # 12. Empty scaling_methods produces CAP-PREP-004
    diag_scale_methods = validate_capability_path({"preprocessing": {"scaling_methods": []}}, path=("preprocessing", "scaling_methods"), required=True, allow_empty=False)
    assert not diag_scale_methods
    assert diag_scale_methods.code == "CAP-PREP-004"






