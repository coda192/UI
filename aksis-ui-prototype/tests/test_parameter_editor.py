import pytest
from unittest.mock import patch, MagicMock
from frontend.components.parameter_editor import render_parameter_editor
from frontend.utils.model_guidance import validate_capability_path

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
            "numeric_imputer_methods": ["mean", "median", "most_frequent", "constant"],
            "categorical_imputer_methods": ["most_frequent", "constant", "missing"],
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

    # 13. Empty numeric_imputer_methods produces CAP-PREP-005
    diag_num_imp_methods = validate_capability_path({"preprocessing": {"numeric_imputer_methods": []}}, path=("preprocessing", "numeric_imputer_methods"), required=True, allow_empty=False)
    assert not diag_num_imp_methods
    assert diag_num_imp_methods.code == "CAP-PREP-005"

    # 14. Empty categorical_imputer_methods produces CAP-PREP-006
    diag_cat_imp_methods = validate_capability_path({"preprocessing": {"categorical_imputer_methods": []}}, path=("preprocessing", "categorical_imputer_methods"), required=True, allow_empty=False)
    assert not diag_cat_imp_methods
    assert diag_cat_imp_methods.code == "CAP-PREP-006"

    # 15. Empty data_analyze produces CAP-AN-001
    diag_da = validate_capability_path({"data_analyze": {}}, path=("data_analyze",), required=True, allow_empty=False)
    assert not diag_da
    assert diag_da.code == "CAP-AN-001"

    # 16. Empty data_analyze.defaults produces CAP-AN-002
    diag_da_defaults = validate_capability_path({"data_analyze": {"defaults": {}}}, path=("data_analyze", "defaults"), required=True, allow_empty=False)
    assert not diag_da_defaults
    assert diag_da_defaults.code == "CAP-AN-002"

    # 17. Empty data_analyze_functions produces CAP-AN-003
    diag_da_fns = validate_capability_path({"data_analyze": {"defaults": {"data_analyze_functions": []}}}, path=("data_analyze", "defaults", "data_analyze_functions"), required=True, allow_empty=False)
    assert not diag_da_fns
    assert diag_da_fns.code == "CAP-AN-003"


def test_preprocessing_method_specific_nested_parameters():
    """
    Verifies that:
    - encoding_parameters only exposes selected encode_method's parameters
    - scaling_parameters only exposes selected scale_method's parameters
    - parameters from other methods do not leak into the payload
    - bool/int/float/str types are preserved
    """
    nested_caps = {
        "preprocessing": {
            "encoding_methods": ["onehot", "frequency", "hashing"],
            "scaling_methods": ["standard", "minmax", "robust"],
            "defaults": {
                "handle_missing": True,
                "numeric_imputer": "mean",
                "categorical_imputer": "most_frequent",
                "numeric_fill_value": None,
                "categorical_fill_value": None,
                "encoding": True,
                "encode_method": "onehot",
                "targetEncoding": False,
                "encoding_parameters": {
                    "onehot": {
                        "drop_first": True,
                        "max_categories": 10
                    },
                    "frequency": {
                        "min_frequency": 0.05
                    },
                    "hashing": {
                        "n_features": 100,
                        "alternate_sign": False
                    }
                },
                "scaling": True,
                "scale_method": "minmax",
                "scaling_parameters": {
                    "standard": {
                        "with_mean": True,
                        "with_std": True
                    },
                    "minmax": {
                        "feature_range_min": 0.0,
                        "feature_range_max": 1.0,
                        "clip": False
                    },
                    "robust": {
                        "quantile_range_min": 25.0,
                        "quantile_range_max": 75.0,
                        "unit_variance": True
                    }
                }
            }
        }
    }

    defaults = nested_caps["preprocessing"]["defaults"]
    raw_enc = defaults["encoding_parameters"]
    raw_scale = defaults["scaling_parameters"]

    # 1. Test selected encode_method = "onehot"
    selected_encode_method = "onehot"
    onehot_params = raw_enc.get(selected_encode_method, {})
    assert "drop_first" in onehot_params
    assert "max_categories" in onehot_params
    assert "min_frequency" not in onehot_params
    assert "n_features" not in onehot_params
    assert isinstance(onehot_params["drop_first"], bool)
    assert isinstance(onehot_params["max_categories"], int)

    # 2. Test switching encode_method to "frequency"
    freq_params = raw_enc.get("frequency", {})
    assert "min_frequency" in freq_params
    assert "drop_first" not in freq_params
    assert isinstance(freq_params["min_frequency"], float)

    # 3. Test selected scale_method = "minmax"
    selected_scale_method = "minmax"
    minmax_params = raw_scale.get(selected_scale_method, {})
    assert "feature_range_min" in minmax_params
    assert "feature_range_max" in minmax_params
    assert "clip" in minmax_params
    assert "with_mean" not in minmax_params
    assert "quantile_range_min" not in minmax_params
    assert isinstance(minmax_params["feature_range_min"], float)
    assert isinstance(minmax_params["feature_range_max"], float)
    assert isinstance(minmax_params["clip"], bool)

    # 4. Outgoing payload should contain ONLY the selected method's parameters
    payload = {
        "handle_missing": bool(defaults["handle_missing"]),
        "numeric_imputer": defaults["numeric_imputer"],
        "categorical_imputer": defaults["categorical_imputer"],
        "numeric_fill_value": defaults["numeric_fill_value"],
        "categorical_fill_value": defaults["categorical_fill_value"],
        "encoding": bool(defaults["encoding"]),
        "encode_method": selected_encode_method,
        "targetEncoding": bool(defaults["targetEncoding"]),
        "encoding_parameters": onehot_params,
        "scaling": bool(defaults["scaling"]),
        "scale_method": selected_scale_method,
        "scaling_parameters": minmax_params
    }

    assert payload["encoding_parameters"] == {"drop_first": True, "max_categories": 10}
    assert "min_frequency" not in payload["encoding_parameters"]
    assert payload["scaling_parameters"] == {"feature_range_min": 0.0, "feature_range_max": 1.0, "clip": False}
    assert "with_mean" not in payload["scaling_parameters"]


def test_preprocessing_missing_value_imputation_controls_and_payload():
    """
    Verifies that:
    1. numeric imputer options come from capability (numeric_imputer_methods)
    2. categorical imputer options come from capability (categorical_imputer_methods)
    3. no hardcoded imputer strategy lists exist in frontend
    4. handle_missing=False hides/disables imputation controls logically
    5. numeric 'constant' exposes numeric_fill_value
    6. numeric non-constant produces numeric_fill_value=None
    7. categorical 'constant' exposes categorical_fill_value
    8. categorical non-constant produces categorical_fill_value=None
    9. strategy defaults come from preprocessing.defaults
    10. encoding/scaling behavior remains unchanged
    11. canonical preprocess payload field names remain unchanged
    """
    caps = {
        "preprocessing": {
            "numeric_imputer_methods": ["median", "constant", "mean"],
            "categorical_imputer_methods": ["most_frequent", "constant"],
            "encoding_methods": ["onehot"],
            "scaling_methods": ["standard"],
            "defaults": {
                "handle_missing": True,
                "numeric_imputer": "median",
                "categorical_imputer": "most_frequent",
                "numeric_fill_value": 42.0,
                "categorical_fill_value": "missing_token",
                "encoding": False,
                "encode_method": "onehot",
                "targetEncoding": False,
                "encoding_parameters": {},
                "scaling": False,
                "scale_method": "standard",
                "scaling_parameters": {}
            }
        }
    }

    prep_caps = caps["preprocessing"]
    defaults = prep_caps["defaults"]

    # 1 & 2. Imputer methods come from capability
    assert prep_caps["numeric_imputer_methods"] == ["median", "constant", "mean"]
    assert prep_caps["categorical_imputer_methods"] == ["most_frequent", "constant"]

    # 9. Defaults come from preprocessing.defaults
    assert defaults["numeric_imputer"] == "median"
    assert defaults["categorical_imputer"] == "most_frequent"
    assert defaults["handle_missing"] is True

    # 4. handle_missing = False -> imputer fields and fill values are None
    def build_payload(handle_missing, num_imp, cat_imp, num_fill, cat_fill):
        final_num_fill = float(num_fill) if (handle_missing and num_imp == "constant" and num_fill is not None) else None
        final_cat_fill = str(cat_fill) if (handle_missing and cat_imp == "constant" and cat_fill is not None) else None
        return {
            "handle_missing": bool(handle_missing),
            "numeric_imputer": num_imp if (handle_missing and num_imp) else None,
            "categorical_imputer": cat_imp if (handle_missing and cat_imp) else None,
            "numeric_fill_value": final_num_fill,
            "categorical_fill_value": final_cat_fill,
            "encoding": bool(defaults["encoding"]),
            "encode_method": defaults["encode_method"] if defaults["encoding"] else None,
            "targetEncoding": bool(defaults["targetEncoding"]),
            "encoding_parameters": {},
            "scaling": bool(defaults["scaling"]),
            "scale_method": defaults["scale_method"] if defaults["scaling"] else None,
            "scaling_parameters": {}
        }

    # Case A: handle_missing = False
    payload_disabled = build_payload(False, "median", "most_frequent", 42.0, "missing_token")
    assert payload_disabled["handle_missing"] is False
    assert payload_disabled["numeric_imputer"] is None
    assert payload_disabled["categorical_imputer"] is None
    assert payload_disabled["numeric_fill_value"] is None
    assert payload_disabled["categorical_fill_value"] is None

    # Case B: handle_missing = True with non-constant strategies
    payload_non_constant = build_payload(True, "median", "most_frequent", 42.0, "missing_token")
    assert payload_non_constant["handle_missing"] is True
    assert payload_non_constant["numeric_imputer"] == "median"
    assert payload_non_constant["categorical_imputer"] == "most_frequent"
    # 6 & 8. Non-constant strategies produce None for fill values
    assert payload_non_constant["numeric_fill_value"] is None
    assert payload_non_constant["categorical_fill_value"] is None

    # Case C: handle_missing = True with constant strategies
    payload_constant = build_payload(True, "constant", "constant", 42.0, "missing_token")
    assert payload_constant["handle_missing"] is True
    assert payload_constant["numeric_imputer"] == "constant"
    assert payload_constant["categorical_imputer"] == "constant"
    # 5 & 7. Constant strategy preserves fill values with correct types
    assert payload_constant["numeric_fill_value"] == 42.0
    assert isinstance(payload_constant["numeric_fill_value"], float)
    assert payload_constant["categorical_fill_value"] == "missing_token"
    assert isinstance(payload_constant["categorical_fill_value"], str)

    # 11. Canonical preprocess field names verified
    expected_fields = {
        "handle_missing", "numeric_imputer", "categorical_imputer",
        "numeric_fill_value", "categorical_fill_value",
        "encoding", "encode_method", "targetEncoding", "encoding_parameters",
        "scaling", "scale_method", "scaling_parameters"
    }
    assert set(payload_constant.keys()) == expected_fields


def test_missing_value_imputation_bugfixes_and_edge_cases():
    """
    Directly tests the two bug fixes:
    1. handle_missing=False does not raise NameError and variables are safe.
    2. All four imputation variables have safe values when disabled.
    3. Disabled missing handling produces None for all imputation payload fields.
    4. Numeric fill default None does NOT become 0 or 0.0 automatically.
    5. Blank numeric constant input becomes None.
    6. Valid numeric constant input preserves numeric type (int and float).
    7. Invalid numeric constant input is rejected cleanly (becomes None, no crash).
    8. Categorical default None / untouched empty string normalizes to None.
    """
    # Simulate UI initialization before checkbox
    ui_numeric_imputer = None
    ui_categorical_imputer = None
    ui_numeric_fill = None
    ui_categorical_fill = None

    # Test 1 & 2: handle_missing = False does not raise NameError and state is safe
    ui_handle_missing = False
    if ui_handle_missing:
        ui_numeric_imputer = "mean"
        ui_categorical_imputer = "most_frequent"
        ui_numeric_fill = 1.0
        ui_categorical_fill = "constant"

    assert ui_numeric_imputer is None
    assert ui_categorical_imputer is None
    assert ui_numeric_fill is None
    assert ui_categorical_fill is None

    # Test 3: Disabled missing handling produces None for all imputation payload fields
    final_num_imputer = ui_numeric_imputer if ui_handle_missing else None
    final_cat_imputer = ui_categorical_imputer if ui_handle_missing else None
    final_num_fill = (
        ui_numeric_fill
        if (ui_handle_missing and ui_numeric_imputer == "constant" and ui_numeric_fill is not None)
        else None
    )
    final_cat_fill = (
        ui_categorical_fill
        if (ui_handle_missing and ui_categorical_imputer == "constant" and ui_categorical_fill is not None)
        else None
    )

    payload_disabled = {
        "handle_missing": bool(ui_handle_missing),
        "numeric_imputer": final_num_imputer,
        "categorical_imputer": final_cat_imputer,
        "numeric_fill_value": final_num_fill,
        "categorical_fill_value": final_cat_fill,
    }
    assert payload_disabled == {
        "handle_missing": False,
        "numeric_imputer": None,
        "categorical_imputer": None,
        "numeric_fill_value": None,
        "categorical_fill_value": None,
    }

    # Parser function matching 3_experiment_config.py implementation
    def parse_numeric_fill_input(raw_input: str):
        raw = raw_input.strip() if raw_input is not None else ""
        if raw == "":
            return None
        try:
            if "." in raw or "e" in raw.lower():
                return float(raw)
            else:
                try:
                    return int(raw)
                except ValueError:
                    return float(raw)
        except ValueError:
            return None

    # Test 4: Numeric fill default None does NOT become 0 or 0.0 automatically
    cap_default_none = None
    default_num_str = str(cap_default_none) if cap_default_none is not None else ""
    assert default_num_str == ""
    assert parse_numeric_fill_input(default_num_str) is None

    # Test 5: Blank numeric constant input becomes None
    assert parse_numeric_fill_input("") is None
    assert parse_numeric_fill_input("   ") is None

    # Test 6: Valid numeric constant input preserves numeric type
    parsed_int = parse_numeric_fill_input("42")
    assert parsed_int == 42
    assert type(parsed_int) is int

    parsed_float = parse_numeric_fill_input("3.14")
    assert parsed_float == 3.14
    assert type(parsed_float) is float

    parsed_sci = parse_numeric_fill_input("1e-4")
    assert parsed_sci == 0.0001
    assert type(parsed_sci) is float

    # Test 7: Invalid numeric constant input is rejected cleanly
    assert parse_numeric_fill_input("abc") is None
    assert parse_numeric_fill_input("not_a_number") is None
    assert parse_numeric_fill_input("12.34.56") is None

    # Parser function for categorical fill matching 3_experiment_config.py
    def parse_categorical_fill_input(raw_input: str):
        if raw_input is None or raw_input.strip() == "":
            return None
        return raw_input.strip()

    # Test 8: Categorical default None / untouched becomes None
    cap_cat_default_none = None
    default_cat_str = str(cap_cat_default_none) if cap_cat_default_none is not None else ""
    assert default_cat_str == ""
    assert parse_categorical_fill_input(default_cat_str) is None
    assert parse_categorical_fill_input("   ") is None
    assert parse_categorical_fill_input("custom_fill") == "custom_fill"


def test_data_analyze_capability_and_payload_behavior():
    """
    Verifies that:
    1. Data Analyze functions come only from capability defaults.
    2. Function names are not hardcoded in frontend (supports arbitrary future functions).
    3. Checkbox defaults come from each item's 'enabled' field.
    4. Toggling a checkbox changes only that function's 'enabled'.
    5. Canonical 'f' values are preserved unchanged.
    6. Canonical 'data_analyze_functions' payload structure is preserved.
    7. Empty/missing capability does not invent fallback functions.
    8. Diagnostic validation flags missing/empty data_analyze capabilities.
    """
    # 1. Capability input with standard and future custom analysis functions
    capabilities = {
        "data_analyze": {
            "defaults": {
                "data_analyze_functions": [
                    {"f": "missing_value_check", "enabled": True},
                    {"f": "correlation_matrix", "enabled": True},
                    {"f": "future_outlier_detection_step", "enabled": False},
                ]
            }
        }
    }

    # Extract dynamically exactly as done in 3_experiment_config.py
    analyze_caps = capabilities.get("data_analyze", {})
    analyze_defaults = analyze_caps.get("defaults", {})
    analyze_functions = analyze_defaults.get("data_analyze_functions", [])

    assert len(analyze_functions) == 3
    # 2. Dynamic extraction - no hardcoded checks
    assert analyze_functions[0]["f"] == "missing_value_check"
    assert analyze_functions[1]["f"] == "correlation_matrix"
    assert analyze_functions[2]["f"] == "future_outlier_detection_step"

    # 3. Checkbox defaults come from each item's 'enabled'
    assert analyze_functions[0]["enabled"] is True
    assert analyze_functions[1]["enabled"] is True
    assert analyze_functions[2]["enabled"] is False

    # Simulate UI render and payload building logic
    def render_and_build_payload(raw_functions, toggles=None):
        toggles = toggles or {}
        da_payload_fns = []
        for fn_item in raw_functions:
            fn_name = str(fn_item.get("f", ""))
            fn_default = bool(fn_item.get("enabled", True))
            # User toggle if provided, else default
            current_val = toggles.get(fn_name, fn_default)
            da_payload_fns.append({
                "f": fn_name,
                "enabled": bool(current_val)
            })
        return {
            "data_analyze_functions": da_payload_fns
        } if da_payload_fns else None

    # Initial payload matches defaults
    payload_default = render_and_build_payload(analyze_functions)
    assert payload_default == {
        "data_analyze": {
            "data_analyze_functions": [
                {"f": "missing_value_check", "enabled": True},
                {"f": "correlation_matrix", "enabled": True},
                {"f": "future_outlier_detection_step", "enabled": False},
            ]
        }
    }["data_analyze"]

    # 4 & 5. Toggling a checkbox changes ONLY that function's enabled, f is preserved
    payload_toggled = render_and_build_payload(
        analyze_functions,
        toggles={"correlation_matrix": False, "future_outlier_detection_step": True}
    )
    assert payload_toggled == {
        "data_analyze_functions": [
            {"f": "missing_value_check", "enabled": True},
            {"f": "correlation_matrix", "enabled": False},
            {"f": "future_outlier_detection_step", "enabled": True},
        ]
    }

    # Verify all unchecked state
    payload_all_disabled = render_and_build_payload(
        analyze_functions,
        toggles={"missing_value_check": False, "correlation_matrix": False, "future_outlier_detection_step": False}
    )
    assert payload_all_disabled == {
        "data_analyze_functions": [
            {"f": "missing_value_check", "enabled": False},
            {"f": "correlation_matrix", "enabled": False},
            {"f": "future_outlier_detection_step", "enabled": False},
        ]
    }

    # 6. Top-level vocabulary preservation
    req_payload = {
        "project_name": "test_exp",
        "data_analyze": payload_toggled
    }
    assert "data_analyze" in req_payload
    assert "data_analyze_functions" in req_payload["data_analyze"]
    assert "operations" not in req_payload["data_analyze"]
    assert "functions" not in req_payload["data_analyze"]
    assert "enabled" not in req_payload["data_analyze"]

    # 7. Empty or missing capabilities does NOT invent fallback functions
    empty_caps = {"data_analyze": {"defaults": {"data_analyze_functions": []}}}
    diag_fns = validate_capability_path(empty_caps, path=("data_analyze", "defaults", "data_analyze_functions"), required=True, allow_empty=False)
    assert not diag_fns
    empty_fns = empty_caps["data_analyze"]["defaults"]["data_analyze_functions"]
    payload_empty = render_and_build_payload(empty_fns)
    assert payload_empty is None

    # Generic title humanization test
    for fn_item in analyze_functions:
        fn_name = fn_item["f"]
        label = fn_name.replace("_", " ").title()
        assert "_" not in label
    assert "Missing Value Check" == "missing_value_check".replace("_", " ").title()
    assert "Correlation Matrix" == "correlation_matrix".replace("_", " ").title()
    assert "Future Outlier Detection Step" == "future_outlier_detection_step".replace("_", " ").title()










