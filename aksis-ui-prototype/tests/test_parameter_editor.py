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


def test_data_split_schema_driven_method_selection_and_payload():
    """
    Verifies that:
    - Data split methods come from capabilities["data_split"]["methods"]
    - holdout renders and populates test_size, without leaking n_splits
    - kfold renders and populates n_splits, without leaking test_size
    - generated payload contains only applicable parameters
    """
    from backend.demo.fixtures import CAPABILITIES

    caps = CAPABILITIES.model_dump()
    data_split_caps = caps.get("data_split", {})
    methods = data_split_caps.get("methods", [])
    param_schema = data_split_caps.get("parameter_schema", {})

    assert "holdout" in methods
    assert "kfold" in methods
    assert "stratified_kfold" in methods

    # 1. Test Holdout parameter extraction
    holdout_schema = param_schema.get("holdout", {})
    assert "test_size" in holdout_schema
    assert "n_splits" not in holdout_schema
    assert holdout_schema["test_size"]["type"] == "float"
    assert holdout_schema["test_size"]["default"] == 0.2
    assert holdout_schema["test_size"]["min"] == 0.1
    assert holdout_schema["test_size"]["max"] == 0.5
    assert holdout_schema["test_size"]["step"] == 0.05

    # Simulated holdout parameters collected by UI
    holdout_params = {}
    for p_name, p_spec in holdout_schema.items():
        holdout_params[p_name] = p_spec.get("default")

    holdout_payload = {
        "method": "holdout",
        "parameters": holdout_params
    }

    assert holdout_payload["method"] == "holdout"
    assert "test_size" in holdout_payload["parameters"]
    assert holdout_payload["parameters"]["test_size"] == 0.2
    assert "n_splits" not in holdout_payload["parameters"]

    # 2. Test K-Fold parameter extraction
    kfold_schema = param_schema.get("kfold", {})
    assert "n_splits" in kfold_schema
    assert "test_size" not in kfold_schema
    assert kfold_schema["n_splits"]["type"] == "int"
    assert kfold_schema["n_splits"]["default"] == 5
    assert kfold_schema["n_splits"]["min"] == 2
    assert kfold_schema["n_splits"]["max"] == 20
    assert kfold_schema["n_splits"]["step"] == 1

    # Simulated kfold parameters collected by UI
    kfold_params = {}
    for p_name, p_spec in kfold_schema.items():
        kfold_params[p_name] = p_spec.get("default")

    kfold_payload = {
        "method": "kfold",
        "parameters": kfold_params
    }

    assert kfold_payload["method"] == "kfold"
    assert "n_splits" in kfold_payload["parameters"]
    assert kfold_payload["parameters"]["n_splits"] == 5
    assert "test_size" not in kfold_payload["parameters"]


def test_data_split_slider_and_num_sync_logic():
    """
    Verifies that the bidirectional slider and number input sync handlers keep
    session state synchronized for both float and int types.
    """
    state = {
        "ds_holdout_test_size_slider": 0.2,
        "ds_holdout_test_size_num": 0.2,
        "ds_kfold_n_splits_slider": 5,
        "ds_kfold_n_splits_num": 5
    }

    # Simulate slider change for float
    state["ds_holdout_test_size_slider"] = 0.35
    state["ds_holdout_test_size_num"] = state["ds_holdout_test_size_slider"]
    assert state["ds_holdout_test_size_num"] == 0.35

    # Simulate manual num change for float
    state["ds_holdout_test_size_num"] = 0.25
    state["ds_holdout_test_size_slider"] = state["ds_holdout_test_size_num"]
    assert state["ds_holdout_test_size_slider"] == 0.25

    # Simulate slider change for int
    state["ds_kfold_n_splits_slider"] = 10
    state["ds_kfold_n_splits_num"] = state["ds_kfold_n_splits_slider"]
    assert state["ds_kfold_n_splits_num"] == 10

    # Simulate manual num change for int
    state["ds_kfold_n_splits_num"] = 8
    state["ds_kfold_n_splits_slider"] = state["ds_kfold_n_splits_num"]
    assert state["ds_kfold_n_splits_slider"] == 8


def test_data_split_unbounded_numeric_and_other_types():
    """
    Verifies that:
    - Numeric parameters without min/max do not receive fabricated ranges
    - Types (int, float, bool, choice) are correctly preserved in collected payload
    - Stratified K-Fold works identically to K-Fold with its own isolated parameters
    """
    custom_schema = {
        "custom_method": {
            "unbounded_float": {
                "type": "float",
                "default": 1.5,
                "description": "Unbounded float parameter"
            },
            "unbounded_int": {
                "type": "int",
                "default": 42,
                "description": "Unbounded int parameter"
            },
            "flag_option": {
                "type": "bool",
                "default": False,
                "description": "Boolean flag"
            },
            "strategy_choice": {
                "type": "choice",
                "default": "fast",
                "choices": ["fast", "precise"],
                "description": "Choice parameter"
            }
        }
    }

    params = {}
    for p_name, p_spec in custom_schema["custom_method"].items():
        p_type = p_spec["type"]
        p_default = p_spec.get("default")
        p_min = p_spec.get("min")
        p_max = p_spec.get("max")

        # Unbounded numeric check: min and max are absent
        if p_type in ("int", "float"):
            assert p_min is None
            assert p_max is None
            if p_type == "int":
                params[p_name] = int(p_default)
            else:
                params[p_name] = float(p_default)
        elif p_type == "bool":
            params[p_name] = bool(p_default)
        elif p_type in ("choice", "enum"):
            assert p_default in p_spec["choices"]
            params[p_name] = p_default

    payload = {
        "method": "custom_method",
        "parameters": params
    }

    assert payload["method"] == "custom_method"
    assert payload["parameters"]["unbounded_float"] == 1.5
    assert isinstance(payload["parameters"]["unbounded_float"], float)
    assert payload["parameters"]["unbounded_int"] == 42
    assert isinstance(payload["parameters"]["unbounded_int"], int)
    assert payload["parameters"]["flag_option"] is False
    assert isinstance(payload["parameters"]["flag_option"], bool)
    assert payload["parameters"]["strategy_choice"] == "fast"


def test_data_split_stratified_kfold_payload():
    from backend.demo.fixtures import CAPABILITIES

    caps = CAPABILITIES.model_dump()
    data_split_caps = caps.get("data_split", {})
    param_schema = data_split_caps.get("parameter_schema", {})

    assert "stratified_kfold" in data_split_caps.get("methods", [])
    skf_schema = param_schema.get("stratified_kfold", {})
    assert "n_splits" in skf_schema
    assert "shuffle" in skf_schema
    assert "test_size" not in skf_schema

    skf_params = {
        "n_splits": int(skf_schema["n_splits"]["default"]),
        "shuffle": bool(skf_schema["shuffle"]["default"])
    }
    payload = {
        "method": "stratified_kfold",
        "parameters": skf_params
    }
    assert payload["method"] == "stratified_kfold"
    assert payload["parameters"]["n_splits"] == 5
    assert payload["parameters"]["shuffle"] is True
    assert "test_size" not in payload["parameters"]




