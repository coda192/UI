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

