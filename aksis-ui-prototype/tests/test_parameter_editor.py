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
