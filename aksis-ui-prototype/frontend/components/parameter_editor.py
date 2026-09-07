"""
AKSIS Platformu - Genel Model Parametre Düzenleyici Bileşeni (Generic Parameter Editor)
Bu bileşen tamamen API tarafından sağlanan 'parameter_schema' tiplerine göre dinamik arayüz üretir.
Kesinlikle modele özgü (XGBoost, SVC, CatBoost vb.) sabit bilgi içermez.
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_parameter_editor(
    param_schema: Optional[Dict[str, Any]],
    key_prefix: str = "custom_param"
) -> Dict[str, Any]:
    """
    Verilen parametre şemasına göre Streamlit giriş elemanlarını dinamik olarak oluşturur
    ve kullanıcının girdiği/seçtiği değerleri bir sözlük olarak döner.

    Desteklenen Parametre Tipleri:
    - int: Sayısal tam sayı girişi (min, max, step, default desteği)
    - float: Sayısal ondalıklı giriş (min, max, step, default desteği)
    - bool: Onay kutusu (checkbox)
    - str: Metin girişi (text_input)
    - choice: Seçim kutusu (selectbox - nullable desteğiyle)

    :param param_schema: API'den gelen { "param_name": { "type": "...", ... } } sözlüğü
    :param key_prefix: Streamlit session state anahtarları için benzersiz önek
    :return: { "param_name": value, ... } sözlüğü
    """
    overrides: Dict[str, Any] = {}

    if not param_schema or not isinstance(param_schema, dict):
        st.warning("Bu model için özelleştirilebilir parametre şeması sağlanmamış.")
        return overrides

    st.markdown("##### ⚙️ Model Hiperparametreleri (Custom Parameters)")
    st.caption("Aşağıdaki parametreler doğrudan API tarafından sunulan şemaya göre dinamik olarak listelenmektedir:")

    # Parametreleri iki sütunlu düzenli bir grid halinde render et
    param_items = list(param_schema.items())
    cols = st.columns(2)

    for idx, (param_name, spec) in enumerate(param_items):
        if not isinstance(spec, dict):
            continue

        target_col = cols[idx % 2]
        param_type = spec.get("type", "str")
        readable_label = param_name.replace("_", " ").title()
        help_text = spec.get("description")
        default_val = spec.get("default")
        widget_key = f"{key_prefix}_{param_name}"

        with target_col:
            if param_type == "int":
                min_val = spec.get("min")
                max_val = spec.get("max")
                step_val = spec.get("step", 1)

                val = st.number_input(
                    label=f"{readable_label} (`{param_name}`)",
                    value=int(default_val) if default_val is not None else 0,
                    min_value=int(min_val) if min_val is not None else None,
                    max_value=int(max_val) if max_val is not None else None,
                    step=int(step_val) if step_val is not None else 1,
                    help=help_text,
                    key=widget_key
                )
                overrides[param_name] = int(val)

            elif param_type == "float":
                min_val = spec.get("min")
                max_val = spec.get("max")
                step_val = spec.get("step", 0.01)

                format_str = "%.4f" if (step_val and float(step_val) < 0.01) else "%.2f"

                val = st.number_input(
                    label=f"{readable_label} (`{param_name}`)",
                    value=float(default_val) if default_val is not None else 0.0,
                    min_value=float(min_val) if min_val is not None else None,
                    max_value=float(max_val) if max_val is not None else None,
                    step=float(step_val) if step_val is not None else 0.01,
                    format=format_str,
                    help=help_text,
                    key=widget_key
                )
                overrides[param_name] = float(val)

            elif param_type == "bool":
                val = st.checkbox(
                    label=f"{readable_label} (`{param_name}`)",
                    value=bool(default_val) if default_val is not None else False,
                    help=help_text,
                    key=widget_key
                )
                overrides[param_name] = bool(val)

            elif param_type == "str":
                val = st.text_input(
                    label=f"{readable_label} (`{param_name}`)",
                    value=str(default_val) if default_val is not None else "",
                    help=help_text,
                    key=widget_key
                )
                overrides[param_name] = str(val)

            elif param_type == "choice":
                choices = list(spec.get("choices", []))
                
                # Default seçim indeksini belirle
                default_idx = 0
                if default_val in choices:
                    default_idx = choices.index(default_val)
                elif None in choices and default_val is None:
                    default_idx = choices.index(None)

                val = st.selectbox(
                    label=f"{readable_label} (`{param_name}`)",
                    options=choices,
                    index=default_idx,
                    format_func=lambda x: "None (Varsayılan yok)" if x is None else str(x),
                    help=help_text,
                    key=widget_key
                )
                overrides[param_name] = val

            else:
                # Bilinmeyen tipler için güvenli metin kutusu fallback'i
                val = st.text_input(
                    label=f"{readable_label} (`{param_name}`)",
                    value=str(default_val) if default_val is not None else "",
                    help=help_text,
                    key=widget_key
                )
                overrides[param_name] = val

    return overrides
