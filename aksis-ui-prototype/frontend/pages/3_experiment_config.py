import streamlit as st
from frontend.api.client import AksisAPIError
from frontend.utils.model_guidance import (
    get_algorithm_display_name,
    get_algorithm_metadata,
    is_config_applicable,
    resolve_task_learning_types,
    validate_capability_path
)
from frontend.components.parameter_editor import render_parameter_editor

# Sayfa Başlığı ve Açıklama
st.markdown("""
<div style="margin-bottom: 1.25rem;">
    <h1 style="margin-bottom: 0.25rem;">⚙️ Deney Yapılandırma Stüdyosu</h1>
    <p style="color: #64748b; font-size: 1.0rem; margin: 0;">
        AKSIS Makine Öğrenmesi Platformu — Veri seti, görev tipi, model mimarisi, önişleme boru hattı ve doğrulama stratejisini yapılandırın.
    </p>
</div>
""", unsafe_allow_html=True)

client = st.session_state.client

# ==============================================================================
# SUNUM ETİKETLERİ VE GÖRÜNTÜLEME EŞLEŞTİRMELERİ (PRESENTATION ALIASES)
# ==============================================================================
TASK_NAMES_TR = {
    "classification": "Sınıflandırma",
    "regression": "Regresyon",
    "anomaly_detection": "Anomali Tespiti"
}

LEARNING_TYPES_TR = {
    "supervised": "Denetimli Öğrenme",
    "unsupervised": "Denetimsiz Öğrenme"
}

MODE_NAMES_TR = {
    "train": "Model Eğitimi (train)",
    "tune": "Hiperparametre Optimizasyonu (tune)",
    "predict": "Toplu Çıkarım / Tahmin (predict)"
}

PRESET_NAMES_TR = {
    "baseline": "Temel Seviye (baseline)",
    "fast": "Hızlı (fast)",
    "strong": "Yüksek Başarım (strong)",
    "custom": "Özel Parametre Ayarı (custom) — Geçici Olarak Kapalı",
    "interpretable": "Yorumlanabilir (interpretable)",
    "accurate": "Yüksek Doğruluk (accurate)",
    "default": "Varsayılan (default)"
}

PREP_TR = {
    "none": "Uygulanmasın (none)",
    "mean": "Ortalama Değer (mean)",
    "median": "Medyan (median)",
    "most_frequent": "En Sık Tekrar Eden (most_frequent)",
    "constant": "Sabit Değer (constant)",
    "drop": "Eksik Satırları Çıkar (drop)",
    "onehot": "One-Hot Encoding (onehot)",
    "frequency": "Frekans Kodlama (frequency)",
    "hashing": "Hashing Encoding (hashing)",
    "label": "Label Encoding (label)",
    "target": "Target Encoding (target)",
    "standard": "StandardScaler (standard)",
    "minmax": "MinMaxScaler (minmax)",
    "robust": "RobustScaler (robust)"
}

VAL_TR = {
    "holdout": "Ayrık Test Kümesi (Holdout / Train-Test)",
    "kfold": "K-Fold Çapraz Doğrulama (K-Fold CV)",
    "stratified_kfold": "Tabakalı K-Fold (Stratified K-Fold CV)"
}

TUNING_SAMPLER_TR = {
    "tpe": "TPE (Tree-structured Parzen Estimator)",
    "random": "Rastgele Arama (Random Search)",
    "grid": "Grid Search"
}

TUNING_PRUNER_TR = {
    "none": "Budama Yapılmasın (none)",
    "median": "Medyan Budayıcı (median)",
    "sha": "Successive Halving (sha)",
    "hyperband": "Hyperband"
}

PARAM_HELP_TR = {
    "n_features": "Özellik uzayının eşleneceği hash boyutu (çıktı sütun sayısı). Yüksek kardinalite için 1000 - 1000000 aralığında seçilebilir.",
    "n_hash_features": "Özellik uzayının eşleneceği hash boyutu (çıktı sütun sayısı).",
    "handle_unknown": "Eğitim kümesinde yer almayan yeni kategorilerle karşılaşıldığında uygulanacak strateji.",
    "drop_invariant": "Tüm satırlarda tek bir sabit değere sahip varyanssız sütunların otomatik elenmesi.",
    "with_mean": "Verilerin ortalamasını sıfıra merkezleme. Seyrek matrislerde bellek tasarrufu için False seçilmelidir.",
    "with_std": "Verileri birim varyansa (standart sapma 1) ölçeklendirme.",
    "feature_range": "Özelliklerin dönüştürüleceği min-max değer aralığı.",
    "quantile_range": "Aykırı değerlere karşı dayanıklılık için kullanılan yüzdelik dilim aralığı (IQR).",
    "random_state": "Rastgele sayı üreteci tohumu (deneyin tekrarlanabilirliği için sabit tam sayı).",
    "test_size": "Veri kümesinin test/doğrulama için ayrılacak oranı (0.0 - 1.0 arası).",
    "shuffle": "Bölümleme öncesinde veri satırlarını rastgele karıştır.",
    "n_splits": "K-Fold çapraz doğrulama için katlama (split) sayısı."
}

# ==============================================================================
# 1. CAPABILITIES VE DATASETS YÜKLEME (API FLOW)
# ==============================================================================
try:
    capabilities = client.get_capabilities()
except (AksisAPIError, Exception) as e:
    st.error(f"⚠️ Sistem yetenekleri (capabilities) backend servisinden yüklenemedi: {str(e)}")
    st.info("Lütfen backend API servisinin çalıştığından emin olun.")
    st.stop()

if not capabilities:
    st.error("⚠️ Boş sistem yetenek verisi alındı.")
    st.stop()

try:
    datasets = client.get_datasets()
except (AksisAPIError, Exception) as e:
    st.error(f"⚠️ Veri setleri yüklenemedi: {str(e)}")
    st.stop()

if not datasets:
    st.warning("Kullanılabilir veri seti bulunamadı. Lütfen önce veri seti tanımlayın.")
    st.stop()

# ==============================================================================
# 2. VERİ SETİ SEÇİMİ VE GÖREV ÇÖZÜMLEME (DATASET -> LEARNING TYPE -> TASK FLOW)
# ==============================================================================
dataset_options = {d["id"]: f"{d.get('display_name')} ({d['name']})" if d.get("display_name") else d["name"] for d in datasets}

default_ds_idx = 0
if "selected_dataset_id" in st.session_state and st.session_state.selected_dataset_id in dataset_options:
    default_ds_idx = list(dataset_options.keys()).index(st.session_state.selected_dataset_id)

with st.container(border=True):
    st.markdown("### 📁 1. Veri Seti & Çalışma Kapsamı")
    st.caption("Deneyin yürütüleceği veri kaynağını, adlandırmasını ve temel makine öğrenmesi görevini belirleyin.")

    c_ds1, c_ds2 = st.columns([1, 1])
    with c_ds1:
        selected_dataset_id = st.selectbox(
            "Kullanılacak Veri Seti (Dataset)", 
            options=list(dataset_options.keys()), 
            format_func=lambda x: dataset_options[x], 
            index=default_ds_idx,
            key="exp_dataset_select"
        )
        st.session_state.selected_dataset_id = selected_dataset_id

    selected_ds_meta = next(d for d in datasets if d["id"] == selected_dataset_id)
    compatible_tasks = selected_ds_meta.get("compatible_tasks") or []

    if not compatible_tasks:
        st.error(f"Seçilen veri seti ('{selected_ds_meta.get('name')}') için DataSpec üzerinde uyumlu bir görev tanımlanmamış.")
        st.stop()

    # AKSIS capabilities["config_applicability"] ve ["tasks"] üzerinden uyumlu görev ve öğrenme türlerini dinamik çözümle
    valid_task_to_lt = resolve_task_learning_types(capabilities, compatible_tasks)

    valid_tasks = list(valid_task_to_lt.keys())
    valid_learning_types = list(dict.fromkeys(valid_task_to_lt.values()))

    if not valid_tasks:
        supported_aksis_tasks = capabilities.get("tasks", [])
        st.error(
            f"Veri setinin uyumlu görevleri ({compatible_tasks}) ile AKSIS çerçevesinin desteklediği görevler ({supported_aksis_tasks}) arasında eşleşme bulunamadı. "
            "Veri seti meta verisi ile mevcut AKSIS yetenekleri birbiriyle uyumsuz."
        )
        st.stop()

    with c_ds2:
        default_exp_name = f"Deney_{selected_ds_meta['name'].replace('.csv', '')}"
        exp_name = st.text_input("Deney Adı (Project Name)", value=default_exp_name)

    # Parametre Seçimleri: Learning Type, Task, Execution Mode
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        if len(valid_learning_types) == 1:
            learning_type = valid_learning_types[0]
            st.caption("Öğrenme Türü (Learning Type)")
            st.markdown(f"**{LEARNING_TYPES_TR.get(learning_type, learning_type)}**")
        else:
            learning_type = st.selectbox(
                "Öğrenme Türü (Learning Type)",
                options=valid_learning_types,
                format_func=lambda x: LEARNING_TYPES_TR.get(x, x)
            )

    with col_info2:
        active_valid_tasks = [t for t, lt in valid_task_to_lt.items() if lt == learning_type]
        if len(active_valid_tasks) == 1:
            task = active_valid_tasks[0]
            st.caption("Görev (Task)")
            st.markdown(f"**{TASK_NAMES_TR.get(task, task)}**")
        else:
            task = st.selectbox(
                "Görev (Task)",
                options=active_valid_tasks,
                format_func=lambda x: TASK_NAMES_TR.get(x, x)
            )

    with col_info3:
        api_modes = capabilities.get("modes", [])
        workflow_modes = [m for m in api_modes if m in ("train", "tune")]
        if not workflow_modes:
            st.warning("Bu iş akışı için geçerli bir çalıştırma modu bulunamadı.")
            selected_mode = None
        else:
            selected_mode = st.selectbox(
                "Çalışma Modu (Execution Mode)",
                options=workflow_modes,
                format_func=lambda m: MODE_NAMES_TR.get(m, m),
                help="'train': Standart model eğitimi | 'tune': Optuna ile otomatik hiperparametre optimizasyonu"
            )

# ==============================================================================
# 3. MODEL VE ALGORİTMA SEÇİMİ (Model & Algorithm)
# ==============================================================================
custom_overrides = {}

with st.container(border=True):
    st.markdown("### 🤖 2. Model ve Algoritma Seçimi")
    st.caption("Göreviniz için AKSIS model kataloğundan uygun algoritmayı ve konfigürasyon hazır ayarını seçin.")

    if task:
        available_algos = capabilities.get("algorithms", {}).get(task, [])
        presets_dict = capabilities.get("model_presets", {})
        if isinstance(presets_dict, dict):
            api_presets = presets_dict.get(learning_type, [])
        elif isinstance(presets_dict, list):
            api_presets = presets_dict
        else:
            api_presets = []
        
        if not available_algos:
            st.warning(f"'{task}' görevi için AKSIS yeteneklerinde kayıtlı algoritma bulunamadı.")
            algorithm = None
            preset = None
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                algorithm = st.selectbox(
                    "Algoritma (Algorithm)", 
                    options=available_algos,
                    format_func=lambda x: get_algorithm_display_name(x, capabilities),
                    help="AKSIS Registry tarafından sunulan algoritmalar",
                    key="model_algo_select"
                )
            with col_m2:
                if api_presets:
                    preset = st.selectbox(
                        "Model Hazır Ayarı (Model Preset)", 
                        options=api_presets,
                        format_func=lambda x: PRESET_NAMES_TR.get(x, x),
                        key="model_preset_select"
                    )
                else:
                    preset = None
                
            # Uyumsuz session state parametrelerini temizle (State cleanup)
            current_model_context = f"{task}_{algorithm}_{preset}"
            if st.session_state.get("last_model_context") != current_model_context:
                for k in list(st.session_state.keys()):
                    if k.startswith("custom_param_"):
                        del st.session_state[k]
                st.session_state.last_model_context = current_model_context

            # Seçilen model için karar destek rehberi (Decision Support Card)
            if algorithm:
                algo_meta = get_algorithm_metadata(algorithm, capabilities)
                display_title = algo_meta.get("display_name") or algorithm
                description = algo_meta.get("description", "Model açıklaması sağlanmamış.")
                strengths = algo_meta.get("strengths", [])
                limitations = algo_meta.get("limitations", [])
                best_for = algo_meta.get("best_for", [])
                
                with st.container(border=True):
                    st.markdown(f"#### ⚡ Model Rehberi: {display_title}")
                    if display_title != algorithm:
                        st.caption(f"Sistem Kayıt Anahtarı: `{algorithm}`")
                    st.write(description)
                    
                    c_guide1, c_guide2, c_guide3 = st.columns(3)
                    with c_guide1:
                        st.markdown("**🎯 En Uygun Senaryolar**")
                        if best_for:
                            items = best_for if isinstance(best_for, list) else [best_for]
                            for item in items:
                                st.markdown(f"- {item}")
                        else:
                            st.caption("Belirtilmemiş")
                    with c_guide2:
                        st.markdown("**✅ Güçlü Yönler**")
                        if strengths:
                            items = strengths if isinstance(strengths, list) else [strengths]
                            for item in items:
                                st.markdown(f"- {item}")
                        else:
                            st.caption("Belirtilmemiş")
                    with c_guide3:
                        st.markdown("**⚠️ Dikkat / Kısıtlar**")
                        if limitations:
                            items = limitations if isinstance(limitations, list) else [limitations]
                            for item in items:
                                st.markdown(f"- {item}")
                        else:
                            st.caption("Belirtilmemiş")

            # ÖZEL MODEL PARAMETRELERİ (Yalnızca preset == "custom" İken Açılır)
            if preset == "custom" and algorithm:
                st.divider()
                st.info("Custom model parameter configuration is temporarily disabled for the demo.")
    else:
        algorithm = None
        preset = None
        st.info("Lütfen önce geçerli bir görev seçin.")

# ==============================================================================
# 4. ÖNİŞLEME BORU HATTI (Preprocessing Pipeline - Açılır/Kapanır)
# ==============================================================================
preprocess_payload = None

if learning_type and task and is_config_applicable(capabilities, "preprocess", learning_type, task):
    with st.expander("🛠️ 3. Önişleme Boru Hattı (Preprocessing Pipeline)", expanded=False):
        st.caption("Verinizi model eğitimine hazırlamak için eksik değer tamamlama, kategorik kodlama ve sayısal ölçeklendirme adımlarını yapılandırın.")

        prep_caps = capabilities.get("preprocessing", {}) if isinstance(capabilities, dict) else {}
        prep_defaults = prep_caps.get("defaults", {}) if isinstance(prep_caps, dict) else {}

        encoding_methods = prep_caps.get("encoding_methods", []) if isinstance(prep_caps, dict) else []
        scaling_methods = prep_caps.get("scaling_methods", []) if isinstance(prep_caps, dict) else []

        # Diagnostics validation
        diag_prep = validate_capability_path(capabilities, path=("preprocessing",), required=True, allow_empty=False)
        diag_defaults = validate_capability_path(capabilities, path=("preprocessing", "defaults"), required=True, allow_empty=False)

        if not diag_prep:
            st.warning(f"Önişleme yetenek tanımı eksik: {diag_prep}")
        elif not diag_defaults:
            st.warning(f"Önişleme varsayılanları eksik: {diag_defaults}")

        tab_missing, tab_encoding, tab_scaling = st.tabs([
            "🧩 Eksik Değer Yönetimi (Imputation)",
            "🔤 Kategorik Kodlama (Encoding)",
            "⚖️ Sayısal Ölçeklendirme (Scaling)"
        ])

        # Tab 1: Eksik Değer Yönetimi
        with tab_missing:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                ui_handle_missing = st.checkbox(
                    "Eksik Değerleri Yönet (handle_missing)",
                    value=bool(prep_defaults.get("handle_missing", True)),
                    key="prep_handle_missing"
                )
                ui_num_imputer = st.text_input(
                    "Sayısal Değer Tamamlama (numeric_imputer)",
                    value=str(prep_defaults.get("numeric_imputer", "") or ""),
                    key="prep_numeric_imputer"
                )
                num_fill_def = prep_defaults.get("numeric_fill_value")
                if num_fill_def is not None:
                    ui_num_fill = st.number_input(
                        "Sayısal Doldurma Değeri (numeric_fill_value)",
                        value=float(num_fill_def),
                        key="prep_numeric_fill_value"
                    )
                else:
                    ui_num_fill = None

            with c_m2:
                ui_cat_imputer = st.text_input(
                    "Kategorik Değer Tamamlama (categorical_imputer)",
                    value=str(prep_defaults.get("categorical_imputer", "") or ""),
                    key="prep_categorical_imputer"
                )
                ui_cat_fill = st.text_input(
                    "Kategorik Doldurma Değeri (categorical_fill_value)",
                    value=str(prep_defaults.get("categorical_fill_value", "") or ""),
                    key="prep_categorical_fill_value"
                )

        # Tab 2: Kategorik Kodlama
        with tab_encoding:
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                ui_encoding = st.checkbox(
                    "Kategorik Kodlama Etkin (encoding)",
                    value=bool(prep_defaults.get("encoding", False)),
                    key="prep_encoding"
                )

                def_encode_method = prep_defaults.get("encode_method")
                if encoding_methods:
                    def_enc_idx = encoding_methods.index(def_encode_method) if def_encode_method in encoding_methods else 0
                    ui_encode_method = st.selectbox(
                        "Kodlama Yöntemi (encode_method)",
                        options=encoding_methods,
                        index=def_enc_idx,
                        key="prep_encode_method"
                    )
                else:
                    if ui_encoding:
                        diag_enc = validate_capability_path(capabilities, path=("preprocessing", "encoding_methods"), required=True, allow_empty=False)
                        st.warning(f"Kodlama yöntemleri bulunamadı: {diag_enc}")
                    ui_encode_method = def_encode_method

                ui_target_encoding = st.checkbox(
                    "Hedef Değişken Kodlama (targetEncoding)",
                    value=bool(prep_defaults.get("targetEncoding", False)),
                    key="prep_target_encoding"
                )

                if ui_encode_method == "hashing":
                    st.caption("💡 **Bilgi:** Hashing kodlama, yüksek kardinaliteli kategorik değişkenleri sabit boyutlu bir öznitelik uzayına indirger.")

            with c_e2:
                st.markdown("**Kodlama Parametreleri (`encoding_parameters`)**")
                ui_encoding_params = {}
                raw_enc_params = prep_defaults.get("encoding_parameters", {})
                
                # method-specific nested dictionary desteği: encoding_parameters[encode_method]
                if isinstance(raw_enc_params, dict):
                    if ui_encode_method and isinstance(raw_enc_params.get(ui_encode_method), dict):
                        selected_enc_params = raw_enc_params.get(ui_encode_method, {})
                    elif any(isinstance(v, dict) for v in raw_enc_params.values()):
                        selected_enc_params = {}
                    else:
                        selected_enc_params = raw_enc_params

                    if not selected_enc_params:
                        st.caption("Bu yöntem için yapılandırılabilir ek parametre bulunmamaktadır.")

                    for p_k, p_v in selected_enc_params.items():
                        p_label = p_k.replace("_", " ").title()
                        field_key = f"prep_enc_{ui_encode_method}_{p_k}"
                        help_text = PARAM_HELP_TR.get(p_k)
                        if isinstance(p_v, bool):
                            ui_encoding_params[p_k] = st.checkbox(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                key=f"{field_key}_bool",
                                help=help_text
                            )
                        elif isinstance(p_v, int):
                            ui_encoding_params[p_k] = st.number_input(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                step=1,
                                key=f"{field_key}_int",
                                help=help_text
                            )
                        elif isinstance(p_v, float):
                            ui_encoding_params[p_k] = st.number_input(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                step=0.01 if p_v < 1.0 else 1.0,
                                format="%.4f" if p_v < 0.01 else "%.2f",
                                key=f"{field_key}_float",
                                help=help_text
                            )
                        elif isinstance(p_v, str):
                            ui_encoding_params[p_k] = st.text_input(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                key=f"{field_key}_str",
                                help=help_text
                            )
                        else:
                            ui_encoding_params[p_k] = p_v

        # Tab 3: Sayısal Ölçeklendirme
        with tab_scaling:
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                ui_scaling = st.checkbox(
                    "Sayısal Ölçeklendirme Etkin (scaling)",
                    value=bool(prep_defaults.get("scaling", False)),
                    key="prep_scaling"
                )

                def_scale_method = prep_defaults.get("scale_method")
                if scaling_methods:
                    def_scale_idx = scaling_methods.index(def_scale_method) if def_scale_method in scaling_methods else 0
                    ui_scale_method = st.selectbox(
                        "Ölçeklendirme Yöntemi (scale_method)",
                        options=scaling_methods,
                        index=def_scale_idx,
                        key="prep_scale_method"
                    )
                else:
                    if ui_scaling:
                        diag_scale = validate_capability_path(capabilities, path=("preprocessing", "scaling_methods"), required=True, allow_empty=False)
                        st.warning(f"Ölçeklendirme yöntemleri bulunamadı: {diag_scale}")
                    ui_scale_method = def_scale_method

            with c_s2:
                st.markdown("**Ölçeklendirme Parametreleri (`scaling_parameters`)**")
                ui_scaling_params = {}
                raw_scale_params = prep_defaults.get("scaling_parameters", {})
                
                # method-specific nested dictionary desteği: scaling_parameters[scale_method]
                if isinstance(raw_scale_params, dict):
                    if ui_scale_method and isinstance(raw_scale_params.get(ui_scale_method), dict):
                        selected_scale_params = raw_scale_params.get(ui_scale_method, {})
                    elif any(isinstance(v, dict) for v in raw_scale_params.values()):
                        selected_scale_params = {}
                    else:
                        selected_scale_params = raw_scale_params

                    if not selected_scale_params:
                        st.caption("Bu yöntem için yapılandırılabilir ek parametre bulunmamaktadır.")

                    # KURAL: scaling parametrelerinde with_mean ve with_std aynı anda seçilemez
                    has_mutual_std_mean = (
                        "with_mean" in selected_scale_params and "with_std" in selected_scale_params
                    )

                    if has_mutual_std_mean:
                        st.caption("⚠️ **Kural:** `with_mean` ve `with_std` parametreleri aynı anda seçilemez. Lütfen birini tercih edin:")
                        # Varsayılan seçim: with_std True ise with_std, aksi halde with_mean, aksi halde none
                        def_choice = "with_std" if selected_scale_params.get("with_std") else ("with_mean" if selected_scale_params.get("with_mean") else "none")
                        radio_opts = ["with_std", "with_mean", "none"]
                        radio_idx = radio_opts.index(def_choice) if def_choice in radio_opts else 0

                        scale_mutual_choice = st.radio(
                            "Standart Ölçeklendirme Parametresi",
                            options=radio_opts,
                            index=radio_idx,
                            format_func=lambda x: {
                                "with_std": "Birim Varyans Ölçeklendirme (`with_std`) — Önerilen",
                                "with_mean": "Ortalama Merkezileştirme (`with_mean`)",
                                "none": "Hiçbiri (none)"
                            }.get(x, x),
                            key=f"prep_scale_{ui_scale_method}_mutual_radio"
                        )
                        ui_scaling_params["with_mean"] = bool(scale_mutual_choice == "with_mean")
                        ui_scaling_params["with_std"] = bool(scale_mutual_choice == "with_std")

                    for p_k, p_v in selected_scale_params.items():
                        if has_mutual_std_mean and p_k in ("with_mean", "with_std"):
                            continue  # Radyo seçimi tarafından yönetildi

                        p_label = p_k.replace("_", " ").title()
                        field_key = f"prep_scale_{ui_scale_method}_{p_k}"
                        help_text = PARAM_HELP_TR.get(p_k)
                        if isinstance(p_v, bool):
                            ui_scaling_params[p_k] = st.checkbox(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                key=f"{field_key}_bool",
                                help=help_text
                            )
                        elif isinstance(p_v, int):
                            ui_scaling_params[p_k] = st.number_input(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                step=1,
                                key=f"{field_key}_int",
                                help=help_text
                            )
                        elif isinstance(p_v, float):
                            ui_scaling_params[p_k] = st.number_input(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                step=0.01 if p_v < 1.0 else 1.0,
                                format="%.4f" if p_v < 0.01 else "%.2f",
                                key=f"{field_key}_float",
                                help=help_text
                            )
                        elif isinstance(p_v, str):
                            ui_scaling_params[p_k] = st.text_input(
                                f"{p_label} (`{p_k}`)",
                                value=p_v,
                                key=f"{field_key}_str",
                                help=help_text
                            )
                        else:
                            ui_scaling_params[p_k] = p_v

        # Check if at least one preprocessing feature is enabled / configured
        has_prep_configured = (
            ui_handle_missing
            or bool(ui_numeric_imputer)
            or bool(ui_categorical_imputer)
            or ui_encoding
            or ui_target_encoding
            or ui_scaling
        )

        if has_prep_configured:
            preprocess_payload = {
                "handle_missing": bool(ui_handle_missing),
                "numeric_imputer": ui_num_imputer if ui_num_imputer else None,
                "categorical_imputer": ui_cat_imputer if ui_cat_imputer else None,
                "numeric_fill_value": ui_num_fill,
                "categorical_fill_value": ui_cat_fill if ui_cat_fill else None,
                "encoding": bool(ui_encoding),
                "encode_method": ui_encode_method if ui_encoding else None,
                "targetEncoding": bool(ui_target_encoding),
                "encoding_parameters": ui_encoding_params,
                "scaling": bool(ui_scaling),
                "scale_method": ui_scale_method if ui_scaling else None,
                "scaling_parameters": ui_scaling_params
            }

# ==============================================================================
# 5. VERİ BÖLÜMLEME VE DOĞRULAMA (Data Splitting & Validation - Açılır/Kapanır)
# ==============================================================================
val_strategy = None
data_split_params = {}

if learning_type and task and is_config_applicable(capabilities, "data_split", learning_type, task):
    with st.expander("📐 4. Veri Bölümleme & Çapraz Doğrulama", expanded=False):
        st.caption("Modelin değerlendirilmesi için eğitim-test kümesi ayrımı veya K-Fold stratejisini belirleyin.")

        split_meta = capabilities.get("data_split", {}) if isinstance(capabilities, dict) else {}
        val_options = split_meta.get("methods", []) if isinstance(split_meta, dict) else []
        split_params = split_meta.get("parameters", {}) if isinstance(split_meta, dict) else {}

        ds_diag = validate_capability_path(capabilities, path=("data_split", "methods"), required=True, allow_empty=False)
        if val_options:
            val_strategy = st.selectbox(
                "Doğrulama / Bölümleme Yöntemi", 
                val_options,
                format_func=lambda x: VAL_TR.get(x, x),
                key="data_split_method_select"
            )
        else:
            st.warning(f"Doğrulama yöntemi bulunamadı: {ds_diag}")
            val_strategy = None

        if val_strategy and isinstance(split_params, dict):
            for p_name, p_default in split_params.items():
                p_label = p_name.replace("_", " ").title()
                field_key = f"ds_{val_strategy}_{p_name}"
                help_text = PARAM_HELP_TR.get(p_name)

                # 1. Special-case: test_size -> slider + synchronized metric ratio
                if p_name == "test_size":
                    def_val = float(p_default) if p_default is not None else 0.2
                    col_ts1, col_ts2 = st.columns([3, 1])
                    with col_ts1:
                        val = st.slider(
                            label=f"{p_label} (`{p_name}`)",
                            min_value=0.0,
                            max_value=1.0,
                            value=def_val,
                            step=0.05,
                            key=f"{field_key}_slider",
                            help=help_text
                        )
                    with col_ts2:
                        train_pct = int(round((1.0 - val) * 100))
                        test_pct = int(round(val * 100))
                        st.metric(
                            label="Eğitim / Test",
                            value=f"%{train_pct} / %{test_pct}"
                        )
                    data_split_params[p_name] = float(val)

                # 2. Special-case: random_state -> text_input without stepper
                elif p_name == "random_state":
                    str_default = str(p_default) if p_default is not None else "42"
                    val_str = st.text_input(
                        label=f"{p_label} (`{p_name}`)",
                        value=str_default,
                        key=f"{field_key}_text",
                        help=help_text
                    )
                    try:
                        data_split_params[p_name] = int(val_str.strip())
                    except (ValueError, TypeError):
                        st.error(f"Geçersiz `{p_name}` değeri: '{val_str}'. Lütfen geçerli bir tam sayı girin.")
                        data_split_params[p_name] = val_str

                # 3. Special-case: shuffle -> checkbox (or any bool)
                elif isinstance(p_default, bool):
                    val = st.checkbox(
                        label=f"{p_label} (`{p_name}`)",
                        value=p_default,
                        key=f"{field_key}_bool",
                        help=help_text
                    )
                    data_split_params[p_name] = bool(val)

                # 4. Generic fallback: int
                elif isinstance(p_default, int):
                    val = st.number_input(
                        label=f"{p_label} (`{p_name}`)",
                        value=p_default,
                        step=1,
                        key=f"{field_key}_int",
                        help=help_text
                    )
                    data_split_params[p_name] = int(val)

                # 5. Generic fallback: float
                elif isinstance(p_default, float):
                    val = st.number_input(
                        label=f"{p_label} (`{p_name}`)",
                        value=p_default,
                        step=0.01 if p_default < 1.0 else 1.0,
                        format="%.4f" if p_default < 0.01 else "%.2f",
                        key=f"{field_key}_float",
                        help=help_text
                    )
                    data_split_params[p_name] = float(val)

                # 6. Generic fallback: str
                elif isinstance(p_default, str):
                    val = st.text_input(
                        label=f"{p_label} (`{p_name}`)",
                        value=p_default,
                        key=f"{field_key}_str",
                        help=help_text
                    )
                    data_split_params[p_name] = str(val)

                # 7. Other unknown types
                else:
                    data_split_params[p_name] = p_default

# Data split konfigürasyonu
data_split_payload = None
if val_strategy:
    data_split_payload = {
        "method": val_strategy,
        "parameters": data_split_params
    }

# ==============================================================================
# 6. HİPERPARAMETRE OPTİMİZASYONU (Tuning - Yalnızca Mode='tune' İken Açılır/Kapanır)
# ==============================================================================
tuning_sampler = None
tuning_pruner = None
tuning_trials = 10
tuning_scoring = None

tuning_applicable = (
    learning_type is not None 
    and task is not None 
    and is_config_applicable(capabilities, f"{learning_type}_tuning", learning_type, task)
)

if selected_mode == "tune" and tuning_applicable:
    with st.expander("⚡ 5. Hiperparametre Optimizasyonu (Optuna Hyperparameter Tuning)", expanded=True):
        st.caption("Mode='tune' seçildiği için Optuna arama uzayı, örnekleyici ve deneme parametreleri aktif edildi.")
        
        tuning_caps = capabilities.get("tuning") or {}
        if hasattr(tuning_caps, "model_dump"):
            tuning_caps = tuning_caps.model_dump()
        elif hasattr(tuning_caps, "dict"):
            tuning_caps = tuning_caps.dict()
        elif not isinstance(tuning_caps, dict):
            tuning_caps = dict(tuning_caps)
        
        samplers = (
            tuning_caps
            .get("samplers", {})
            .get(learning_type, [])
        )
        pruners = (
            tuning_caps
            .get("pruners", {})
            .get(learning_type, [])
        )
        scoring_opts = (
            tuning_caps
            .get("objectives", {})
            .get(task, [])
        )
            
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if samplers:
                tuning_sampler = st.selectbox(
                    "Arama Örnekleyicisi (Sampler)",
                    options=samplers,
                    format_func=lambda s: TUNING_SAMPLER_TR.get(s, s)
                )
            else:
                tuning_sampler = None
            tuning_trials = st.slider("Deneme Sayısı (n_trials)", min_value=3, max_value=100, value=10, step=1)
            
        with t_col2:
            if pruners:
                tuning_pruner = st.selectbox(
                    "Budayıcı (Pruner)",
                    options=pruners,
                    format_func=lambda p: TUNING_PRUNER_TR.get(p, p)
                )
            else:
                tuning_pruner = None
                
            if scoring_opts:
                tuning_scoring = st.selectbox(
                    "Optimizasyon Hedef Skoru (Scoring Metric)",
                    options=scoring_opts
                )
            else:
                tuning_scoring = None
elif selected_mode == "tune" and not tuning_applicable:
    st.info(f"'{task}' görevi veya '{learning_type}' öğrenme türü için hiperparametre optimizasyonu konfigürasyonu desteklenmemektedir.")

# Tuning konfigürasyonu
tuning_payload = None
if selected_mode == "tune" and tuning_applicable:
    tuning_payload = {
        "enabled": True,
        "n_trials": tuning_trials,
        "sampler": tuning_sampler,
        "pruner": tuning_pruner,
        "metric": tuning_scoring,
        "parameters": {}
    }

# Model parameters (sadece preset == "custom" iken overrides ekle, aksi halde {})
final_parameters = custom_overrides if (task and algorithm and preset == "custom") else {}

# API JSON Sözleşmesini Oluştur (Top-Level Canonical Vocabulary)
req_payload = {
    "project_name": exp_name,
    "dataset_id": selected_dataset_id,
    "algorithm_type": learning_type,
    "task": task or "",
    "mode": selected_mode,
    "model": {
        "name": algorithm or "",
        "preset": preset,
        "parameters": final_parameters
    }
}
if preprocess_payload is not None:
    req_payload["preprocess"] = preprocess_payload
if data_split_payload is not None:
    req_payload["data_split"] = data_split_payload
if tuning_payload is not None:
    req_payload["tuning"] = tuning_payload

# ==============================================================================
# 7. DENEY YAPILANDIRMA ÖZETİ VE GÖNDERİM
# ==============================================================================
with st.container(border=True):
    st.markdown("### 📋 Deney Yapılandırma Özeti")
    st.caption("Oluşturulacak deneyin yapılandırma parametrelerini onaylayın.")

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.markdown("**📁 Veri & Görev**")
        ds_display = selected_ds_meta.get('display_name') or selected_ds_meta.get('name')
        st.write(f"• **Veri:** {ds_display}")
        st.write(f"• **Görev:** {TASK_NAMES_TR.get(task, task)}")
        st.write(f"• **Öğrenme:** {LEARNING_TYPES_TR.get(learning_type, learning_type)}")

    with s_col2:
        st.markdown("**🤖 Model & Mimari**")
        algo_name_display = get_algorithm_display_name(algorithm, capabilities) if algorithm else "-"
        st.write(f"• **Model:** {algo_name_display}")
        st.write(f"• **Hazır Ayar:** {PRESET_NAMES_TR.get(preset, preset) if preset else '-'}")

    with s_col3:
        st.markdown("**⚙️ Mod & Doğrulama**")
        st.write(f"• **Mod:** {MODE_NAMES_TR.get(selected_mode, selected_mode)}")
        if val_strategy:
            st.write(f"• **Bölümleme:** {VAL_TR.get(val_strategy, val_strategy)}")
            if "test_size" in data_split_params:
                ts_pct = int(round(float(data_split_params['test_size']) * 100))
                st.write(f"• **Test Kümesi:** %{ts_pct}")
        else:
            st.write("• **Bölümleme:** Belirtilmedi")

    with s_col4:
        st.markdown("**🛠️ Önişleme Adımları**")
        prep_status_items = []
        if preprocess_payload:
            if preprocess_payload.get("handle_missing"):
                prep_status_items.append("Eksik Değer Yönetimi")
            if preprocess_payload.get("encoding"):
                enc_m = preprocess_payload.get("encode_method") or "aktif"
                prep_status_items.append(f"Encoding ({enc_m})")
            if preprocess_payload.get("scaling"):
                scale_m = preprocess_payload.get("scale_method") or "aktif"
                scale_params_dict = preprocess_payload.get("scaling_parameters", {})
                if scale_m == "standard":
                    active_p = "with_std" if scale_params_dict.get("with_std") else ("with_mean" if scale_params_dict.get("with_mean") else "yalın")
                    prep_status_items.append(f"Scaling ({scale_m}: {active_p})")
                else:
                    prep_status_items.append(f"Scaling ({scale_m})")
        if prep_status_items:
            for item in prep_status_items:
                st.write(f"• {item}")
        else:
            st.write("• Standart / Ham Veri")

# Developer Payload Expander
with st.expander("🔍 Geliştirici & API İstek Yükü (Canonical JSON Payload)", expanded=False):
    st.caption("AKSIS Core Engine tarafına (`POST /api/v1/experiments`) iletilecek resmi JSON veri yükü:")
    st.json(req_payload)

st.divider()

submitted = st.button("🚀 Deneyi Oluştur ve Başlat (Create Experiment)", use_container_width=True, type="primary")

if submitted:
    if not task or not algorithm:
        st.error("Lütfen görev ve algoritma seçimini eksiksiz tamamlayın.")
    else:
        try:
            exp = client.create_experiment(req_payload)
            st.session_state.last_experiment_id = exp["id"]
            st.success(f"✅ '{exp['name']}' ({exp['id']}) başarıyla yapılandırıldı! (Mod: {selected_mode.upper()})")
            st.info("Eğitimi veya optimizasyonu başlatmak için 'Çalıştırma & Sonuçlar' sayfasına geçebilirsiniz.")
        except AksisAPIError as e:
            st.error(f"Hata: {str(e)}")
