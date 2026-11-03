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
        Veri setinizi, modelinizi ve deney akışınızı tek bir yerden yapılandırın.
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


VIS_GROUP_TITLES_TR = {
    "pretrain_list": "Eğitim Öncesi Grafikler",
    "tuning_list": "Optimizasyon Grafikleri",
    "cla_posttrain_list": "Sınıflandırma Grafikleri",
    "reg_posttrain_list": "Regresyon Grafikleri",
    "ano_posttrain_list": "Anomali Tespiti Grafikleri"
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
    st.markdown("### 📁 Veri ve Çalışma Kapsamı")
    st.caption("Deneyin yürütüleceği veri kaynağını, adlandırmasını ve temel makine öğrenmesi görevini belirleyin.")

    c_ds1, c_ds2 = st.columns([1, 1])
    with c_ds1:
        selected_dataset_id = st.selectbox(
            "Kullanılacak Veri Seti", 
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
        exp_name = st.text_input("Deney Adı", value=default_exp_name)

    # Parametre Seçimleri: Learning Type, Task, Execution Mode
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        if len(valid_learning_types) == 1:
            learning_type = valid_learning_types[0]
            st.caption("Öğrenme Türü")
            st.markdown(f"**{LEARNING_TYPES_TR.get(learning_type, learning_type)}**")
        else:
            learning_type = st.selectbox(
                "Öğrenme Türü",
                options=valid_learning_types,
                format_func=lambda x: LEARNING_TYPES_TR.get(x, x)
            )

    with col_info2:
        active_valid_tasks = [t for t, lt in valid_task_to_lt.items() if lt == learning_type]
        if len(active_valid_tasks) == 1:
            task = active_valid_tasks[0]
            st.caption("Görev")
            st.markdown(f"**{TASK_NAMES_TR.get(task, task)}**")
        else:
            task = st.selectbox(
                "Görev",
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
                "Çalışma Modu",
                options=workflow_modes,
                format_func=lambda m: MODE_NAMES_TR.get(m, m),
                help="'train': Standart model eğitimi | 'tune': Optuna ile otomatik hiperparametre optimizasyonu"
            )

# ==============================================================================
# 3. MODEL VE ALGORİTMA SEÇİMİ (Model & Algorithm)
# ==============================================================================
custom_overrides = {}

with st.container(border=True):
    st.markdown("### 🤖 Model")
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
                    "Algoritma", 
                    options=available_algos,
                    format_func=lambda x: get_algorithm_display_name(x, capabilities),
                    help="AKSIS Registry tarafından sunulan algoritmalar",
                    key="model_algo_select"
                )
            with col_m2:
                if api_presets:
                    preset = st.selectbox(
                        "Model Hazır Ayarı", 
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
                
                with st.expander(f"💡 Model Hakkında: {display_title}", expanded=False):
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
# 3. HİPERPARAMETRE OPTİMİZASYONU (Tuning - Yalnızca Mode='tune' İken Açılır/Kapanır)
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
    with st.expander("⚡ Hiperparametre Optimizasyonu", expanded=True):
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
                    "Arama Örnekleyicisi",
                    options=samplers,
                    format_func=lambda s: TUNING_SAMPLER_TR.get(s, s)
                )
            else:
                tuning_sampler = None
            tuning_trials = st.slider("Deneme Sayısı", min_value=3, max_value=100, value=10, step=1)
            
        with t_col2:
            if pruners:
                tuning_pruner = st.selectbox(
                    "Budayıcı",
                    options=pruners,
                    format_func=lambda p: TUNING_PRUNER_TR.get(p, p)
                )
            else:
                tuning_pruner = None
                
            if scoring_opts:
                tuning_scoring = st.selectbox(
                    "Hedef Skor Metriği",
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

# ==============================================================================
# 4. ÖNİŞLEME BORU HATTI (Preprocessing Pipeline - Açılır/Kapanır)
# ==============================================================================
preprocess_payload = None

if learning_type and task and is_config_applicable(capabilities, "preprocess", learning_type, task):
    with st.expander("🛠️ Önişleme", expanded=False):
        st.caption("Verinizi model eğitimine hazırlamak için eksik değer tamamlama, kategorik kodlama ve sayısal ölçeklendirme adımlarını yapılandırın.")

        prep_caps = capabilities.get("preprocessing", {}) if isinstance(capabilities, dict) else {}
        prep_defaults = prep_caps.get("defaults", {}) if isinstance(prep_caps, dict) else {}

        numeric_imputer_methods = prep_caps.get("numeric_imputer_methods", []) if isinstance(prep_caps, dict) else []
        categorical_imputer_methods = prep_caps.get("categorical_imputer_methods", []) if isinstance(prep_caps, dict) else []
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
            "🧩 Eksik Değerler",
            "🔤 Kategorik Kodlama",
            "⚖️ Sayısal Ölçeklendirme"
        ])

        # Tab 1: Eksik Değer Yönetimi
        with tab_missing:
            ui_numeric_imputer = None
            ui_categorical_imputer = None
            ui_numeric_fill = None
            ui_categorical_fill = None

            ui_handle_missing = st.checkbox(
                "Eksik Değerleri Yönet", help="Pipeline parametresi: handle_missing",
                value=bool(prep_defaults.get("handle_missing", True)),
                key="prep_handle_missing"
            )

            if ui_handle_missing:
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    def_num_imputer = prep_defaults.get("numeric_imputer")
                    if numeric_imputer_methods:
                        def_num_idx = (
                            numeric_imputer_methods.index(def_num_imputer)
                            if def_num_imputer in numeric_imputer_methods
                            else 0
                        )
                        ui_numeric_imputer = st.selectbox(
                            "Sayısal Değer Tamamlama",
                            options=numeric_imputer_methods,
                            index=def_num_idx,
                            format_func=lambda x: PREP_TR.get(x, x),
                            key="prep_numeric_imputer"
                        )
                    else:
                        diag_num_imp = validate_capability_path(
                            capabilities,
                            path=("preprocessing", "numeric_imputer_methods"),
                            required=True,
                            allow_empty=False
                        )
                        st.warning(f"Sayısal tamamlama yöntemleri bulunamadı: {diag_num_imp}")
                        ui_numeric_imputer = def_num_imputer

                    if ui_numeric_imputer == "constant":
                        num_fill_def = prep_defaults.get("numeric_fill_value")
                        default_num_str = str(num_fill_def) if num_fill_def is not None else ""
                        raw_num_fill = st.text_input(
                            "Sayısal Doldurma Değeri",
                            value=default_num_str,
                            key="prep_numeric_fill_value",
                            help="Sayısal tamamlama sabit değerini girin. Boş bırakılırsa atanmaz."
                        ).strip()

                        if raw_num_fill == "":
                            ui_numeric_fill = None
                        else:
                            try:
                                if "." in raw_num_fill or "e" in raw_num_fill.lower():
                                    ui_numeric_fill = float(raw_num_fill)
                                else:
                                    try:
                                        ui_numeric_fill = int(raw_num_fill)
                                    except ValueError:
                                        ui_numeric_fill = float(raw_num_fill)
                            except ValueError:
                                st.error("Lütfen geçerli bir sayısal değer girin (örn: 0, 42, 3.14).")
                                ui_numeric_fill = None

                with c_m2:
                    def_cat_imputer = prep_defaults.get("categorical_imputer")
                    if categorical_imputer_methods:
                        def_cat_idx = (
                            categorical_imputer_methods.index(def_cat_imputer)
                            if def_cat_imputer in categorical_imputer_methods
                            else 0
                        )
                        ui_categorical_imputer = st.selectbox(
                            "Kategorik Değer Tamamlama",
                            options=categorical_imputer_methods,
                            index=def_cat_idx,
                            format_func=lambda x: PREP_TR.get(x, x),
                            key="prep_categorical_imputer"
                        )
                    else:
                        diag_cat_imp = validate_capability_path(
                            capabilities,
                            path=("preprocessing", "categorical_imputer_methods"),
                            required=True,
                            allow_empty=False
                        )
                        st.warning(f"Kategorik tamamlama yöntemleri bulunamadı: {diag_cat_imp}")
                        ui_categorical_imputer = def_cat_imputer

                    if ui_categorical_imputer == "constant":
                        cat_fill_def = prep_defaults.get("categorical_fill_value")
                        default_cat_str = str(cat_fill_def) if cat_fill_def is not None else ""
                        raw_cat_fill = st.text_input(
                            "Kategorik Doldurma Değeri",
                            value=default_cat_str,
                            key="prep_categorical_fill_value",
                            help="Kategorik tamamlama sabit değerini girin. Boş bırakılırsa atanmaz."
                        )
                        if raw_cat_fill.strip() == "":
                            ui_categorical_fill = None
                        else:
                            ui_categorical_fill = raw_cat_fill.strip()

        # Tab 2: Kategorik Kodlama
        with tab_encoding:
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                ui_encoding = st.checkbox(
                    "Kategorik Kodlama Etkin", help="Pipeline parametresi: encoding",
                    value=bool(prep_defaults.get("encoding", False)),
                    key="prep_encoding"
                )

                def_encode_method = prep_defaults.get("encode_method")
                if encoding_methods:
                    def_enc_idx = encoding_methods.index(def_encode_method) if def_encode_method in encoding_methods else 0
                    ui_encode_method = st.selectbox(
                        "Kodlama Yöntemi",
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
                    "Hedef Değişken Kodlama", help="Pipeline parametresi: targetEncoding",
                    value=bool(prep_defaults.get("targetEncoding", False)),
                    key="prep_target_encoding"
                )

                if ui_encode_method == "hashing":
                    st.caption("💡 **Bilgi:** Hashing kodlama, yüksek kardinaliteli kategorik değişkenleri sabit boyutlu bir öznitelik uzayına indirger.")

            with c_e2:
                st.markdown("**Kodlama Parametreleri**")
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
                                p_label,
                                value=p_v,
                                key=f"{field_key}_bool",
                                help=help_text
                            )
                        elif isinstance(p_v, int):
                            ui_encoding_params[p_k] = st.number_input(
                                p_label,
                                value=p_v,
                                step=1,
                                key=f"{field_key}_int",
                                help=help_text
                            )
                        elif isinstance(p_v, float):
                            ui_encoding_params[p_k] = st.number_input(
                                p_label,
                                value=p_v,
                                step=0.01 if p_v < 1.0 else 1.0,
                                format="%.4f" if p_v < 0.01 else "%.2f",
                                key=f"{field_key}_float",
                                help=help_text
                            )
                        elif isinstance(p_v, str):
                            ui_encoding_params[p_k] = st.text_input(
                                p_label,
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
                    "Sayısal Ölçeklendirme Etkin", help="Pipeline parametresi: scaling",
                    value=bool(prep_defaults.get("scaling", False)),
                    key="prep_scaling"
                )

                def_scale_method = prep_defaults.get("scale_method")
                if scaling_methods:
                    def_scale_idx = scaling_methods.index(def_scale_method) if def_scale_method in scaling_methods else 0
                    ui_scale_method = st.selectbox(
                        "Ölçeklendirme Yöntemi",
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
                st.markdown("**Ölçeklendirme Parametreleri**")
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
                                "with_std": "Birim Varyans Ölçeklendirme (Önerilen)",
                                "with_mean": "Ortalama Merkezileştirme",
                                "none": "Hiçbiri"
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
                                p_label,
                                value=p_v,
                                key=f"{field_key}_bool",
                                help=help_text
                            )
                        elif isinstance(p_v, int):
                            ui_scaling_params[p_k] = st.number_input(
                                p_label,
                                value=p_v,
                                step=1,
                                key=f"{field_key}_int",
                                help=help_text
                            )
                        elif isinstance(p_v, float):
                            ui_scaling_params[p_k] = st.number_input(
                                p_label,
                                value=p_v,
                                step=0.01 if p_v < 1.0 else 1.0,
                                format="%.4f" if p_v < 0.01 else "%.2f",
                                key=f"{field_key}_float",
                                help=help_text
                            )
                        elif isinstance(p_v, str):
                            ui_scaling_params[p_k] = st.text_input(
                                p_label,
                                value=p_v,
                                key=f"{field_key}_str",
                                help=help_text
                            )
                        else:
                            ui_scaling_params[p_k] = p_v

        # Payload sanitization and construction
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

        preprocess_payload = {
            "handle_missing": bool(ui_handle_missing),
            "numeric_imputer": final_num_imputer,
            "categorical_imputer": final_cat_imputer,
            "numeric_fill_value": final_num_fill,
            "categorical_fill_value": final_cat_fill,
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
    with st.expander("📐 Veri Bölümleme", expanded=False):
        st.caption("Modelin değerlendirilmesi için eğitim-test kümesi ayrımı veya K-Fold stratejisini belirleyin.")

        split_meta = capabilities.get("data_split", {}) if isinstance(capabilities, dict) else {}
        val_options = split_meta.get("methods", []) if isinstance(split_meta, dict) else []
        split_params = split_meta.get("parameters", {}) if isinstance(split_meta, dict) else {}

        ds_diag = validate_capability_path(capabilities, path=("data_split", "methods"), required=True, allow_empty=False)
        if val_options:
            val_strategy = st.selectbox(
                "Bölümleme Yöntemi", 
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
                            label=p_label,
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
                        label=p_label,
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
                        label=p_label,
                        value=p_default,
                        key=f"{field_key}_bool",
                        help=help_text
                    )
                    data_split_params[p_name] = bool(val)

                # 4. Generic fallback: int
                elif isinstance(p_default, int):
                    val = st.number_input(
                        label=p_label,
                        value=p_default,
                        step=1,
                        key=f"{field_key}_int",
                        help=help_text
                    )
                    data_split_params[p_name] = int(val)

                # 5. Generic fallback: float
                elif isinstance(p_default, float):
                    val = st.number_input(
                        label=p_label,
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
                        label=p_label,
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
# 6. VERİ ANALİZİ (Data Analysis - Açılır/Kapanır)
# ==============================================================================
data_analyze_payload = None

if learning_type and task and is_config_applicable(capabilities, "data_analyze", learning_type, task):
    with st.expander("📊 Veri Analizi", expanded=False):
        st.caption("Veri seti üzerinde modelleme öncesinde çalıştırılacak analiz fonksiyonlarını yapılandırın.")

        analyze_caps = capabilities.get("data_analyze", {}) if isinstance(capabilities, dict) else {}
        analyze_defaults = analyze_caps.get("defaults", {}) if isinstance(analyze_caps, dict) else {}
        analyze_functions = analyze_defaults.get("data_analyze_functions", []) if isinstance(analyze_defaults, dict) else []

        diag_da = validate_capability_path(capabilities, path=("data_analyze",), required=True, allow_empty=False)
        diag_da_defaults = validate_capability_path(capabilities, path=("data_analyze", "defaults"), required=True, allow_empty=False)
        diag_da_fns = validate_capability_path(capabilities, path=("data_analyze", "defaults", "data_analyze_functions"), required=True, allow_empty=False)

        if not diag_da:
            st.warning(f"Veri analizi yetenek tanımı eksik: {diag_da}")
        elif not diag_da_defaults:
            st.warning(f"Veri analizi varsayılanları eksik: {diag_da_defaults}")
        elif not diag_da_fns:
            st.warning(f"Veri analizi fonksiyonları bulunamadı: {diag_da_fns}")
        else:
            da_payload_fns = []
            cols = st.columns(min(len(analyze_functions), 2) if len(analyze_functions) > 1 else 1)
            for idx, fn_item in enumerate(analyze_functions):
                if not isinstance(fn_item, dict):
                    continue
                fn_name = str(fn_item.get("f", ""))
                fn_enabled_default = bool(fn_item.get("enabled", True))
                fn_label = fn_name.replace("_", " ").title()

                with cols[idx % len(cols)]:
                    val = st.checkbox(
                        fn_label,
                        value=fn_enabled_default,
                        key=f"data_analyze_fn_{fn_name}",
                        help=f"`{fn_name}` analiz adımını çalıştır."
                    )
                    da_payload_fns.append({
                        "f": fn_name,
                        "enabled": bool(val)
                    })

            if da_payload_fns:
                data_analyze_payload = {
                    "data_analyze_functions": da_payload_fns
                }

# ==============================================================================
# 7. MODEL DEĞERLENDİRME (Evaluation Metrics - Açılır/Kapanır)
# ==============================================================================
evaluation_payload = None

eval_applicable = (
    learning_type is not None
    and task is not None
    and (
        is_config_applicable(capabilities, f"{learning_type}_evaluation", learning_type, task)
        or is_config_applicable(capabilities, "evaluation", learning_type, task)
        or not (capabilities.get("config_applicability") if isinstance(capabilities, dict) else None)
    )
)

if eval_applicable:
    with st.expander("📈 Değerlendirme", expanded=False):
        st.caption("Model performansını ölçmek için kullanılacak değerlendirme metriklerini belirleyin.")

        evaluation_caps = capabilities.get("evaluation", {}) if isinstance(capabilities, dict) else {}
        evaluation_task = evaluation_caps.get(task, {}) if isinstance(evaluation_caps, dict) else {}

        evaluation_field = evaluation_task.get("field")
        evaluation_items = evaluation_task.get("items", [])

        diag_eval_root = validate_capability_path(capabilities, path=("evaluation",), required=True, allow_empty=False)
        diag_eval_task = validate_capability_path(capabilities, path=("evaluation", task), required=True, allow_empty=False)
        diag_eval_field = validate_capability_path(capabilities, path=("evaluation", task, "field"), required=True, allow_empty=False)
        diag_eval_items = validate_capability_path(capabilities, path=("evaluation", task, "items"), required=True, allow_empty=False)

        if not diag_eval_root:
            st.warning(f"Değerlendirme yetenek tanımı eksik: {diag_eval_root}")
        elif not diag_eval_task:
            st.warning(f"'{task}' görevi için değerlendirme yetenekleri bulunamadı: {diag_eval_task}")
        elif not diag_eval_field:
            st.warning(f"Değerlendirme alan adı eksik: {diag_eval_field}")
        elif not diag_eval_items:
            st.warning(f"Değerlendirme metrikleri bulunamadı: {diag_eval_items}")
        else:
            eval_payload_items = []
            cols = st.columns(min(len(evaluation_items), 3) if len(evaluation_items) > 1 else 1)
            for idx, item in enumerate(evaluation_items):
                if not isinstance(item, dict):
                    continue
                fn_name = str(item.get("f", ""))
                fn_enabled = bool(item.get("enabled", True))
                fn_label = fn_name.replace("_", " ").title()

                with cols[idx % len(cols)]:
                    val = st.checkbox(
                        fn_label,
                        value=fn_enabled,
                        key=f"eval_fn_{evaluation_field}_{fn_name}",
                        help=f"`{fn_name}` metriğini hesapla."
                    )
                    eval_payload_items.append({
                        "f": fn_name,
                        "enabled": bool(val)
                    })

            if evaluation_field and eval_payload_items:
                evaluation_payload = {
                    evaluation_field: eval_payload_items
                }

# ==============================================================================
# 8. GÖRSELLEŞTİRME (Visualization - Açılır/Kapanır)
# ==============================================================================
visualization_payload = None

vis_applicable = (
    learning_type is not None
    and task is not None
    and (
        is_config_applicable(capabilities, f"{learning_type}_visualization", learning_type, task)
        or is_config_applicable(capabilities, "visualization", learning_type, task)
        or not (capabilities.get("config_applicability") if isinstance(capabilities, dict) else None)
    )
)

if vis_applicable:
    with st.expander("📉 Görselleştirme", expanded=False):
        st.caption("Eğitim öncesi ve sonrası model ve veri grafiklerini yapılandırın.")

        visualization_caps = capabilities.get("visualization", {}) if isinstance(capabilities, dict) else {}
        visualization_group = visualization_caps.get(learning_type, {}) if isinstance(visualization_caps, dict) else {}

        common_visualization = visualization_group.get("common", {}) if isinstance(visualization_group, dict) else {}
        tasks_visualization = visualization_group.get("tasks", {}) if isinstance(visualization_group, dict) else {}
        task_visualization = tasks_visualization.get(task, {}) if isinstance(tasks_visualization, dict) else {}

        diag_vis_root = validate_capability_path(capabilities, path=("visualization",), required=True, allow_empty=False)
        diag_vis_group = validate_capability_path(capabilities, path=("visualization", learning_type), required=True, allow_empty=False)
        diag_vis_common = validate_capability_path(capabilities, path=("visualization", learning_type, "common"), required=True, allow_empty=False)
        diag_vis_tasks = validate_capability_path(capabilities, path=("visualization", learning_type, "tasks"), required=True, allow_empty=False)

        if not diag_vis_root:
            st.warning(f"Görselleştirme yetenek tanımı eksik: {diag_vis_root}")
        elif not diag_vis_group:
            st.warning(f"'{learning_type}' için görselleştirme yetenekleri bulunamadı: {diag_vis_group}")
        elif not diag_vis_common:
            st.warning(f"Ortak görselleştirme ayarları bulunamadı: {diag_vis_common}")
        elif not diag_vis_tasks:
            st.warning(f"Göreve özel görselleştirme ayarları bulunamadı: {diag_vis_tasks}")
        else:
            has_show_plots_key = "show_plots" in common_visualization
            def_show_plots = bool(common_visualization.get("show_plots", True)) if has_show_plots_key else True

            if has_show_plots_key:
                ui_show_plots = st.checkbox(
                    "Grafikleri Göster",
                    value=def_show_plots,
                    key="vis_master_show_plots",
                    help="Model eğitim ve değerlendirme aşamalarında grafik üretimini etkinleştirir."
                )
            else:
                ui_show_plots = True

            vis_payload_dict = {}
            if has_show_plots_key:
                vis_payload_dict["show_plots"] = bool(ui_show_plots)

            groups_to_process = {}
            if "pretrain_list" in common_visualization and isinstance(common_visualization["pretrain_list"], list):
                groups_to_process["pretrain_list"] = common_visualization["pretrain_list"]

            if selected_mode == "tune" and "tuning_list" in common_visualization and isinstance(common_visualization["tuning_list"], list):
                groups_to_process["tuning_list"] = common_visualization["tuning_list"]

            if isinstance(task_visualization, dict):
                for field_name, items in task_visualization.items():
                    if isinstance(items, list):
                        groups_to_process[field_name] = items

            if ui_show_plots:
                for group_key, items in groups_to_process.items():
                    group_title = VIS_GROUP_TITLES_TR.get(group_key, group_key.replace("_", " ").title())
                    st.markdown(f"**{group_title}**")
                    group_payload_items = []
                    cols = st.columns(min(len(items), 3) if len(items) > 1 else 1)
                    for idx, item in enumerate(items):
                        if not isinstance(item, dict):
                            continue
                        fn_name = str(item.get("f", ""))
                        fn_enabled = bool(item.get("enabled", True))
                        fn_label = fn_name.replace("_", " ").title()

                        with cols[idx % len(cols)]:
                            val = st.checkbox(
                                fn_label,
                                value=fn_enabled,
                                key=f"vis_fn_{group_key}_{fn_name}",
                                help=f"`{fn_name}` grafiğini oluştur."
                            )
                            group_payload_items.append({
                                "f": fn_name,
                                "enabled": bool(val)
                            })
                    vis_payload_dict[group_key] = group_payload_items
            else:
                for group_key, items in groups_to_process.items():
                    group_payload_items = []
                    for item in items:
                        if isinstance(item, dict):
                            group_payload_items.append({
                                "f": str(item.get("f", "")),
                                "enabled": bool(item.get("enabled", True))
                            })
                    vis_payload_dict[group_key] = group_payload_items

            if vis_payload_dict:
                visualization_payload = vis_payload_dict

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
if data_analyze_payload is not None:
    req_payload["data_analyze"] = data_analyze_payload
if evaluation_payload is not None:
    req_payload["evaluation"] = evaluation_payload
if visualization_payload is not None:
    req_payload["visualization"] = visualization_payload

# ==============================================================================
# 9. DENEY YAPILANDIRMA ÖZETİ VE GÖNDERİM
# ==============================================================================
with st.container(border=True):
    st.markdown("### 📋 Deney Özeti")
    st.caption("Oluşturulacak deneyin yapılandırma parametrelerini onaylayın.")

    # 1. Bilgileri dinamik olarak hazırla
    ds_display = selected_ds_meta.get('display_name') or selected_ds_meta.get('name')
    task_label = TASK_NAMES_TR.get(task, task)
    lt_label = LEARNING_TYPES_TR.get(learning_type, learning_type)
    algo_name_display = get_algorithm_display_name(algorithm, capabilities) if algorithm else "-"
    preset_label = PRESET_NAMES_TR.get(preset, preset) if preset else "Standart"
    mode_label = MODE_NAMES_TR.get(selected_mode, selected_mode.title() if selected_mode else "-")

    # Bölümleme özeti
    if val_strategy:
        val_name = VAL_TR.get(val_strategy, val_strategy)
        if "test_size" in data_split_params:
            ts_pct = int(round(float(data_split_params['test_size']) * 100))
            split_desc = f"{val_name} · %{ts_pct} test"
        else:
            split_desc = val_name
    else:
        split_desc = "Bölümleme belirtilmedi"

    # Hiperparametre özeti (mode == tune ise)
    s_sampler = TUNING_SAMPLER_TR.get(tuning_sampler, tuning_sampler or "-")
    s_pruner = TUNING_PRUNER_TR.get(tuning_pruner, tuning_pruner or "-")
    s_trials = f"{tuning_trials} deneme"
    s_metric = f"Hedef: {tuning_scoring}" if tuning_scoring else "Varsayılan hedef"

    # Önişleme özeti
    prep_summary_items = []
    if preprocess_payload:
        if preprocess_payload.get("handle_missing"):
            num_imp = preprocess_payload.get("numeric_imputer")
            if num_imp and num_imp != "none":
                prep_summary_items.append(f"Eksik Değer ({PREP_TR.get(num_imp, num_imp)})")
            else:
                prep_summary_items.append("Eksik Değer Yönetimi")
        if preprocess_payload.get("encoding"):
            enc_m = preprocess_payload.get("encode_method")
            enc_label = PREP_TR.get(enc_m, enc_m.title() if enc_m else "Encoding")
            prep_summary_items.append(enc_label if ("Encoding" in enc_label or "Kodlama" in enc_label) else f"{enc_label} Kodlama")
        if preprocess_payload.get("scaling"):
            scale_m = preprocess_payload.get("scale_method")
            scale_label = PREP_TR.get(scale_m, scale_m.title() if scale_m else "Scaling")
            prep_summary_items.append(scale_label if ("Scaler" in scale_label or "Ölçeklendirme" in scale_label) else f"{scale_label} Ölçeklendirme")
    if not prep_summary_items:
        prep_summary_items = ["Ek önişleme uygulanmıyor"]

    # Veri Analizi özeti (seçilen fonksiyonların listesi)
    active_da_items = []
    if data_analyze_payload and "data_analyze_functions" in data_analyze_payload:
        da_fns = data_analyze_payload.get("data_analyze_functions", [])
        for item in da_fns:
            if isinstance(item, dict) and item.get("enabled"):
                active_da_items.append(str(item.get("f", "")).replace("_", " ").title())
        if not active_da_items:
            active_da_items = ["Seçili analiz adımı yok"]
    else:
        active_da_items = ["Uygulanmıyor"]

    # Değerlendirme özeti (seçilen metriklerin listesi)
    active_eval_items = []
    if evaluation_payload and isinstance(evaluation_payload, dict):
        eval_items = []
        for k_ev, v_ev in evaluation_payload.items():
            if isinstance(v_ev, list):
                eval_items = v_ev
                break
        for item in eval_items:
            if isinstance(item, dict) and item.get("enabled"):
                active_eval_items.append(str(item.get("f", "")).replace("_", " ").title())
        if not active_eval_items:
            active_eval_items = ["Seçili metrik yok"]
    else:
        active_eval_items = ["Uygulanmıyor"]

    # Görselleştirme özeti (seçilen grafiklerin listesi)
    active_vis_items = []
    if visualization_payload and isinstance(visualization_payload, dict):
        if visualization_payload.get("show_plots") is False:
            active_vis_items = ["Grafik üretimi kapalı"]
        else:
            for k_vis, v_vis in visualization_payload.items():
                if k_vis == "show_plots":
                    continue
                if isinstance(v_vis, list):
                    for item in v_vis:
                        if isinstance(item, dict) and item.get("enabled"):
                            active_vis_items.append(str(item.get("f", "")).replace("_", " ").title())
            if not active_vis_items:
                active_vis_items = ["Seçili grafik yok"]
    else:
        active_vis_items = ["Uygulanmıyor"]

    # 1. Satır: Temel Kapsam, Model, Çalışma Modu ve Hiperparametre
    if selected_mode == "tune":
        c_vg, c_md, c_wk, c_tn = st.columns(4)
    else:
        c_vg, c_md, c_wk = st.columns(3)
        c_tn = None

    with c_vg:
        st.markdown("**📁 Veri & Görev**")
        st.markdown(f"**{ds_display}**")
        st.caption(f"{task_label} · {lt_label}")

    with c_md:
        st.markdown("**🤖 Model**")
        st.markdown(f"**{algo_name_display}**")
        st.caption(preset_label)

    with c_wk:
        st.markdown("**⚙️ Çalışma**")
        st.markdown(f"**{mode_label}**")
        st.caption(split_desc)

    if c_tn:
        with c_tn:
            st.markdown("**⚡ Hiperparametre**")
            st.markdown(f"**{s_sampler}**")
            st.caption(f"{s_pruner} · {s_trials}\n\n{s_metric}")

    st.divider()

    # 2. Satır: Boru Hattı Bileşenleri (Seçilenleri doğrudan listeleyen yapı)
    c_pr, c_da, c_ev, c_vi = st.columns(4)

    with c_pr:
        st.markdown("**🛠️ Önişleme**")
        for p_item in prep_summary_items:
            st.markdown(f"<span style='font-size: 0.9rem; color: #334155;'>• {p_item}</span>", unsafe_allow_html=True)

    with c_da:
        st.markdown("**📊 Veri Analizi**")
        for da_item in active_da_items:
            st.markdown(f"<span style='font-size: 0.9rem; color: #334155;'>• {da_item}</span>", unsafe_allow_html=True)

    with c_ev:
        st.markdown("**📈 Değerlendirme**")
        for ev_item in active_eval_items:
            st.markdown(f"<span style='font-size: 0.9rem; color: #334155;'>• {ev_item}</span>", unsafe_allow_html=True)

    with c_vi:
        st.markdown("**📉 Görselleştirme**")
        for vi_item in active_vis_items:
            st.markdown(f"<span style='font-size: 0.9rem; color: #334155;'>• {vi_item}</span>", unsafe_allow_html=True)

# Developer Payload Expander
with st.expander("🔍 Geliştirici & API İstek Yükü (Canonical JSON Payload)", expanded=False):
    st.caption("AKSIS Core Engine tarafına (`POST /api/v1/experiments`) iletilecek resmi JSON veri yükü:")
    st.json(req_payload)

st.divider()

submitted = st.button("🚀 Deneyi Oluştur", use_container_width=True, type="primary")

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
