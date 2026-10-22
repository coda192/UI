import streamlit as st
from api.client import AksisAPIError
from utils.model_guidance import (
    get_algorithm_display_name,
    get_algorithm_metadata,
    is_config_applicable,
    resolve_task_learning_types
)
from components.parameter_editor import render_parameter_editor

st.title("⚙️ Deney Yapılandırma")
st.caption("Makine öğrenmesi deneyiniz için veri seti, görev, model ve çalışma ayarlarını yapılandırın.")

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

selected_dataset_id = st.selectbox(
    "📁 Kullanılacak Veri Seti (Dataset)", 
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

# ==============================================================================
# 3. GENEL PARAMETRELER & ÇALIŞMA MODU
# ==============================================================================
st.subheader("1. Genel Parametreler & Çalışma Modu (Execution Mode)")
default_exp_name = f"Deney_{selected_ds_meta['name'].replace('.csv', '')}"
exp_name = st.text_input("Deney Adı (Project Name)", value=default_exp_name)

# Bilgi Kartı veya Seçim: Learning Type & Task
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
    # Seçilen learning_type altında kalan geçerli görevler
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

st.divider()

# ==============================================================================
# 4. MODEL VE ALGORİTMA SEÇİMİ (Model & Algorithm)
# ==============================================================================
st.subheader("2. Model ve Algoritma (Model & Algorithm)")

custom_overrides = {}

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

        # Seçilen model için anlık karar destek rehberi
        if algorithm:
            algo_meta = get_algorithm_metadata(algorithm, capabilities)
            display_title = algo_meta.get("display_name") or algorithm
            description = algo_meta.get("description", "Model açıklaması sağlanmamış.")
            strengths = algo_meta.get("strengths", [])
            limitations = algo_meta.get("limitations", [])
            best_for = algo_meta.get("best_for", [])
            
            with st.container(border=True):
                st.markdown(f"### ⚡ {display_title}")
                if display_title != algorithm:
                    st.caption(f"`{algorithm}`")
                st.write(description)
                if best_for:
                    st.markdown(f"**🎯 En Uygun Senaryo:** {', '.join(best_for) if isinstance(best_for, list) else best_for}")
                if strengths:
                    st.caption(f"**✅ Güçlü Yönler:** {', '.join(strengths) if isinstance(strengths, list) else strengths}")
                if limitations:
                    st.caption(f"**⚠️ Dikkat Edilmesi Gerekenler:** {', '.join(limitations) if isinstance(limitations, list) else limitations}")

        # ==============================================================================
        # ÖZEL MODEL PARAMETRELERİ (Yalnızca preset == "custom" İken Açılır)
        # ==============================================================================
        if preset == "custom" and algorithm:
            st.divider()
            # TODO: Re-enable custom parameter editing after the schema-driven parameter contract is finalized.
            st.info("Custom model parameter configuration is temporarily disabled for the demo.")
            # param_schemas_dict = capabilities.get("parameter_schema") or {}
            # algo_param_schema = param_schemas_dict.get(task, {}).get(algorithm)
            # custom_overrides = render_parameter_editor(
            #     algo_param_schema,
            #     key_prefix=f"custom_param_{task}_{algorithm}"
            # )
else:
    algorithm = None
    preset = None
    st.info("Lütfen önce geçerli bir görev seçin.")

# ==============================================================================
# 5. HİPERPARAMETRE OPTİMİZASYONU (Tuning - Yalnızca Mode='tune' ve İlgili Tuning Konfigürasyonu Uygulanabilir İken Açılır)
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
    st.divider()
    st.subheader("⚡ 3. Hiperparametre Optimizasyonu (Hyperparameter Tuning)")
    st.caption("Mode='tune' seçildiği için Optuna arama uzayı ve deneme parametreleri aktif edildi.")
    
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

st.divider()

# ==============================================================================
# 6. ÖNİŞLEME VE DOĞRULAMA AYARLARI (İsteğe Bağlı)
# ==============================================================================
missing_val = "none"
encoding = "none"
scaling = "none"

if learning_type and task and is_config_applicable(capabilities, "preprocess", learning_type, task):
    with st.expander("🛠️ Önişleme (Preprocessing)"):
        prep_strats = capabilities.get("preprocessing", {})
        
        missing_options = ["none"] + prep_strats.get("missing_value", [])
        encoding_options = ["none"] + prep_strats.get("encoding", [])
        scaling_options = ["none"] + prep_strats.get("scaling", [])
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            missing_val = st.selectbox(
                "Eksik Değer Yönetimi (Imputer)", 
                missing_options,
                format_func=lambda x: PREP_TR.get(x, x)
            )
        with c_p2:
            encoding = st.selectbox(
                "Kategorik Kodlama (Encoding)", 
                encoding_options,
                format_func=lambda x: PREP_TR.get(x, x)
            )
        with c_p3:
            scaling = st.selectbox(
                "Sayısal Ölçeklendirme (Scaling)", 
                scaling_options,
                format_func=lambda x: PREP_TR.get(x, x)
            )

val_strategy = None
test_size = 0.2

if learning_type and task and is_config_applicable(capabilities, "data_split", learning_type, task):
    with st.expander("📐 Veri Bölümleme ve Doğrulama (Data Splitting & Validation)"):
        split_meta = capabilities.get("data_split", {})
        val_options = split_meta.get("methods", []) if isinstance(split_meta, dict) else []
        
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            if val_options:
                val_strategy = st.selectbox(
                    "Doğrulama / Bölümleme Yöntemi", 
                    val_options,
                    format_func=lambda x: VAL_TR.get(x, x)
                )
            else:
                st.warning("Doğrulama yöntemi bulunamadı.")
                val_strategy = None
        with c_v2:
            test_size = st.slider("Test Kümesi Oranı (test_size)", 0.1, 0.5, 0.2, 0.05)

# Preprocess konfigürasyonu
preprocess_payload = None
if missing_val != "none" or encoding != "none" or scaling != "none":
    preprocess_payload = {
        "handle_missing": (missing_val != "none"),
        "numeric_imputer": None if missing_val == "none" else missing_val,
        "categorical_imputer": None if missing_val == "none" else missing_val,
        "encoding": (encoding != "none"),
        "encode_method": None if encoding == "none" else encoding,
        "encoding_parameters": {},
        "scaling": (scaling != "none"),
        "scale_method": None if scaling == "none" else scaling,
        "scaling_parameters": {}
    }

# Data split konfigürasyonu
data_split_payload = None
if val_strategy:
    data_split_payload = {
        "method": val_strategy,
        "parameters": {
            "test_size": test_size
        }
    }

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

with st.expander("Gönderilecek Deney Payload'u"):
    st.json(req_payload)

st.divider()

submitted = st.button("🚀 Deneyi Oluştur (Create Experiment)", use_container_width=True, type="primary")

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
