import streamlit as st
from api.client import AksisAPIError
from utils.model_guidance import get_algorithm_metadata, get_algorithm_display_name

st.title("⚙️ Deney Yapılandırma (Experiment Visual Editor)")
st.caption("AKSIS Çerçevesi için yeni bir makine öğrenmesi veya optimizasyon deneyi tanımlayın.")

client = st.session_state.client

# ==============================================================================
# SUNUM ETİKETLERİ VE GÖRÜNTÜLEME EŞLEŞTİRMELERİ (PRESENTATION ALIASES)
# ==============================================================================
TASK_NAMES_TR = {
    "classification": "Sınıflandırma (Classification)",
    "regression": "Regresyon (Regression)",
    "anomaly_detection": "Anomali Tespiti (Anomaly Detection)"
}

LEARNING_TYPES_TR = {
    "supervised": "Denetimli Öğrenme (Supervised)",
    "unsupervised": "Denetimsiz Öğrenme (Unsupervised)"
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
    "custom": "Özel Ayar (custom)",
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

TUNING_SPACE_TR = {
    "baseline": "Temel Arama Uzayı (baseline)",
    "deep": "Genişletilmiş / Derin Arama (deep)"
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
# 2. VERİ SETİ VE ÇALIŞMA MODU SEÇİMİ
# ==============================================================================
dataset_options = {d["id"]: d["name"] for d in datasets}

# Oturum durumundaki veri setini hatırla
default_ds_idx = 0
if "selected_dataset_id" in st.session_state and st.session_state.selected_dataset_id in dataset_options:
    default_ds_idx = list(dataset_options.keys()).index(st.session_state.selected_dataset_id)

selected_dataset_id = st.selectbox(
    "📁 Kullanılacak Veri Seti", 
    options=list(dataset_options.keys()), 
    format_func=lambda x: dataset_options[x], 
    index=default_ds_idx,
    key="exp_dataset_select"
)
st.session_state.selected_dataset_id = selected_dataset_id

selected_ds_meta = next(d for d in datasets if d["id"] == selected_dataset_id)
compatible_tasks = selected_ds_meta.get("compatible_tasks", [])

if not compatible_tasks:
    st.error("Seçilen veri seti için uyumlu bir görev tanımlanmamış.")
    st.stop()

# ==============================================================================
# 3. DENEY FORMU (VISUAL CONFIG EDITOR)
# ==============================================================================
with st.form("experiment_config_form"):
    
    st.subheader("1. Genel Parametreler & Çalışma Modu")
    default_exp_name = f"Deney_{selected_ds_meta['name'].replace('.csv', '')}"
    exp_name = st.text_input("Deney Adı (Project / Experiment Name)", value=default_exp_name)
    
    col_mode1, col_mode2 = st.columns(2)
    
    with col_mode1:
        # API'den gelen modes içinden New Experiment akışına uygun olanları filtrele (train, tune)
        # predict modu arayüzde 'Modeller & Toplu Tahmin' sayfasında özel olarak yönetilir.
        api_modes = capabilities.get("modes", [])
        workflow_modes = [m for m in api_modes if m in ("train", "tune")]
        if not workflow_modes:
            st.warning("Bu iş akışı için geçerli bir çalıştırma modu (train/tune) bulunamadı.")
            selected_mode = None
        else:
            selected_mode = st.selectbox(
                "🚀 Çalışma Modu (Mode)",
                options=workflow_modes,
                format_func=lambda m: MODE_NAMES_TR.get(m, m),
                help="'train': Standart model eğitimi | 'tune': Optuna ile otomatik hiperparametre optimizasyonu"
            )
        
    with col_mode2:
        api_learning_types = capabilities.get("learning_types", [])
        if not api_learning_types:
            st.warning("Sistemde kayıtlı öğrenme türü bulunamadı.")
            learning_type = None
        else:
            learning_type = st.selectbox(
                "🧠 Öğrenme Türü (Learning Type)", 
                options=api_learning_types,
                format_func=lambda x: LEARNING_TYPES_TR.get(x, x)
            )
        
    # Görev Seçimi (Learning Type ve Dataset Uyumuna Göre Filtrelenir)
    if learning_type:
        tasks_by_learning = capabilities.get("tasks", {}).get(learning_type, [])
        valid_tasks = [t for t in tasks_by_learning if t in compatible_tasks]
        
        if not valid_tasks:
            st.warning(f"Seçilen veri seti '{learning_type}' altında doğrudan uyumlu görev içermiyor.")
            task = None
        else:
            task = st.selectbox(
                "🎯 Görev (Task)", 
                options=valid_tasks,
                format_func=lambda x: TASK_NAMES_TR.get(x, x)
            )
    else:
        task = None

    st.divider()

    # ==============================================================================
    # 4. MODEL VE ALGORİTMA SEÇİMİ (Dinamik Task Filtrelemesi)
    # ==============================================================================
    st.subheader("2. Model ve Algoritma Seçimi")
    
    if task:
        # Algoritmaları doğrudan API capabilities içinden çek
        available_algos = capabilities.get("algorithms", {}).get(task, [])
        api_presets = capabilities.get("model_presets", [])
        
        if not available_algos:
            st.warning("Bu görev için API üzerinde kayıtlı algoritma bulunamadı.")
            algorithm = None
            preset = None
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                algorithm = st.selectbox(
                    "Algoritma", 
                    options=available_algos,
                    format_func=lambda x: get_algorithm_display_name(x, capabilities),
                    help="AKSIS Registry tarafından sunulan algoritmalar"
                )
            with col_m2:
                if api_presets:
                    preset = st.selectbox(
                        "Model Hazır Ayarı (Model Preset)", 
                        options=api_presets,
                        format_func=lambda x: PRESET_NAMES_TR.get(x, x)
                    )
                else:
                    preset = None
                
            # Seçilen model için anlık karar destek rehberi
            if algorithm:
                algo_meta = get_algorithm_metadata(algorithm, capabilities)
                display_title = algo_meta.get("display_name") or algorithm
                description = algo_meta.get("description", "Model açıklaması sağlanmamış.")
                strengths = algo_meta.get("strengths", [])
                limitations = algo_meta.get("limitations", [])
                best_for = algo_meta.get("best_for", [])
                
                with st.container(border=True):
                    st.markdown(f"💡 **Model Bilgi Rehberi: `{algorithm}`** ({display_title})")
                    st.write(description)
                    if best_for:
                        st.markdown(f"**🎯 En Uygun Senaryo:** {', '.join(best_for) if isinstance(best_for, list) else best_for}")
                    if strengths:
                        st.caption(f"**✅ Güçlü Yönler:** {', '.join(strengths) if isinstance(strengths, list) else strengths}")
                    if limitations:
                        st.caption(f"**⚠️ Dikkat Edilmesi Gerekenler:** {', '.join(limitations) if isinstance(limitations, list) else limitations}")
    else:
        algorithm = None
        preset = None
        st.info("Lütfen önce geçerli bir görev seçin.")

    # ==============================================================================
    # 5. HİPERPARAMETRE OPTİMİZASYONU (Tuning - Yalnızca Mode='tune' İken Açılır)
    # ==============================================================================
    tuning_sampler = None
    tuning_pruner = None
    tuning_trials = 10
    tuning_scoring = None
    tuning_space = "baseline"
    
    if selected_mode == "tune":
        st.divider()
        st.subheader("⚡ 3. Hiperparametre Optimizasyonu Ayarları (Optuna Tuning)")
        st.caption("Mode='tune' seçildiği için Optuna arama uzayı ve deneme parametreleri aktif edildi.")
        
        tuning_opts = capabilities.get("tuning_options", {})
        samplers = tuning_opts.get("sampler", [])
        pruners = tuning_opts.get("pruner", [])
        spaces = tuning_opts.get("space_preset", [])
        
        scoring_opts = capabilities.get("scoring_options", {}).get(task, [])
            
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

    st.divider()

    # ==============================================================================
    # 6. ÖNİŞLEME VE DOĞRULAMA AYARLARI (İsteğe Bağlı)
    # ==============================================================================
    with st.expander("🛠️ Önişleme Seçenekleri (PreprocessConfig)"):
        prep_strats = capabilities.get("preprocessing_strategies", {})
        
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
        
    with st.expander("📐 Doğrulama Seçenekleri (Validation / DataSplitConfig)"):
        val_options = capabilities.get("validation_options", [])
        
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

    st.divider()
    
    submitted = st.form_submit_button("🚀 Deneyi Oluştur (Create Experiment)", use_container_width=True)
    
    if submitted:
        if not task or not algorithm:
            st.error("Lütfen görev ve algoritma seçimini eksiksiz tamamlayın.")
        else:
            # API JSON Sözleşmesini Oluştur
            req_payload = {
                "name": exp_name,
                "dataset_id": selected_dataset_id,
                "learning_type": learning_type,
                "task": task,
                "mode": selected_mode,
                "model": {
                    "algorithm": algorithm,
                    "preset": preset,
                    "overrides": {}
                },
                "preprocessing": {
                    "missing_value": None if missing_val == "none" else missing_val,
                    "encoding": None if encoding == "none" else encoding,
                    "scaling": None if scaling == "none" else scaling
                },
                "validation": {
                    "strategy": val_strategy,
                    "test_size": test_size
                },
                "tuning": {
                    "enabled": (selected_mode == "tune"),
                    "sampler": tuning_sampler,
                    "pruner": tuning_pruner,
                    "trials": tuning_trials,
                    "scoring": tuning_scoring
                } if selected_mode == "tune" else None
            }
            
            try:
                exp = client.create_experiment(req_payload)
                st.session_state.last_experiment_id = exp["id"]
                st.success(f"✅ '{exp['name']}' ({exp['id']}) başarıyla yapılandırıldı! (Mod: {selected_mode.upper()})")
                st.info("Eğitimi veya optimizasyonu başlatmak için 'Çalıştırma & Sonuçlar' sayfasına geçebilirsiniz.")
            except AksisAPIError as e:
                st.error(f"Hata: {str(e)}")
