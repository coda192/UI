import streamlit as st
from api.client import AksisAPIError
from utils.model_guidance import get_model_guidance

st.title("⚙️ Deney Yapılandırma")

client = st.session_state.client

TASK_NAMES_TR = {
    "classification": "Sınıflandırma (Classification)",
    "regression": "Regresyon (Regression)",
    "anomaly_detection": "Anomali Tespiti (Anomaly Detection)"
}

LEARNING_TYPES_TR = {
    "supervised": "Denetimli Öğrenme (Supervised)",
    "unsupervised": "Denetimsiz Öğrenme (Unsupervised)"
}

PRESET_NAMES_TR = {
    "fast": "Hızlı (Fast)",
    "accurate": "Yüksek Doğruluk (Accurate)",
    "interpretable": "Yorumlanabilir (Interpretable)"
}

PREP_TR = {
    "none": "Uygulanmasın (None)",
    "mean": "Ortalama Değer (Mean)",
    "median": "Medyan (Median)",
    "most_frequent": "En Sık Tekrar Eden (Mode)",
    "drop": "Eksik Satırları Çıkar (Drop)",
    "onehot": "One-Hot Encoding",
    "label": "Label Encoding",
    "target": "Target Encoding",
    "standard": "StandardScaler (Z-Score)",
    "minmax": "MinMaxScaler [0, 1]",
    "robust": "RobustScaler (Aykırı Değerlere Dayanıklı)"
}

VAL_TR = {
    "holdout": "Ayrık Test Kümesi (Holdout / Train-Test)",
    "kfold": "K-Fold Çapraz Doğrulama (Cross-Validation)",
    "stratified_kfold": "Tabakalı K-Fold (Stratified K-Fold)"
}

try:
    capabilities = client.get_capabilities()
    
    # Veri seti seçimi
    datasets = client.get_datasets()
    if not datasets:
        st.warning("Kullanılabilir veri seti bulunamadı. Lütfen önce veri seti tanımlayın.")
        st.stop()
        
    dataset_options = {d["id"]: d["name"] for d in datasets}
    
    default_idx = 0
    if "selected_dataset_id" in st.session_state:
        ids = list(dataset_options.keys())
        if st.session_state.selected_dataset_id in ids:
            default_idx = ids.index(st.session_state.selected_dataset_id)
            
    selected_dataset_id = st.selectbox(
        "Kullanılacak Veri Seti", 
        options=list(dataset_options.keys()), 
        format_func=lambda x: dataset_options[x], 
        index=default_idx
    )
    st.session_state.selected_dataset_id = selected_dataset_id
    
    # Seçilen veri setine uyumlu görevleri çözümle
    selected_ds_meta = next(d for d in datasets if d["id"] == selected_dataset_id)
    compatible_tasks = selected_ds_meta.get("compatible_tasks", [])
    
    if not compatible_tasks:
        st.error("Seçilen veri seti için uyumlu bir görev tanımlanmamış.")
        st.stop()

    with st.form("experiment_config_form"):
        st.subheader("1. Genel Ayarlar")
        exp_name = st.text_input("Deney Adı", value=f"Deney_{selected_ds_meta['name'].replace('.csv', '')}")
        
        col1, col2 = st.columns(2)
        with col1:
            learning_type = st.selectbox(
                "Öğrenme Türü", 
                options=capabilities.get("learning_types", []),
                format_func=lambda x: LEARNING_TYPES_TR.get(x, x)
            )
        with col2:
            # Görevleri veri seti uyumluluğuna ve öğrenme türüne göre filtrele
            valid_tasks = [t for t in capabilities.get("tasks", {}).get(learning_type, []) if t in compatible_tasks]
            if not valid_tasks:
                st.warning(f"Bu veri seti için {LEARNING_TYPES_TR.get(learning_type, learning_type)} kapsamında uyumlu görev bulunamadı.")
                task = None
            else:
                task = st.selectbox(
                    "Görev (Task)", 
                    options=valid_tasks,
                    format_func=lambda x: TASK_NAMES_TR.get(x, x)
                )
                
        st.subheader("2. Model ve Algoritma Seçimi")
        if task:
            algos = capabilities.get("algorithms", {}).get(task, [])
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                algorithm = st.selectbox("Algoritma", algos)
            with col_m2:
                preset = st.selectbox(
                    "Model Hazır Ayarı (Preset)", 
                    options=capabilities.get("model_presets", []),
                    format_func=lambda x: PRESET_NAMES_TR.get(x, x)
                )
                
            # Seçilen model için anlık karar destek rehberi
            if algorithm:
                guidance = get_model_guidance(algorithm)
                with st.container(border=True):
                    st.markdown(f"💡 **Model Karar Destek Rehberi: `{algorithm}`**")
                    st.markdown(f"**🎯 En Uygun Senaryo:** {guidance['best_for']}")
                    st.caption(f"**✅ Güçlü:** {guidance['pros']}")
                    st.caption(f"**⚠️ Dikkat:** {guidance['cons']}")
        else:
            algorithm = None
            preset = None
            st.info("Lütfen önce geçerli bir görev seçin.")
            
        with st.expander("3. Önişleme Seçenekleri (İsteğe Bağlı)"):
            prep_strats = capabilities.get("preprocessing_strategies", {})
            missing_val = st.selectbox(
                "Eksik Değer Stratejisi", 
                ["none"] + prep_strats.get("missing_value", []),
                format_func=lambda x: PREP_TR.get(x, x)
            )
            encoding = st.selectbox(
                "Kategorik Kodlama (Encoding)", 
                ["none"] + prep_strats.get("encoding", []),
                format_func=lambda x: PREP_TR.get(x, x)
            )
            scaling = st.selectbox(
                "Ölçeklendirme (Scaling)", 
                ["none"] + prep_strats.get("scaling", []),
                format_func=lambda x: PREP_TR.get(x, x)
            )
            
        with st.expander("4. Doğrulama Seçenekleri (İsteğe Bağlı)"):
            val_options = capabilities.get("validation_options", ["holdout", "kfold"])
            val_strategy = st.selectbox(
                "Doğrulama Yöntemi", 
                val_options,
                format_func=lambda x: VAL_TR.get(x, x)
            )
            test_size = st.slider("Test Kümesi Oranı (Test Size)", 0.1, 0.5, 0.2, 0.05)
            
        submitted = st.form_submit_button("🚀 Deneyi Oluştur")
        
        if submitted:
            if not task or not algorithm:
                st.error("Lütfen görev ve algoritma seçimini tamamlayın.")
            else:
                req = {
                    "name": exp_name,
                    "dataset_id": selected_dataset_id,
                    "learning_type": learning_type,
                    "task": task,
                    "model": {
                        "algorithm": algorithm,
                        "preset": preset
                    },
                    "preprocessing": {
                        "missing_value": None if missing_val == "none" else missing_val,
                        "encoding": None if encoding == "none" else encoding,
                        "scaling": None if scaling == "none" else scaling
                    },
                    "validation": {
                        "strategy": val_strategy,
                        "test_size": test_size
                    }
                }
                
                try:
                    exp = client.create_experiment(req)
                    st.session_state.last_experiment_id = exp["id"]
                    st.success(f"✅ '{exp['name']}' ({exp['id']}) deneyi oluşturuldu! Eğitimi başlatmak için 'Çalıştırma & Sonuçlar' sayfasına geçebilirsiniz.")
                except AksisAPIError as e:
                    st.error(f"Hata: {str(e)}")

except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")
