import streamlit as st
import pandas as pd
from api.client import AksisAPIError

st.title("📁 Veri Setleri")

client = st.session_state.client

TASK_NAMES_TR = {
    "classification": "Sınıflandırma",
    "regression": "Regresyon",
    "anomaly_detection": "Anomali Tespiti"
}

try:
    datasets = client.get_datasets()
    
    if not datasets:
        st.info("Kullanılabilir veri seti bulunamadı.")
        st.stop()
        
    dataset_options = {d["id"]: d["name"] for d in datasets}
    
    # Oturum durumunda önceden seçili veri seti varsa onu kullan
    default_idx = 0
    if "selected_dataset_id" in st.session_state:
        ids = list(dataset_options.keys())
        if st.session_state.selected_dataset_id in ids:
            default_idx = ids.index(st.session_state.selected_dataset_id)
            
    selected_id = st.selectbox(
        "İncelenecek Veri Setini Seçin", 
        options=list(dataset_options.keys()), 
        format_func=lambda x: dataset_options[x],
        index=default_idx
    )
    
    if selected_id:
        st.session_state.selected_dataset_id = selected_id
        
        ds = client.get_dataset(selected_id)
        
        if ds.get("display_name"):
            st.subheader(f"📌 {ds['display_name']}")
        if ds.get("description"):
            st.markdown(f"*{ds['description']}*")
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Satır Sayısı", f"{ds.get('row_count', 0):,}")
        col2.metric("Sütun Sayısı", ds.get("column_count"))
        target_val = ds.get("target")
        col3.metric("Hedef Değişken (Target)", target_val if target_val else "Yok (Etiketsiz)")
        
        comp_tasks = [TASK_NAMES_TR.get(t, t) for t in ds.get("compatible_tasks", [])]
        st.write(f"**Uyumlu Görevler:** {', '.join(comp_tasks)}")
        
        if ds.get("identifier_columns"):
            st.write(f"**Kimlik / ID Sütunları:** {', '.join(ds.get('identifier_columns'))}")
        
        st.subheader("Sütun Detayları ve Veri Tipleri")
        df_cols = pd.DataFrame(ds.get("columns", []))
        df_cols = df_cols.rename(columns={
            "name": "Sütun Adı",
            "dtype": "Veri Tipi",
            "missing_count": "Eksik Değer Sayısı"
        })
        st.dataframe(df_cols, use_container_width=True)

except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")
