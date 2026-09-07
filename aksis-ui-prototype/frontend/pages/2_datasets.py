import streamlit as st
import pandas as pd
from api.client import AksisAPIError

st.title("📁 Veri Setleri")
st.caption("AKSIS Çerçevesinde kayıtlı veri setleri, şemalar ve teknik meta veriler.")

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
        
    dataset_options = {
        d["id"]: f"{d.get('display_name')} ({d['id']})" if d.get("display_name") else d["id"]
        for d in datasets
    }
    
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
        
        # 1. BAŞLIK VE HİYERARŞİ (DataSpec Display Name ve Raw ID)
        display_title = ds.get("display_name") or ds.get("id")
        raw_id = ds.get("id")
        
        st.markdown(f"## 📁 {display_title}")
        if ds.get("display_name") and ds.get("display_name") != raw_id:
            st.caption(f"`{raw_id}`")
            
        # 2. AÇIKLAMA BÖLÜMÜ (Doğrudan Başlığın Altında)
        desc = ds.get("description")
        if desc:
            st.write(desc)
        else:
            st.caption("Açıklama sağlanmamış.")
            
        st.divider()
        
        # 3. STATİK METAVERİ VE TEKNİK DETAYLAR (DataSpec Alanları)
        st.subheader("⚙️ Veri Seti Tanımı & Teknik Özellikler")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_val = ds.get("target")
            st.metric("Hedef Değişken (Target)", target_val if target_val else "Yok (Etiketsiz)")
        with col2:
            source_val = ds.get("source")
            st.metric("Veri Kaynağı (Source)", source_val if source_val else "Belirtilmemiş")
        with col3:
            local_val = ds.get("local_data")
            if local_val is True:
                local_str = "Evet (Yerel)"
            elif local_val is False:
                local_str = "Hayır (Uzak / Stream)"
            else:
                local_str = "Belirtilmemiş"
            st.metric("Yerel Veri (Local Data)", local_str)
            
        if ds.get("columns_to_use"):
            st.write(f"**Kullanılacak Sütunlar (Columns to Use):** {', '.join(ds.get('columns_to_use'))}")
            
        if ds.get("compatible_tasks"):
            comp_tasks = [TASK_NAMES_TR.get(t, t) for t in ds.get("compatible_tasks", [])]
            st.write(f"**Uyumlu Görevler:** {', '.join(comp_tasks)}")
            
        if ds.get("identifier_columns"):
            st.write(f"**Kimlik / ID Sütunları:** {', '.join(ds.get('identifier_columns'))}")
            
        st.divider()
        
        # 4. ÇALIŞMA ZAMANI İSTATİSTİKLERİ (Runtime / Data-Derived Statistics)
        st.subheader("📊 Çalışma Zamanı Veri İstatistikleri & Sütun Analizi")
        
        row_count = ds.get("row_count")
        col_count = ds.get("column_count")
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.metric("Satır Sayısı", f"{row_count:,}" if row_count is not None else "Hesaplanmadı / Mevcut Değil")
        with c_m2:
            st.metric("Sütun Sayısı", str(col_count) if col_count is not None else "Hesaplanmadı / Mevcut Değil")
            
        columns_data = ds.get("columns", [])
        if columns_data:
            df_cols = pd.DataFrame(columns_data)
            df_cols = df_cols.rename(columns={
                "name": "Sütun Adı",
                "dtype": "Veri Tipi",
                "missing_count": "Eksik Değer Sayısı"
            })
            st.dataframe(df_cols, use_container_width=True)
        else:
            st.caption("Detaylı sütun şeması ve eksik değer istatistikleri henüz hesaplanmamış.")

except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")
