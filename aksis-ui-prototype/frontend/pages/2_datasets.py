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
        
        # 4. VERİ PROFİLİ & İSTATİSTİKSEL ÖZET (Independent Profiling Pipeline Slot)
        st.subheader("📊 Veri Profili & İstatistiksel Özet")
        
        profile = client.get_dataset_profile(selected_id)
        
        if not profile:
            st.info("Henüz analiz çalıştırılmadı.")
        else:
            p_m1, p_m2, p_m3, p_m4, p_m5 = st.columns(5)
            with p_m1:
                row_count = profile.get("row_count", 0)
                st.metric("Satır Sayısı", f"{row_count:,}")
            with p_m2:
                col_count = profile.get("column_count", 0)
                st.metric("Sütun Sayısı", str(col_count))
            with p_m3:
                mem_mb = profile.get("memory_usage_mb", 0.0)
                st.metric("Bellek Kullanımı", f"{mem_mb:.2f} MB")
            with p_m4:
                cols_missing = profile.get("columns_with_missing", 0)
                st.metric("Eksik Değerli Sütun", str(cols_missing))
            with p_m5:
                total_missing = profile.get("total_missing_values", 0)
                st.metric("Toplam Eksik Değer", f"{total_missing:,}")
                
            cols_list = profile.get("columns", [])
            if cols_list:
                df_profile_cols = pd.DataFrame(cols_list)
                display_cols = ["name", "detected_type", "dtype", "missing_count", "missing_percentage"]
                existing_cols = [c for c in display_cols if c in df_profile_cols.columns]
                if existing_cols:
                    df_profile_cols = df_profile_cols[existing_cols]
                rename_map = {
                    "name": "Sütun Adı",
                    "detected_type": "Algılanan Tip",
                    "dtype": "Veri Tipi",
                    "missing_count": "Eksik Değer Sayısı",
                    "missing_percentage": "Eksik Değer (%)"
                }
                df_profile_cols = df_profile_cols.rename(columns=rename_map)
                st.dataframe(df_profile_cols, use_container_width=True)
            else:
                st.caption("Sütun profil bilgisi bulunamadı.")

except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")
