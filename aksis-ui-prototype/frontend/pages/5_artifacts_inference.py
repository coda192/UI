import streamlit as st
import pandas as pd
from api.client import AksisAPIError

st.title("🚀 Model Artefaktları & Toplu Tahmin")

client = st.session_state.client

try:
    st.header("📦 Model Artefaktları")
    artifacts = client.list_artifacts()
    
    if not artifacts:
        st.info("Kayıtlı model artefaktı bulunamadı. Lütfen önce 'Çalıştırma & Sonuçlar' sayfasında bir deneyi tamamlayın.")
    else:
        df_art = pd.DataFrame(artifacts)
        df_art = df_art.rename(columns={
            "id": "Artefakt Kimliği",
            "name": "Artefakt Adı",
            "type": "Tür",
            "experiment_id": "İlgili Deney ID",
            "created_at": "Oluşturulma Tarihi"
        })
        st.dataframe(df_art, use_container_width=True)
        
        st.divider()
        st.header("🎯 Toplu Tahmin")
        
        model_artifacts = [a for a in artifacts if a["type"] == "model"]
        if not model_artifacts:
            st.warning("Tahmin için kullanılabilir model artefaktı bulunamadı.")
            st.stop()
            
        datasets = client.get_datasets()
        
        art_options = {a["id"]: f"{a['name']} ({a['experiment_id']})" for a in model_artifacts}
        ds_options = {d["id"]: d["name"] for d in datasets}
        
        col1, col2 = st.columns(2)
        with col1:
            sel_artifact = st.selectbox(
                "Kullanılacak Eğitilmiş Modeli Seçin", 
                options=list(art_options.keys()), 
                format_func=lambda x: art_options[x]
            )
        with col2:
            sel_dataset = st.selectbox(
                "Tahmin Yapılacak Veri Setini Seçin", 
                options=list(ds_options.keys()), 
                format_func=lambda x: ds_options[x]
            )
        
        if st.button("🚀 Tahminleri Üret"):
            with st.spinner("Model veri seti üzerinde çalıştırılıyor..."):
                resp = client.run_inference({
                    "artifact_id": sel_artifact,
                    "dataset_id": sel_dataset
                })
                
                st.success(f"✅ Tahmin süreci tamamlandı! Toplam Tahmin Sayısı: {resp.get('total_predictions')}")
                st.write("**Tahmin Sonuçları Önizlemesi:**")
                st.dataframe(pd.DataFrame(resp.get("predictions_preview", [])), use_container_width=True)
                
except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")
