import streamlit as st
import pandas as pd
from api.client import AksisAPIError

st.title("📊 Sistem Özeti & Registry Kataloğu")
st.caption("AKSIS bünyesinde kayıtlı makine öğrenmesi algoritmaları, veri setleri, önişleme yetenekleri ve deney geçmişi.")

client = st.session_state.client

TASK_NAMES_TR = {
    "classification": "Sınıflandırma (Classification)",
    "regression": "Regresyon (Regression)",
    "anomaly_detection": "Anomali Tespiti (Anomaly Detection)"
}

LEARNING_TYPES_TR = {
    "supervised": "Denetimli (Supervised)",
    "unsupervised": "Denetimsiz (Unsupervised)"
}

STATUS_TR = {
    "configured": "⚙️ Yapılandırıldı",
    "running": "⏳ Eğitiliyor...",
    "completed": "✅ Tamamlandı",
    "failed": "❌ Başarısız"
}

from utils.model_guidance import get_model_guidance

try:
    capabilities = client.get_capabilities()
    datasets = client.get_datasets()
    experiments = client.list_experiments()
    
    # Toplam algoritma sayısı hesabı
    algorithms_dict = capabilities.get("algorithms", {})
    total_algos = sum(len(v) for v in algorithms_dict.values())
    total_tasks = sum(len(v) for v in capabilities.get("tasks", {}).values())
    
    # 4 Üst Yönetici Metrik Kartı
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric("🧠 Kayıtlı Model & Algoritma", f"{total_algos} Adet")
    with m2:
        with st.container(border=True):
            st.metric("🎯 Desteklenen ML Görevi", f"{total_tasks} Görev")
    with m3:
        with st.container(border=True):
            st.metric("📁 Kayıtlı Veri Seti", f"{len(datasets)} Veri Seti")
    with m4:
        with st.container(border=True):
            st.metric("🧪 Kayıtlı Deney", f"{len(experiments)} Deney")

    st.divider()

    # Model & Algoritma Kataloğu (Registry View)
    st.subheader("📚 Kayıtlı Model & Algoritma Kataloğu (Registry & Rehber)")
    st.caption("Her algoritmanın güçlü/zayıf yönleri ve kurumsal karar destek için önerilen kullanım senaryoları.")
    
    task_keys = list(algorithms_dict.keys())
    tab_names = [f"🎯 {TASK_NAMES_TR.get(k, k.title())}" for k in task_keys]
    tabs = st.tabs(tab_names)
    
    for idx, (task_key, algos) in enumerate(algorithms_dict.items()):
        with tabs[idx]:
            st.markdown(f"**Bu görev için kullanıma hazır {len(algos)} algoritma tanımlıdır:**")
            
            # Algoritmaları 2 sütunlu detaylı kartlar halinde listele
            cols = st.columns(2)
            for i, algo in enumerate(algos):
                guidance = get_model_guidance(algo)
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"### ⚡ {algo}")
                        st.markdown(f"**🎯 En Uygun Senaryo:** {guidance['best_for']}")
                        with st.expander("🔍 Güçlü ve Zayıf Yönleri Gör", expanded=False):
                            st.markdown(f"**✅ Güçlü Yönler:**\n{guidance['pros']}")
                            st.markdown(f"**⚠️ Dikkat Edilmesi Gerekenler:**\n{guidance['cons']}")
                        
    st.divider()
    
    # Preprocessing & Validation Stratejileri
    st.subheader("⚙️ Standart Önişleme & Doğrulama Yetenekleri")
    
    prep_col1, prep_col2, prep_col3 = st.columns(3)
    prep_strats = capabilities.get("preprocessing_strategies", {})
    
    with prep_col1:
        with st.container(border=True):
            st.markdown("#### 🩹 Eksik Değer (Missing Value)")
            for s in prep_strats.get("missing_value", []):
                st.write(f"- `{s}`")
                
    with prep_col2:
        with st.container(border=True):
            st.markdown("#### 🔤 Kategorik Kodlama (Encoding)")
            for s in prep_strats.get("encoding", []):
                st.write(f"- `{s}`")
                
    with prep_col3:
        with st.container(border=True):
            st.markdown("#### ⚖️ Ölçeklendirme (Scaling)")
            for s in prep_strats.get("scaling", []):
                st.write(f"- `{s}`")

    st.divider()

    # Son Deneyler Tablosu
    st.subheader("🕒 Son Deneyler & İzlenebilirlik")
    if experiments:
        df_exp = pd.DataFrame(experiments)
        if "status" in df_exp.columns:
            df_exp["status"] = df_exp["status"].map(lambda s: STATUS_TR.get(s, s))
        df_exp = df_exp.rename(columns={
            "id": "Deney ID",
            "name": "Deney Adı",
            "status": "Güncel Durum",
            "created_at": "Oluşturulma Tarihi"
        })
        st.dataframe(df_exp, use_container_width=True)
    else:
        st.info("💡 Henüz kayıtlı bir deney bulunmamaktadır. 'Deney Yapılandırma' sayfasından yeni bir deney oluşturabilirsiniz.")
        
except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")
