import streamlit as st
from api.client import AksisClient

# Initialize state globally
if "client" not in st.session_state:
    st.session_state.client = AksisClient()

st.set_page_config(
    page_title="AKSIS - Kurumsal ML Platformu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit varsayılan Deploy butonunu ve menüsünü gizle
st.markdown("""
<style>
.stDeployButton, footer, #MainMenu {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Kurumsal Başlık
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0; text-align: left;">
    <h2 style="margin: 0; color: #1E3A8A; font-weight: 800; font-size: 1.5rem; letter-spacing: -0.5px;">⚡ AKSIS</h2>
    <p style="margin: 2px 0 0 0; color: #64748B; font-size: 0.8rem; font-weight: 500;">Kurumsal MLOps Platformu</p>
</div>
""", unsafe_allow_html=True)

# Menü navigasyonu için sayfa tanımları
pages = {
    "Genel Bakış": [
        st.Page("pages/0_home.py", title="Ana Sayfa", icon="🏠"),
        st.Page("pages/1_dashboard.py", title="Sistem Özeti", icon="📊"),
        st.Page("pages/2_datasets.py", title="Veri Setleri", icon="📁"),
    ],
    "Deney Yönetimi": [
        st.Page("pages/3_experiment_config.py", title="Deney Yapılandırma", icon="⚙️"),
        st.Page("pages/4_execution_results.py", title="Çalıştırma & Sonuçlar", icon="📈"),
    ],
    "Model & Dağıtım": [
        st.Page("pages/5_artifacts_inference.py", title="Modeller & Toplu Tahmin", icon="🚀"),
    ]
}

# Backend bağlantı kontrolü
st.sidebar.divider()
if not st.session_state.client.get_health():
    st.sidebar.error("⚠️ Backend Bağlantısı Yok")
else:
    st.sidebar.success("✅ Backend Servisi Aktif")

pg = st.navigation(pages)
pg.run()
