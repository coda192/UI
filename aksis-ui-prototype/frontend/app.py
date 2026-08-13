import streamlit as st
from api.client import AksisClient

# Initialize state globally
if "client" not in st.session_state:
    st.session_state.client = AksisClient()

st.set_page_config(page_title="AKSIS UI Prototype", layout="wide")

# Pages definition for sidebar navigation
pages = {
    "Overview": [
        st.Page("pages/1_dashboard.py", title="Dashboard", icon="📊"),
        st.Page("pages/2_datasets.py", title="Datasets", icon="📁"),
    ],
    "Experiments": [
        st.Page("pages/3_experiment_config.py", title="Configuration", icon="⚙️"),
        st.Page("pages/4_execution_results.py", title="Execution & Results", icon="📈"),
    ],
    "Deployment": [
        st.Page("pages/5_artifacts_inference.py", title="Artifacts & Inference", icon="🚀"),
    ]
}

# Add a health check warning if backend is down
if not st.session_state.client.get_health():
    st.sidebar.error("⚠️ Backend Unavailable. Please start FastAPI.")
else:
    st.sidebar.success("✅ Backend Connected")

pg = st.navigation(pages)
pg.run()
