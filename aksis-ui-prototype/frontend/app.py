import streamlit as st
import time
from api.client import AksisClient
import pandas as pd

st.set_page_config(page_title="AKSIS Prototype", layout="wide")

st.title("AKSIS UI Prototype (Phase 1)")

client = AksisClient()

if not client.get_health():
    st.error("Cannot connect to FastAPI backend. Ensure it is running at http://127.0.0.1:8000")
    st.stop()

st.success("Connected to FastAPI Backend")

tab1, tab2, tab3 = st.tabs(["Capabilities", "Datasets", "Demo Execution Flow"])

with tab1:
    st.header("Capabilities")
    st.write("This data is driven entirely by the backend API (`/api/v1/capabilities`), ensuring Streamlit maintains no hardcoded lists of algorithms or tasks.")
    cap = client.get_capabilities()
    st.json(cap)

with tab2:
    st.header("Available Datasets")
    datasets = client.get_datasets()
    df = pd.DataFrame(datasets)
    st.dataframe(df)

with tab3:
    st.header("Demo Execution Flow")
    st.write("Test the asynchronous experiment execution loop using MockAksisService.")
    
    if st.button("Run Classification Demo"):
        with st.spinner("Configuring..."):
            # 1. Configure
            req = {
                "name": "Demo Churn Classification",
                "dataset_id": "ds_class_01",
                "learning_type": "supervised",
                "task": "classification",
                "model": {
                    "algorithm": "CatBoost"
                }
            }
            exp_meta = client.create_experiment(req)
            exp_id = exp_meta["id"]
            st.write(f"✅ Created Experiment: `{exp_id}` (Status: {exp_meta['status']})")
            
            # 2. Run
            client.run_experiment(exp_id)
            st.write(f"🚀 Execution started.")
            
            # 3. Poll Status
            status = "running"
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # For demo, loop until completed or timeout
            for i in range(10):
                meta = client.get_experiment(exp_id)
                status = meta["status"]
                status_text.text(f"Status: {status}...")
                
                if status == "completed":
                    progress_bar.progress(100)
                    break
                elif status == "failed":
                    st.error("Experiment failed!")
                    break
                
                progress_bar.progress((i + 1) * 10)
                time.sleep(1)
                
            if status == "completed":
                st.success("Experiment Completed!")
                # 4. Fetch Results
                results = client.get_experiment_results(exp_id)
                st.write("### Evaluation Results")
                
                # Show Metrics
                metrics = results.get("metrics", {}).get("classification_metrics", {})
                cols = st.columns(len(metrics))
                for col, (k, v) in zip(cols, metrics.items()):
                    col.metric(k.capitalize(), v)
                    
                # Show visual data payload
                st.write("### Visualization Payload")
                st.json(results.get("visualizations", []))
