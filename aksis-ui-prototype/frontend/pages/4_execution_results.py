import streamlit as st
import time
from api.client import AksisAPIError
import pandas as pd

st.title("📈 Execution & Results")

client = st.session_state.client

try:
    experiments = client.list_experiments()
    if not experiments:
        st.info("No experiments found. Configure one first.")
        st.stop()
        
    exp_options = {e["id"]: f"{e['name']} ({e['status']})" for e in experiments}
    
    default_idx = 0
    if "last_experiment_id" in st.session_state:
        ids = list(exp_options.keys())
        if st.session_state.last_experiment_id in ids:
            default_idx = ids.index(st.session_state.last_experiment_id)
            
    selected_exp_id = st.selectbox("Select Experiment", options=list(exp_options.keys()), format_func=lambda x: exp_options[x], index=default_idx)
    st.session_state.last_experiment_id = selected_exp_id
    
    meta = client.get_experiment(selected_exp_id)
    status = meta.get("status")
    
    st.write(f"**Status:** `{status}`")
    
    if status == "configured":
        if st.button("Run Experiment"):
            client.run_experiment(selected_exp_id)
            st.success("Execution started...")
            st.rerun()
            
    elif status == "running":
        progress = st.progress(0)
        status_text = st.empty()
        
        # Poll
        for i in range(15):
            meta = client.get_experiment(selected_exp_id)
            if meta["status"] != "running":
                break
            status_text.text(f"Polling... {i}s")
            progress.progress(min((i+1)*10, 100))
            time.sleep(1)
            
        st.rerun()
        
    elif status == "completed":
        st.success("Experiment completed successfully.")
        
        results = client.get_experiment_results(selected_exp_id)
        
        st.header("Results Summary")
        summary = results.get("summary", {})
        if summary:
            cols = st.columns(len(summary))
            for col, (k, v) in zip(cols, summary.items()):
                col.metric(k.replace('_', ' ').title(), v)
                
        metrics = results.get("metrics", {})
        
        # Display typed metrics based on task
        task = results.get("task")
        has_gt = results.get("has_ground_truth", False)
        
        if task == "classification":
            st.subheader("Classification Metrics")
            cls_metrics = metrics.get("classification_metrics", {})
            mcols = st.columns(4)
            for i, (k, v) in enumerate(cls_metrics.items()):
                mcols[i % 4].metric(k.upper(), v)
                
        elif task == "regression":
            st.subheader("Regression Metrics")
            reg_metrics = metrics.get("regression_metrics", {})
            mcols = st.columns(3)
            for i, (k, v) in enumerate(reg_metrics.items()):
                mcols[i % 3].metric(k.upper(), v)
                
        elif task == "anomaly_detection":
            if has_gt:
                st.subheader("Labeled Anomaly Metrics")
                anom_metrics = metrics.get("anomaly_metrics", {})
                mcols = st.columns(3)
                for i, (k, v) in enumerate(anom_metrics.items()):
                    mcols[i % 3].metric(k.upper(), v)
            else:
                st.info("Unlabeled anomaly detection: No supervised accuracy/F1 metrics available.")
                
        st.header("Visualizations & Tables")
        
        tables = results.get("tables", {})
        for table_name, data in tables.items():
            st.write(f"**{table_name.replace('_', ' ').title()}**")
            st.dataframe(pd.DataFrame(data))
            
        visualizations = results.get("visualizations", [])
        for viz in visualizations:
            with st.expander(viz.get("title", viz.get("type"))):
                st.json(viz.get("data", {}))
                
except AksisAPIError as e:
    st.error(str(e))
