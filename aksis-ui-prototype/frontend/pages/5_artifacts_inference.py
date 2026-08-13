import streamlit as st
import pandas as pd
from api.client import AksisAPIError

st.title("🚀 Artifacts & Inference")

client = st.session_state.client

try:
    st.header("Artifacts")
    artifacts = client.list_artifacts()
    
    if not artifacts:
        st.info("No artifacts found. Run an experiment first.")
    else:
        st.dataframe(pd.DataFrame(artifacts))
        
        st.header("Batch Inference")
        
        model_artifacts = [a for a in artifacts if a["type"] == "model"]
        if not model_artifacts:
            st.warning("No model artifacts available for inference.")
            st.stop()
            
        datasets = client.get_datasets()
        
        art_options = {a["id"]: f"{a['name']} ({a['experiment_id']})" for a in model_artifacts}
        ds_options = {d["id"]: d["name"] for d in datasets}
        
        sel_artifact = st.selectbox("Select Model", options=list(art_options.keys()), format_func=lambda x: art_options[x])
        sel_dataset = st.selectbox("Select Inference Dataset", options=list(ds_options.keys()), format_func=lambda x: ds_options[x])
        
        if st.button("Run Inference"):
            with st.spinner("Running batch inference..."):
                resp = client.run_inference({
                    "artifact_id": sel_artifact,
                    "dataset_id": sel_dataset
                })
                
                st.success(f"Inference Completed! Total Predictions: {resp.get('total_predictions')}")
                st.write("**Prediction Preview:**")
                st.dataframe(pd.DataFrame(resp.get("predictions_preview", [])))
                
except AksisAPIError as e:
    st.error(str(e))
