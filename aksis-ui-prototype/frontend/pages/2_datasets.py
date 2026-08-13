import streamlit as st
import pandas as pd
from api.client import AksisAPIError

st.title("📁 Datasets")

client = st.session_state.client

try:
    datasets = client.get_datasets()
    
    if not datasets:
        st.info("No datasets available.")
        st.stop()
        
    dataset_options = {d["id"]: d["name"] for d in datasets}
    
    # Check session state for pre-selection
    default_idx = 0
    if "selected_dataset_id" in st.session_state:
        ids = list(dataset_options.keys())
        if st.session_state.selected_dataset_id in ids:
            default_idx = ids.index(st.session_state.selected_dataset_id)
            
    selected_id = st.selectbox(
        "Select Dataset", 
        options=list(dataset_options.keys()), 
        format_func=lambda x: dataset_options[x],
        index=default_idx
    )
    
    if selected_id:
        st.session_state.selected_dataset_id = selected_id
        
        ds = client.get_dataset(selected_id)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", ds.get("row_count"))
        col2.metric("Columns", ds.get("column_count"))
        col3.metric("Target", ds.get("target", "None"))
        
        st.write("**Compatible Tasks:**", ", ".join(ds.get("compatible_tasks", [])))
        
        st.subheader("Columns Metadata")
        df_cols = pd.DataFrame(ds.get("columns", []))
        st.dataframe(df_cols, use_container_width=True)

except AksisAPIError as e:
    st.error(str(e))
