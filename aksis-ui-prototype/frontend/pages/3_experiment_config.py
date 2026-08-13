import streamlit as st
from api.client import AksisAPIError

st.title("⚙️ Experiment Configuration")

client = st.session_state.client

try:
    capabilities = client.get_capabilities()
    
    # Dataset selection context
    datasets = client.get_datasets()
    if not datasets:
        st.warning("Please ensure datasets are available.")
        st.stop()
        
    dataset_options = {d["id"]: d["name"] for d in datasets}
    
    default_idx = 0
    if "selected_dataset_id" in st.session_state:
        ids = list(dataset_options.keys())
        if st.session_state.selected_dataset_id in ids:
            default_idx = ids.index(st.session_state.selected_dataset_id)
            
    selected_dataset_id = st.selectbox("Dataset", options=list(dataset_options.keys()), format_func=lambda x: dataset_options[x], index=default_idx)
    st.session_state.selected_dataset_id = selected_dataset_id
    
    # Resolve compatible tasks
    selected_ds_meta = next(d for d in datasets if d["id"] == selected_dataset_id)
    compatible_tasks = selected_ds_meta.get("compatible_tasks", [])
    
    if not compatible_tasks:
        st.error("Dataset has no compatible tasks configured.")
        st.stop()

    with st.form("experiment_config_form"):
        st.subheader("General")
        exp_name = st.text_input("Experiment Name", value=f"Exp_{selected_ds_meta['name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            learning_type = st.selectbox("Learning Type", capabilities.get("learning_types", []))
        with col2:
            # Filter tasks by dataset compatibility and learning type
            valid_tasks = [t for t in capabilities.get("tasks", {}).get(learning_type, []) if t in compatible_tasks]
            if not valid_tasks:
                st.warning(f"No compatible tasks for {learning_type} on this dataset.")
                task = None
            else:
                task = st.selectbox("Task", valid_tasks)
                
        st.subheader("Model Selection")
        if task:
            algos = capabilities.get("algorithms", {}).get(task, [])
            algorithm = st.selectbox("Algorithm", algos)
            preset = st.selectbox("Preset", capabilities.get("model_presets", []))
        else:
            algorithm = None
            preset = None
            st.info("Select a valid task first.")
            
        with st.expander("Preprocessing Options (Optional)"):
            prep_strats = capabilities.get("preprocessing_strategies", {})
            missing_val = st.selectbox("Missing Value Strategy", ["none"] + prep_strats.get("missing_value", []))
            encoding = st.selectbox("Encoding", ["none"] + prep_strats.get("encoding", []))
            scaling = st.selectbox("Scaling", ["none"] + prep_strats.get("scaling", []))
            
        with st.expander("Validation Options (Optional)"):
            val_options = capabilities.get("validation_options", ["holdout", "kfold"])
            val_strategy = st.selectbox("Strategy", val_options)
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            
        submitted = st.form_submit_button("Create Experiment")
        
        if submitted:
            if not task or not algorithm:
                st.error("Incomplete configuration.")
            else:
                req = {
                    "name": exp_name,
                    "dataset_id": selected_dataset_id,
                    "learning_type": learning_type,
                    "task": task,
                    "model": {
                        "algorithm": algorithm,
                        "preset": preset
                    },
                    "preprocessing": {
                        "missing_value": None if missing_val == "none" else missing_val,
                        "encoding": None if encoding == "none" else encoding,
                        "scaling": None if scaling == "none" else scaling
                    },
                    "validation": {
                        "strategy": val_strategy,
                        "test_size": test_size
                    }
                }
                
                try:
                    exp = client.create_experiment(req)
                    st.session_state.last_experiment_id = exp["id"]
                    st.success(f"Experiment {exp['id']} created! Go to Execution & Results page.")
                except AksisAPIError as e:
                    st.error(str(e))

except AksisAPIError as e:
    st.error(str(e))
