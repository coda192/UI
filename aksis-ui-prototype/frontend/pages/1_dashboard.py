import streamlit as st
from api.client import AksisAPIError

st.title("📊 System Dashboard")

client = st.session_state.client

try:
    capabilities = client.get_capabilities()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Capabilities")
        st.write("**Supported Learning Types:**")
        st.write(", ".join(capabilities.get("learning_types", [])))
        
        st.write("**Supported Tasks:**")
        for l_type, tasks in capabilities.get("tasks", {}).items():
            st.write(f"- {l_type}: {', '.join(tasks)}")
            
    with col2:
        st.subheader("Algorithms by Task")
        for task, algos in capabilities.get("algorithms", {}).items():
            with st.expander(f"{task.title()} Algorithms"):
                st.write(", ".join(algos))
                
    st.subheader("Recent Experiments")
    experiments = client.list_experiments()
    if experiments:
        st.dataframe(experiments)
    else:
        st.info("No experiments found.")
        
except AksisAPIError as e:
    st.error(str(e))
