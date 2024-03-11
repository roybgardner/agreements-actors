import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

if not "keep_network_graphic" in st.session_state:
    st.session_state["keep_network_graphic"] = False          

# *********************************************************************************************************************


#define css for different classes 
st.markdown("""
    <style>
    .maintitle {
        letter-spacing: 1px;
        color: #000080;
        font-size: 45px;
        font-family: "Lucida Grande", Verdana, Helvetica, Arial, sans-serif;
        font-weight: 100;
        
    }
    .info {
        
        letter-spacing: 1px;
        color: #000080;
        font-size: 15px;
        font-family: "Lucida Grande", Verdana, Helvetica, Arial, sans-serif;
        font-weight: 100;
        
    }    
    </style>
    """, unsafe_allow_html=True)


st.markdown('<p class="maintitle">Signatories Network Analysis</p>', unsafe_allow_html=True)
st.header("Select peace process")

# *********************************************************************************************************************


with st.form("peace_process"):
    st.subheader("View peace process agreement-actor network")

    # Select a peace process       
    st.write('Select a peace process from the list below')
    pp_names = get_peace_processes(data_dict)
    if len(pp_data_dict) > 0:
        index = pp_names.index(pp_data_dict['pp_name'])
    else:
        index = 0

    pp_selection=st.selectbox("", pp_names, index=index, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose a Peace Process", disabled=False, label_visibility="visible")
    pp_data_dict = get_peace_process_data(pp_selection,data_dict)

    submitted = st.form_submit_button("Submit")
    if submitted or st.session_state["keep_network_graphic"]:
        st.session_state["keep_network_graphic"] = True

        if pp_data_dict['pp_matrix'].shape[0] == 0 or pp_data_dict['pp_matrix'].shape[1] == 0:
            st.write('ISSUE: peace process submatrix is empty')
            raise Exception('error')

        st.write('Number of agreements in peace process:',pp_data_dict['pp_matrix'].shape[0])
        st.write('Number of actors in peace process:',pp_data_dict['pp_matrix'].shape[1])


        # Build peace process adjacency matrix and get adjacency matrix vertices and display graph using networkX
        adj_matrix,adj_vertices = adjacency_from_biadjacency(pp_data_dict)
        display_networkx_graph(adj_matrix,range(0,len(adj_vertices)),adj_vertices,data_dict)

        st.session_state["pp_data_dict"] = pp_data_dict
        st.session_state["adj_matrix"] = adj_matrix
        st.session_state["adj_vertices"] = adj_vertices
            
