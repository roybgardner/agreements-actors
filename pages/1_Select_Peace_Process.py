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
st.header("Select a Peace Process")

st.write("The first step is to select a peace process from the dropdown list below.\
          Clicking on the Submit button will:")
st.write("1. Feed your selected peace process into the other pages of the demonstrator.")
st.write("2. Display the actor-agreement network of the selected peace process.")
st.write("3. Display the key to the colour code of the network nodes.")
st.write("4. Display the number of agreements and the number of actors in the selected peace process.")


st.write(':purple[POTENTIAL FUNCTIONS]')
st.write(':purple[Interactive network diagram with zoom, rearrangement, and access to node data]')

st.write('From here you can select another peace process or move on to any of the five other pages.\
          These pages do not have to be used in any particular order and you can return to the Select Peace Process\
          page to change the peace process at any point.')

# *********************************************************************************************************************


with st.form("peace_process"):

    # Select a peace process       
    st.write('Select a peace process from the list below')
    pp_names = get_peace_processes(data_dict)
    if len(pp_data_dict) > 0:
        index = pp_names.index(pp_data_dict['pp_name'])
    else:
        index = 0

    pp_selection=st.selectbox("", pp_names, index=index, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose a Peace Process", disabled=False, label_visibility="visible")
    
    if len(pp_data_dict) == 0:
        pp_data_dict = get_peace_process_data(pp_selection,data_dict)
    else:
        if st.session_state["pp_data_dict"]['pp_name'] != pp_selection:
            pp_data_dict = get_peace_process_data(pp_selection,data_dict)
            # Clear state for the query interface
            st.session_state["selected_actors"] = []       
            st.session_state["selected_agreements"] = []       
            st.session_state["selected_data_actor"] = ''
            st.session_state["selected_metadata_actor"] = ''
            st.session_state["selected_data_agreement"] = ''

    submitted = st.form_submit_button("Submit")
    if submitted or st.session_state["keep_network_graphic"]:
        st.session_state["keep_network_graphic"] = True

        if pp_data_dict['pp_matrix'].shape[0] == 0 or pp_data_dict['pp_matrix'].shape[1] == 0:
            st.write('ISSUE: peace process submatrix is empty')
            raise Exception('error')
        
        st.write('Network key:')
        st.caption(':red[Red nodes are agreements — identifier prefix AGT_]')
        st.caption(':blue[Blue nodes are country actors — identifier prefix CON_]')
        st.caption('Other colours represent different actor types, e.g., military, political, IGO etc.')

        s = 'Number of agreements in ' + pp_data_dict['pp_name'] + ':'
        st.write(s,pp_data_dict['pp_matrix'].shape[0])
        s = 'Number of actors in ' + pp_data_dict['pp_name'] + ':'
        st.write(s,pp_data_dict['pp_matrix'].shape[1])


        # Build peace process adjacency matrix and get adjacency matrix vertices and display graph using networkX
        adj_matrix,adj_vertices = adjacency_from_biadjacency(pp_data_dict)
        display_networkx_graph(adj_matrix,range(0,len(adj_vertices)),adj_vertices,data_dict)

        st.session_state["pp_data_dict"] = pp_data_dict
        st.session_state["adj_matrix"] = adj_matrix
        st.session_state["adj_vertices"] = adj_vertices
            
