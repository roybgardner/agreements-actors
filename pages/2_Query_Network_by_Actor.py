import streamlit as st

from streamlit_shared import *

def update_actor_state():
    st.write(st.session_state['selected_actors'])

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

if not "keep_actor_query_graphic" in st.session_state:
    st.session_state["keep_actor_query_graphic"] = False          

# *********************************************************************************************************************

st.header("Query a Peace Process Network by Actor")

# *********************************************************************************************************************

if len(st.session_state["pp_data_dict"]) > 0:
    st.subheader(st.session_state["pp_data_dict"]['pp_name'])

    #Query vertices using depth-first search
    with st.form("query"):
        st.write('Interface for formulating queries and providing users with insight into peace process actors and agreements.')
    
        # Get actors in alpha order
        actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
        actor_options = sorted(actor_options,key=lambda t:t[1])
        actor_options = [t[0] + ': ' + t[1] for t in actor_options]

        options_actor = st.multiselect(
        'Select one or more actors.',
        actor_options,st.session_state["selected_actors"],on_change=update_actor_state,key='selected_actors')

        disabled = False
        if len(options_actor) < 2:
            disabled = True

        operator=["Show only agreements signed by all selected actors", "Show all agreements signed by each of the selected actors"]
        select_operator=st.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=disabled, horizontal=False, captions=None, label_visibility="visible")

    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted or st.session_state["keep_actor_query_graphic"]:
            if len(options_actor) == 0:
                st.session_state["selected_actors"] = []
                st.session_state["keep_actor_query_graphic"] = False
            else:
                st.session_state["keep_actor_query_graphic"] = True
                options = [v.split(':')[0] for v in options_actor]
                query_indices = [adj_vertices.index(vertex) for vertex in options]
                query_matrix,found_indices = get_query_matrix(query_indices,adj_matrix,max_depth=1,operator=select_operator)
                display_networkx_graph(query_matrix,found_indices,adj_vertices,data_dict)
                st.session_state["selected_actors"] = options_actor

else:
    st.write('Please select a peace process in the Select Peace Process page.')