import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

if not "keep_query_graphic" in st.session_state:
    st.session_state["keep_query_graphic"] = False          

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
st.header("Query a Peace Process Network")

# *********************************************************************************************************************

if len(st.session_state["pp_data_dict"]) > 0:

    #Query vertices using depth-first search
    with st.form("query"):
        st.write('Interface for formulating queries and providing users with insight into peace process actors and agreements.')
        st.write('Select actors (in alpha order) with agreements (in date order) using the selectors below.')
        st.write('Mixing and matching actor and agreements is supported but might not be sensible.')
    
        # Get actors in alpha order
        actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
        actor_options = sorted(actor_options,key=lambda t:t[1])
        actor_options = [t[0] + ': ' + t[1] for t in actor_options]

        # Get agreements in date order
        agreement_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5],data_dict['dates_dict'][vertex_id]) for vertex_id in pp_data_dict['pp_agreement_ids']]
        agreement_options = sorted(agreement_options,key=lambda t:t[2])
        agreement_options = [t[0] + ': ' + t[1] for t in agreement_options]

        options_actor = st.multiselect(
        'Select zero or more actors.',
        actor_options,
        st.session_state["selected_actors"])

        options_agreement = st.multiselect(
        'Select zero or more agreements',
        agreement_options,
        st.session_state["selected_agreements"])

        operator=["AND", "OR"]
        select_operator=st.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, horizontal=False, captions=None, label_visibility="visible")
        #depth=st.slider("Select depth", min_value=1, max_value=2, value=1, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted or st.session_state["keep_query_graphic"]:
            st.session_state["keep_query_graphic"] = True
            options = [v.split(':')[0] for v in options_actor]
            options.extend([v.split(':')[0] for v in options_agreement])
            query_indices = [adj_vertices.index(vertex) for vertex in options]
            query_matrix,found_indices = get_query_matrix(query_indices,adj_matrix,max_depth=1,operator=select_operator)
            display_networkx_graph(query_matrix,found_indices,adj_vertices,data_dict)

else:
    st.write('Please select a peace process in the Select Peace Process page.')