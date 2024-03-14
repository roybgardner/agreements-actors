import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

if not "keep_agreement_query_graphic" in st.session_state:
    st.session_state["keep_agreement_query_graphic"] = False          

# *********************************************************************************************************************

st.header("Query a Peace Process Network")

# *********************************************************************************************************************

if len(st.session_state["pp_data_dict"]) > 0:
    st.subheader(':blue[' + st.session_state["pp_data_dict"]['pp_name'] + ']')

    #Query vertices using depth-first search
    with st.form("query"):
        st.write('Interface for formulating queries and providing users with insight into peace process actors and agreements.')
    
        # Get agreements in date order
        agreement_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5],data_dict['dates_dict'][vertex_id]) for vertex_id in pp_data_dict['pp_agreement_ids']]
        agreement_options = sorted(agreement_options,key=lambda t:t[2])
        agreement_options = [t[0] + ': ' + t[1] for t in agreement_options]

        options_agreement = st.multiselect(
        'Select one or more agreements',
        agreement_options,st.session_state["selected_agreements"])

        disabled = False
        #if len(options_agreement) < 2:
        #    disabled = True

        operator=["AND: Show only the actors (if any) that are signatories to all the selected agreements", "OR: Show all actors that are signatories to the selected agreements"]
        select_operator=st.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=disabled, horizontal=False, captions=None, label_visibility="visible")

    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted or st.session_state["keep_agreement_query_graphic"]:
            st.write('Network key:')
            st.caption(':red[Red nodes are agreements — identifier prefix AGT_]')
            st.caption(':blue[Blue nodes are country actors — identifier prefix CON_]')
            st.caption(':rainbow[Other colours represent different actor types, e.g., military, political, IGO etc]')

            st.session_state["keep_agreement_query_graphic"] = True
            options = [v.split(':')[0] for v in options_agreement]
            query_indices = [adj_vertices.index(vertex) for vertex in options]
            query_matrix,found_indices = get_query_matrix(query_indices,adj_matrix,max_depth=1,operator=select_operator.split(':')[0])
            display_networkx_graph(query_matrix,found_indices,adj_vertices,data_dict)
            st.session_state["selected_agreements"] = options_agreement

else:
    st.write('Please select a peace process in the Select Peace Process page.')