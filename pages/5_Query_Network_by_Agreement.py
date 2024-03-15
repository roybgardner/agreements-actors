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
        
        st.write("Here you can query your chosen peace process actor-agreement network by selecting one or more agreements from the drop-down menu.\
                If only one agreement is chosen, then AND/OR is irrelevant, otherwise:")
        st.text("AND means show actors that signed every one of the selected agreements.\n\
OR means show actors that signed any one of the selected agreements.")
        st.write("Clicking on the Submit button will:")
        st.text("1. Display the actor-agreement network for the selected agreements.\n\
2. Display the key to the colour code of the network nodes.")
        st.write('You can stay on this page adding or removing agreements from your list and re-submitting.')

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

        operator=["AND: Show actors that signed every one of the selected agreements", "OR: Show actors that signed any one of the selected agreements"]
        select_operator=st.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=disabled, horizontal=False, captions=None, label_visibility="visible")

    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted or st.session_state["keep_agreement_query_graphic"]:
            st.divider()
            if len(options_agreement) > 0:

                st.session_state["keep_agreement_query_graphic"] = True
                options = [v.split(':')[0] for v in options_agreement]
                query_indices = [adj_vertices.index(vertex) for vertex in options]
                query_matrix,found_indices = get_query_matrix(query_indices,adj_matrix,max_depth=1,operator=select_operator.split(':')[0])
                display_networkx_graph(query_matrix,found_indices,adj_vertices,data_dict)

                st.caption('Network key:')
                st.caption(':red[Red nodes are agreements — identifier prefix AGT_]')
                st.caption(':blue[Blue nodes are country actors — identifier prefix CON_]')
                st.caption('Other colours represent different actor types, e.g., military, political, IGO etc.')

                st.session_state["selected_agreements"] = options_agreement
            else:
                st.write('Please select one or more agreements.')

    st.divider()
    st.write(':violet[POTENTIAL FUNCTIONS]')
    st.write(':violet[Interactive network diagram with zoom, rearrangement, and access to node data]')

else:
    st.write('Please select a peace process in the Select Peace Process page.')