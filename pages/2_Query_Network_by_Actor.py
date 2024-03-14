import streamlit as st

from streamlit_shared import *

def update_actor_state():
    st.write(st.session_state['actors'])

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

if not "keep_actor_query_graphic" in st.session_state:
    st.session_state["keep_actor_query_graphic"] = False          

# *********************************************************************************************************************

st.header("Query a Peace Process Network by Actor")

st.write("Here you can query your chosen peace process by selecting one or more actors from the drop-down menu.\
          If only one actor is chosen, the AND/OR is irrelevant.\
         Clicking on the Submit button will:")
st.text("1. Display a subset of the actor-agreement network based on the actors you selected.\n\
2. Display the key to the colour code of the network nodes.")

st.write('From here you can select another peace process or move on to any of the five other pages.\
          These pages do not have to be used in any particular order and you can return to the Select Peace Process\
          page to change the peace process at any point.')

# *********************************************************************************************************************

if len(st.session_state["pp_data_dict"]) > 0:

    st.subheader(':blue[' + st.session_state["pp_data_dict"]['pp_name'] + ']')
 
 
    #Query vertices using depth-first search
    with st.form("query"):
        st.write('Interface for formulating queries and providing users with insight into peace process actors and agreements.')
    
        # Get actors in alpha order
        actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
        actor_options = sorted(actor_options,key=lambda t:t[1])
        actor_options = [t[0] + ': ' + t[1] for t in actor_options]

        options_actor = st.multiselect(
        'Select one or more actors.',
        actor_options,st.session_state["selected_actors"])

        disabled = False
        #if len(options_actor) < 2:
        #    disabled = True

        operator=["AND: Show only the agreements (if any) to which all the selected actors are signatories", "OR: Show all agreements to which the selected actors are signatories"]
        select_operator=st.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=disabled, horizontal=False, captions=None, label_visibility="visible")

    # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted or st.session_state["keep_actor_query_graphic"]:
            if len(options_actor) > 0:

                st.session_state["keep_actor_query_graphic"] = True
                options = [v.split(':')[0] for v in options_actor]
                query_indices = [adj_vertices.index(vertex) for vertex in options]
                query_matrix,found_indices = get_query_matrix(query_indices,adj_matrix,max_depth=1,operator=select_operator.split(':')[0])
                display_networkx_graph(query_matrix,found_indices,adj_vertices,data_dict)

                st.caption('Network key:')
                st.caption(':red[Red nodes are agreements — identifier prefix AGT_]')
                st.caption(':blue[Blue nodes are country actors — identifier prefix CON_]')
                st.caption('Other colours represent different actor types, e.g., military, political, IGO etc.')

                st.divider()
                st.write(':violet[POTENTIAL FUNCTIONS]')
                st.write(':violet[Interactive network diagram with zoom, rearrangement, and access to node data]')
 
                st.session_state["selected_actors"] = options_actor
            else:
                st.write('Please select one or more actors.')

else:
    st.write('Please select a peace process in the Select Peace Process page.')