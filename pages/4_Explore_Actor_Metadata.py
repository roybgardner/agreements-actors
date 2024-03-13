import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]


# *********************************************************************************************************************

st.header("Explore Actor Metadata")
st.subheader(st.session_state["pp_data_dict"]['pp_name'])

# *********************************************************************************************************************



if len(st.session_state["pp_data_dict"]) > 0:
    
# *********************************************************************************************************************
    st.divider()
    st.subheader('Agreement year')

    st.session_state["keep_year_graphic"] = True

    labels = [data_dict['vertices_dict'][v][5] for v in pp_data_dict['pp_actor_ids']]
    z = list(zip(range(0,len(labels)),labels))
    z = sorted(z,key=lambda t:t[1])

    # Get a sorted list of years

    pp_ag_ids = pp_data_dict['pp_agreement_ids']

    year_list = []
    for i,agreement_id in enumerate(pp_ag_ids):
        if not agreement_id in data_dict['dates_dict']:
            continue
        ag_year = int(str(data_dict['dates_dict'][agreement_id])[0:4])
        year_list.append(ag_year)
    # Sort by year    
    year_list = sorted(set(year_list))
    print(year_list)

    year_matrix = np.zeros((len(pp_data_dict['pp_actor_ids']),len(year_list)))

    matrix_t = pp_data_dict['pp_matrix'].T
    for i,row in enumerate(matrix_t):
        for j,v in enumerate(row):
            if v == 0:
                continue
            agreement_id = pp_data_dict['pp_agreement_ids'][j]
            year = int(str(data_dict['dates_dict'][agreement_id])[0:4])
            year_index = year_list.index(year)
            year_matrix[i][year_index] += 1
            
    # Get matrix in actor alpha order
    ordered_year_matrix = []
    for t in z:
        ordered_year_matrix.append(year_matrix[t[0]])
        
    ordered_year_matrix = np.array(ordered_year_matrix)
            
            
    fig = plt.figure(figsize=(16,16),layout="constrained")
    plt.imshow(ordered_year_matrix,aspect='auto',cmap=plt.cm.Blues)
    plt.xticks(range(0,len(year_list)),year_list,fontsize='xx-large',rotation=90)
    plt.yticks(range(0,len(labels)),[t[1] for t in z],fontsize='x-large')
    plt.xlabel('Year',fontsize='xx-large')
    cbar = plt.colorbar()
    cbar.set_label('Signed in year',rotation=270,labelpad=15,fontsize='xx-large')
    st.pyplot(fig)

# *********************************************************************************************************************
    st.divider()

    with st.form("actors_metadata"):
    
        # Get actors in alpha order
        actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
        actor_options = sorted(actor_options,key=lambda t:t[1])
        actor_options = [t[1] + ': ' + t[0] for t in actor_options]

        # Get currently selected actor if any
        if len(st.session_state["selected_metadata_actor"]) > 0:
            index = actor_options.index(st.session_state["selected_metadata_actor"])
        else:
            index = 0
        index = 0

        actor_option = st.selectbox(
        'Select an actor:',
        actor_options,index=index)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state["selected_metadata_actor"] = actor_option
            actor = actor_option.split(': ')[1]
            actor_label = get_actor_name(actor,data_dict) + ' ' + actor
            actor_vector = year_matrix[pp_data_dict['pp_actor_ids'].index(actor)]

            fig = plt.figure(figsize=(8,8))
            y = actor_vector
            x = range(0,len(y))
            plt.bar(x,y)
            plt.xlabel('Year',fontsize='xx-large')
            plt.xticks(range(0,len(year_list)),year_list,fontsize='xx-large',rotation=90)
            plt.ylabel('Number of agreements signed',fontsize='xx-large')
            yint = range(0, math.ceil(max(y))+1)
            plt.yticks(yint,fontsize='xx-large')
            st.pyplot(fig)

else:
    st.write('Please select a peace process in the Select Peace Process page.')
