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
    plt.ylabel('Actor',fontsize='large')
    plt.xlabel('Year',fontsize='xx-large')
    cbar = plt.colorbar()
    cbar.set_label('Signed in year',rotation=270,labelpad=15,fontsize='xx-large')
    st.pyplot(fig)

# *********************************************************************************************************************
    st.divider()
    st.subheader('Agreement stage')

    st.write('Key to agreement stages:')
    st.caption('Level 1: Ceasefire related')
    st.caption('Level 2: Pre-negotiation process')
    st.caption('Level 3: Partial Framework - substantive')
    st.caption('Level 4: Comprehensive Framework - substantive')
    st.caption('Level 5: Implementation Renegotiation/Renewal')


    # Stage analysis
    stage_dict = {}
    stage_dict['Cea'] = [1,'Ceasefire related']
    stage_dict['Pre'] = [2,'Pre-negotiation process']
    stage_dict['SubPar'] = [3,'Partial Framework - substantive']
    stage_dict['SubComp'] = [4,'Comprehensive Framework - substantive']
    stage_dict['Ren'] = [5,'Implementation Renegotiation/Renewal']
    stage_dict['Imp'] = [5,'Implementation Renegotiation/Renewal']
    stage_dict['Oth'] = [0,'']


    # Map agreements on to stages
    stage_map = {}
    for i,agreement_id in enumerate(pp_data_dict['pp_agreement_ids']):
        ss_id = agreement_id.split('_')[1]
        if ss_id in data_dict['agreements_dict']:
            stage_map[i] = stage_dict[data_dict['agreements_dict'][ss_id]['Stage']][0]
        else:
            stage_map[i] = 0

    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    actor_diag = np.diag(co_matrices[0])

    # Plot
    labels = [data_dict['vertices_dict'][v][5] for v in pp_data_dict['pp_actor_ids']]
    z = list(zip(labels,actor_diag))
    z = sorted(z,key=lambda t:t[1])
    values = [t[1] for t in z]
                    
    fig = plt.figure(figsize=(16,16),layout="constrained")

    gs = GridSpec(1, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.barh(range(0,len(actor_diag)),values)
    ax1.set_yticks(range(0,len(actor_diag)),[t[0] for t in z],fontsize='large')
    ax1.set_xlim(0,max(values)+5)
    ax1.margins(y=0)
    ax1.set_title('All Stages',fontsize='xx-large')

    stage_levels = [1,2,3,4,5]
    for i,stage_level in enumerate(stage_levels):
        stage_agreement_indices = [k for k,v in stage_map.items() if v == stage_level]
        stage_matrix = pp_data_dict['pp_matrix'][np.ix_(stage_agreement_indices)]
        co_matrices = get_cooccurrence_matrices(stage_matrix)
        # Same order as all agreements so y-axes are consistent
        actor_diag = np.diag(co_matrices[0])
        x = list(zip(labels,actor_diag))
        x = sorted(x,key=lambda t:[g[0] for g in z].index(t[0]))
        ax = fig.add_subplot(gs[0,i+1])
        ax.barh(range(0,len(actor_diag)),[t[1] for t in x])
        ax.set_yticks([],[])
        ax.set_xlim(0,max(values)+5)
        ax.margins(y=0)
        ax.set_title('Level ' + str(stage_level),fontsize='xx-large')
        if i == 2:
            plt.xlabel('Number of agreements signed')
    st.pyplot(fig)

else:
    st.write('Please select a peace process in the Select Peace Process page.')
