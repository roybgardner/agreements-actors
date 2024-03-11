import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

if "keep_network_graphics" not in st.session_state:
    st.session_state["keep_network_graphics"] = True          

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
st.subheader("Peace Process Analysis")

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
    if submitted or st.session_state["keep_network_graphics"]:


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
            
# *********************************************************************************************************************
st.divider()
with st.form("engagement_analysis"):

    st.subheader("View actor engagements in over time")

    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write('Actors are on y-axis ordered by first appearance in a peace process. The peace process is represented as a time-ordered set of agreements on the x-axis.\
                Actor, agreement, and date information are available but are not shown on this plot.\
                A dot indicates that the actor is a cosignatory to an agreement.')

        pp_ag_ids = pp_data_dict['pp_agreement_ids']
        # We want to sort agreements in date order so build list of agreement index-agreement_id-date tuples
        t_list = []
        for i,agreement_id in enumerate(pp_ag_ids):
            if not agreement_id in data_dict['dates_dict']:
                continue
            ag_date = data_dict['dates_dict'][agreement_id]
            # Might use the agreement_id later but currently not used
            t_list.append((i,agreement_id,ag_date))
        # Sort the agreements by date by date    
        t_list = sorted(t_list,key=lambda t:t[2])

        # Build a time-order agreement-actor matrix
        ordered_matrix = []
        for t in t_list:
            ordered_matrix.append(pp_data_dict['pp_matrix'][t[0]])
            
        ordered_matrix = np.array(ordered_matrix)
        # Put actors in rows
        ordered_matrix = ordered_matrix.T

        # Now order actors by first appearance in process (process is defined as a sequence of agreements)
        row_indices = []
        for i,row in enumerate(ordered_matrix):
            where = np.where(row==1)
            v = 0
            if len(where[0]) > 0:
                v = where[0][0]
            row_indices.append((i,v))
        sorted_row_indices = [t[0] for t in sorted(row_indices,key=lambda t:t[1])]

        sorted_matrix = ordered_matrix[np.ix_(sorted_row_indices)]

        f = plt.figure(figsize=(16,8))
        for i,row in enumerate(sorted_matrix):
            x = [j for j,x in enumerate(row) if x > 0]
            y = [i]*len(x)
            plt.scatter(x,y,alpha=0.9,linewidth=0.5,s=10)
            plt.plot(x,y,alpha=0.9,linewidth=0.5)
        plt.xticks(fontsize='xx-large')    
        plt.yticks(fontsize='xx-large')    
        plt.ylabel('Actor index (in order of first appearance)',fontsize='xx-large')
        plt.xlabel('Agreement index in time order',fontsize='xx-large')
        st.pyplot(f)


# *********************************************************************************************************************

st.divider()
with st.form("stage_analysis"):
    st.subheader("View actor signatory counts by stage")

    submitted = st.form_submit_button("Submit")
    if submitted:
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
        fig.suptitle('Actor Signatory Counts by Agreement Stage',fontsize='xx-large')
        st.pyplot(fig)

        # *********************************************************************************************************************

st.divider()
with st.form("year_analysis"):
    st.subheader("View actor signatory counts by year")

    submitted = st.form_submit_button("Submit")
    if submitted:
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

