import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]


# *********************************************************************************************************************

st.header("Explore Actor Data")
st.subheader(st.session_state["pp_data_dict"]['pp_name'])


# *********************************************************************************************************************

if len(st.session_state["pp_data_dict"]) > 0:

 # *********************************************************************************************************************
   
    st.divider()
    st.subheader('Actor engagements over time')

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
    st.subheader('Actor co-occurrence matrix')
    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    actor_upper = np.triu(co_matrices[0],k=1)

    st.write('The actor co-occurrence matrix provides the number of agreements to which a pair of actors are co-signatories.')
    st.write('Co-occurrence matrices are visualised below as heatmaps — the deeper the blue of a cell the greater the number of agreements in the cell.')

    st.write('Various operations on co-occurrence matrices are supported. The example below gives the pair of actors with the most agreements in common.')

    # Actors with max agreements between them
    actor_max = np.amax(actor_upper)
    actor_indices = np.unravel_index(np.argmax(actor_upper,axis=None),actor_upper.shape)
    actors = [(pp_data_dict['pp_actor_ids'][index],\
            data_dict['vertices_dict'][pp_data_dict['pp_actor_ids'][index]][5]) for index in actor_indices]
    s = actors[0][1] + ' (' + actors[0][0] + ') and ' + actors[1][1] + ' (' + actors[1][0] + ')'
    st.caption(str(s))
    s = 'Number of agreements in common: ' + str(actor_upper[actor_indices])
    st.caption(str(s))

    f = plt.figure(figsize=(8,8))
    plt.imshow(actor_upper,cmap=plt.cm.Blues)
    ticks = range(0,actor_upper.shape[0])
    plt.xticks([],[])    
    plt.yticks([],[])    
    plt.ylabel('Actors',fontsize='x-large')
    plt.xlabel('Actors',fontsize='x-large')
    plt.title('Actors co-occurrence matrix')
    cbar = plt.colorbar()
    cbar.set_label('Number of agreements',rotation=270,labelpad=15,fontsize='x-large')
    st.pyplot(f)


# *********************************************************************************************************************

    st.divider()
    st.subheader('Explore individual actors using actor co-occurrence data')
    with st.form("actors"):
    
        # Get actors in alpha order
        actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
        actor_options = sorted(actor_options,key=lambda t:t[1])
        actor_options = [t[1] + ': ' + t[0] for t in actor_options]

        # Get currently selected actor if any
        if len(st.session_state["selected_data_actor"]) > 0:
            index = actor_options.index(st.session_state["selected_data_actor"])
        else:
            index = 0

        actor_option = st.selectbox(
        'Select an actor:',
        actor_options,index=index)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state["selected_data_actor"] = actor_option
            actor = actor_option.split(': ')[1]
            actor_label = get_actor_name(actor,data_dict) + ' ' + actor

            st.write(':blue[Agreements signed by]',actor_label)
            agreements = get_agreements(actor,pp_data_dict)
            tuples = [(agreement,get_agreement_name(agreement,data_dict),get_agreement_date(agreement,data_dict)) for agreement in agreements]
            tuples = sorted(tuples,key=lambda t:t[1])
            for t in tuples:
                s = t[1] + ' ' + t[0] + ' [' + t[2] + ']'
                st.caption(str(s))
            st.write()

            st.write(':blue[Co-signatories of]',actor_label,'organised by agreement')
            cosigns = get_consignatories(actor,pp_data_dict)
            agreement_cosign_dict = {}
            for cosign in cosigns:
                agreements = get_consignatory_agreements([actor,cosign],pp_data_dict)
                for agreement in agreements:
                    if agreement in agreement_cosign_dict:
                        agreement_cosign_dict[agreement].append((cosign,get_actor_name(cosign,data_dict)))
                    else:
                        agreement_cosign_dict[agreement] = [(cosign,get_actor_name(cosign,data_dict))]
            for agreement,actors in agreement_cosign_dict.items():
                actors = sorted(actors,key=lambda t:t[1])
                s = agreement + ' ' + get_agreement_name(agreement,data_dict) + ' [' + get_agreement_date(agreement,data_dict) + ']'
                st.write(str(s))
                for a in actors:
                    if a[0] == actor:
                        continue
                    s = '\t' + a[1] + ' ' + a[0]
                    st.caption(str(s))
            st.write()

else:
    st.write('Please select a peace process in the Select Peace Process page.')
