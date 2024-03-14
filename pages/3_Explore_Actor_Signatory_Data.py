import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]


# *********************************************************************************************************************

st.header("Explore Actor Signatory Data")

# *********************************************************************************************************************

if len(st.session_state["pp_data_dict"]) > 0:
    st.subheader(':blue[' + st.session_state["pp_data_dict"]['pp_name'] + ']')

# *********************************************************************************************************************

    st.subheader('Explore actor co-occurrence data')
    with st.form("actors"):
    
        # Get actors in alpha order
        actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
        actor_options = sorted(actor_options,key=lambda t:t[1])
        actor_options = [t[1] + ': ' + t[0] for t in actor_options]

        # Get currently selected actor if any - doesn't work
        if len(st.session_state["selected_data_actor"]) > 0:
            index = actor_options.index(st.session_state["selected_data_actor"])
        else:
            index = 0
        index = 0

        actor_option = st.selectbox(
        'Select an actor:',
        actor_options,index=index)

        submitted = st.form_submit_button("Submit")
        if submitted:
            actor = actor_option.split(': ')[1]
            actor_label = get_actor_name(actor,data_dict) + ' ' + actor

            st.write(':blue[Agreements signed by]',actor_label)
            agreements = get_agreements(actor,pp_data_dict)
            tuples = [(agreement,get_agreement_name(agreement,data_dict),get_agreement_date(agreement,data_dict)) for agreement in agreements]
            tuples = sorted(tuples,key=lambda t:t[2])
            for t in tuples:
                s = t[1] + ' ' + t[0] + ' [' + t[2] + ']'
                st.caption(str(s))
            st.write()

            st.divider()

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
            sorted_agreements = [(agreement_id,get_agreement_name(agreement_id,data_dict),get_agreement_date(agreement_id,data_dict),v) for agreement_id,v in agreement_cosign_dict.items()]
            sorted_agreements = sorted(sorted_agreements,key=lambda t:t[2])
            for t in sorted_agreements:
                if len(t[3]) <= 1:
                    continue
                actors = sorted(t[3],key=lambda a:a[1])
                s = t[0] + ' ' + t[1] + ' [' + t[2] + ']'
                st.write(str(s))
                for a in actors:
                    if a[0] == actor:
                        continue
                    s = '\t' + a[1] + ' ' + a[0]
                    st.caption(str(s))
            st.write()
            st.session_state["selected_data_actor"] = actor_option

 # *********************************************************************************************************************
   
    
    st.divider()
    st.subheader('Actor co-occurrence matrix')
    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    actor_upper = np.triu(co_matrices[0],k=1)

    st.write('The actor co-occurrence matrix provides the number of agreements to which a pair of actors are co-signatories.')
    st.write('The actor co-occurrence matrix for the selected peace process is visualised below as a heat map — the deeper the blue of a cell the greater the number of agreements in the cell.')
    st.write('Various operations on co-occurrence matrices are supported. The example below gives the pair of actors with the most agreements in common.')

    # Actors with max agreements between them
    actor_max = np.amax(actor_upper)
    actor_indices = np.unravel_index(np.argmax(actor_upper,axis=None),actor_upper.shape)
    actors = [(pp_data_dict['pp_actor_ids'][index],\
            data_dict['vertices_dict'][pp_data_dict['pp_actor_ids'][index]][5]) for index in actor_indices]
    s = actors[0][1] + ' (' + actors[0][0] + ') and ' + actors[1][1] + ' (' + actors[1][0] + ')'
    st.caption(':blue[' + s + ']')
    s = 'Number of agreements in common: ' + str(actor_upper[actor_indices])
    st.caption(':blue[' + s + ']')

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



else:
    st.write('Please select a peace process in the Select Peace Process page.')
