import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]


# *********************************************************************************************************************

st.header("Explore Agreement Data")

# *********************************************************************************************************************

if len(st.session_state["pp_data_dict"]) > 0:
    st.subheader(':blue[' + st.session_state["pp_data_dict"]['pp_name'] + ']')

# *********************************************************************************************************************

    st.subheader('Explore agreement co-occurrence data')

    st.write("Actor co-occurrence is when two agreements are both signed the same actors.")
    st.write("Using the drop-down menu, select an agreement from the chosen peace process to discover which other agreements were signed by the signatories of the chosen agreement.\
             Clicking on the Submit button will:")
    st.text("1. Display the agreements to which the actor is a co-signatory.\n\
2. Display the names of the other co-signatories on each agreement")
    with st.form("agreements"):
    
        # Get agreements in date order
        agreement_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5],data_dict['dates_dict'][vertex_id]) for vertex_id in pp_data_dict['pp_agreement_ids']]
        agreement_options = sorted(agreement_options,key=lambda t:t[2])
        agreement_options = [t[0] + ': ' + t[1] for t in agreement_options]

        # Get currently selected agreement if any
        if len(st.session_state["selected_data_agreement"]) > 0:
            index = agreement_options.index(st.session_state["selected_data_agreement"])
        else:
            index = 0
        index = 0

        agreement_option = st.selectbox(
        'Select an agreement:',
        agreement_options,index=index)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state["selected_data_agreement"] = agreement_option
            agreement = agreement_option.split(':')[0]

            st.write(':blue[Signatories of]',agreement,get_agreement_name(agreement,data_dict))
            actors = get_actors(agreement,pp_data_dict)
            tuples = [(actor,get_actor_name(actor,data_dict)) for actor in actors]
            tuples = sorted(tuples,key=lambda t:t[1])
            for t in tuples:
                s = t[0] + ' ' + t[1]
                st.caption(str(s))
            st.write()

            st.divider()

            st.write(':blue[Agreements also signed by the signatories of]',agreement)
            coagrees = get_coagreements(agreement,pp_data_dict)
            cosign_agreement_dict = {}
            for coagree in coagrees:
                cosignatories = get_agreement_cosignatories([agreement,coagree],pp_data_dict)
                for cosign in cosignatories:
                    if cosign in cosign_agreement_dict:
                        cosign_agreement_dict[cosign].append(coagree)
                    else:
                        cosign_agreement_dict[cosign] = [coagree]

            agreement_ids = []
            for _,v in cosign_agreement_dict.items():
                agreement_ids.extend(v)
            agreement_ids = list(set(agreement_ids))
            agreements = []
            for agreement_id in agreement_ids:
                agreements.append((agreement_id,get_agreement_name(agreement_id,data_dict),get_agreement_date(agreement_id,data_dict)))
                
            agreements = sorted(agreements,key=lambda t:t[2])
            for t in agreements:
                s = t[0] + ' ' + t[1] + ' [' + t[2] + ']'
                st.caption(str(s))
            st.write()

 # *********************************************************************************************************************
   
    st.divider()
    st.subheader('Agreement co-occurrence matrix')
    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    agreement_upper = np.triu(co_matrices[1],k=1)

    st.write('The graph shows an example of a co-occurrence matrix represented as a heat map. The example here shows the co-occurrence of agreements, i.e.,\
              when two agreements are both signed by the same actors.\
              The deeper the blue, the greater the number of actors that a particular pair of agreements have in common.\
              The lightest blue indicates that a particular pair of agreements had no actors in common.')
    st.write('Various operations on co-occurrence matrices are supported including the ability to recover the identities of the actors in a matrix cell.\
             The example below gives the pair of agreements with the most signatories in common.')

    # Agreements with max actors between them
    agreement_indices = np.unravel_index(np.argmax(agreement_upper,axis=None),agreement_upper.shape)
    agreements = [(pp_data_dict['pp_agreement_ids'][index],\
                    data_dict['vertices_dict'][pp_data_dict['pp_agreement_ids'][index]][5]) for index in agreement_indices]
    s = agreements[0][1] + ' (' + agreements[0][0] + ') and ' + agreements[1][1] + ' (' + agreements[1][0] + ')'
    st.caption(':blue[' + s + ']')
    s = 'Number of co-signatories: ' + str(agreement_upper[agreement_indices])
    st.caption(':blue[' + s + ']')

    f = plt.figure(figsize=(8,8))
    plt.imshow(agreement_upper,cmap=plt.cm.Blues)
    ticks = range(0,agreement_upper.shape[0])
    plt.xticks([],[])    
    plt.yticks([],[])    
    plt.ylabel('Agreements',fontsize='x-large')
    plt.xlabel('Agreements',fontsize='x-large')
    plt.title('Agreements co-occurrence matrix')
    cbar = plt.colorbar()
    yint = range(0, math.ceil(np.amax(agreement_upper))+1)
    cbar.set_ticks(yint)
    cbar.set_label('Number of actors',rotation=270,labelpad=15,fontsize='x-large')
    st.pyplot(f)



else:
    st.write('Please select a peace process in the Select Peace Process page.')
