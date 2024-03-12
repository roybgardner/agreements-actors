import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]


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
st.header("Exploring actor data")

# *********************************************************************************************************************


st.divider()

if len(st.session_state["pp_data_dict"]) > 0:

    with st.form("actors"):
    
        # Get actors in alpha order
        actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
        actor_options = sorted(actor_options,key=lambda t:t[1])
        actor_options = [t[0] + ': ' + t[1] for t in actor_options]

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
            actor = actor_option.split(':')[0]

            st.write(':blue[Agreements signed by]',actor,get_actor_name(actor,data_dict))
            agreements = get_agreements(actor,pp_data_dict)
            tuples = [(agreement,get_agreement_name(agreement,data_dict)) for agreement in agreements]
            tuples = sorted(tuples,key=lambda t:t[1])
            for t in tuples:
                s = t[0] + ' ' + t[1]
                st.caption(str(s))
            st.write()

            st.write(':blue[Co-signatories of]',actor,'organised by agreement')
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
                s = agreement + ' ' + get_agreement_name(agreement,data_dict)
                st.write(str(s))
                for a in actors:
                    if a[0] == actor:
                        continue
                    s = '\t' + a[0] + ' ' + a[1]
                    st.caption(str(s))
            st.write()

else:
    st.write('Please select a peace process in the Select Peace Process page.')
