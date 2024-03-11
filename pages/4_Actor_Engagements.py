import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

if not "keep_engagement_graphic" in st.session_state:
    st.session_state["keep_engagement_graphic"] = False          

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
st.divider()
with st.form("engagement_analysis"):

    st.subheader("View actor engagements in over time")

    submitted = st.form_submit_button("Submit")
    if submitted or st.session_state["keep_engagement_graphic"]:
        st.session_state["keep_engagement_graphic"] = True

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

