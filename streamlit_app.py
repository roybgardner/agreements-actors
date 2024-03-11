import streamlit as st

from streamlit_shared import *


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

# *********************************************************************************************************************


# Load data from CSV
data_path = './data/'
nodes_file = 'node_table.csv'
links_file = 'links_table.csv'
agreements_dict = 'agreements_dict.json'

data_dict = load_agreement_actor_data(nodes_file,links_file,agreements_dict,data_path)

if "data_dict" not in st.session_state:
   st.session_state["data_dict"] = data_dict

st.subheader('Credits') 
st.write('Signatory data: Niamh Henry and Sanja Badanjak') 
st.write('Analysis/coding: Roy Gardner') 

st.subheader('Approach') 
st.write('Agreement-actor signatory data are an example of an undirected bipartite graph, i.e.,\
          there are edges (links) between agreements and actors, but not between agreements or between actors.') 
st.write('Agreement-actor bipartite graphs are represented by binary-valued biadjacency matrices (BMs).\
          The rows of the matrix correspond to agreements and the columns to actors. Cell values contain\
          the value 1 if an actor is a signatory to an agreement, otherwise cells values are 0.') 
st.write('Agreement-actor biadjacency matrices provide the basis of peace process network analysis as follows:') 
st.write('1. Extraction of BMs containing data from individual peace processes.') 
st.write('2. Generation of adjacency matrices used for depth-first search network queries,\
          and for passing to network packages for network visualisation.') 
st.write('3. Generation of co-occurrence matrices measuring, a)\
          the number of agreements to which a pair of actors are co-signatories, b)\
          the number of signatories a pair of agreements have in common.\
         The indices of entities in co-occurrence matrix cells can be recovered.') 
st.write('4. Support for metadata-based analysis within and across peace processes.') 

# *********************************************************************************************************************

# *********************************************************************************************************************
st.divider()

#Query vertices using depth-first search
with st.form("query"):
    st.subheader("Query peace process network")
    st.write('Interface for formulating queries and providing users with insight into peace process actors and agreements.')
    st.write('Select actors (in alpha order) with agreements (in date order) using the selectors below.')
    st.write('Mixing and matching actor and agreements is supported but might not be sensible.')
 
    # Get actors in alpha order
    actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
    actor_options = sorted(actor_options,key=lambda t:t[1])
    actor_options = [t[0] + ': ' + t[1] for t in actor_options]

    # Get agreements in date order
    agreement_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5],data_dict['dates_dict'][vertex_id]) for vertex_id in pp_data_dict['pp_agreement_ids']]
    agreement_options = sorted(agreement_options,key=lambda t:t[2])
    agreement_options = [t[0] + ': ' + t[1] for t in agreement_options]

    options_actor = st.multiselect(
    'Select zero or more actors.',
    actor_options,
    [])

    options_agreement = st.multiselect(
    'Select zero or more agreements',
    agreement_options,
    [])

    operator=["AND", "OR"]
    select_operator=st.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, horizontal=False, captions=None, label_visibility="visible")
    #depth=st.slider("Select depth", min_value=1, max_value=2, value=1, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        options = [v.split(':')[0] for v in options_actor]
        options.extend([v.split(':')[0] for v in options_agreement])
        query_indices = [adj_vertices.index(vertex) for vertex in options]
        query_matrix,found_indices = get_query_matrix(query_indices,adj_matrix,max_depth=1,operator=select_operator)
        display_networkx_graph(query_matrix,found_indices,adj_vertices,data_dict)


# *********************************************************************************************************************
st.divider()

st.subheader("Actor and agreement co-occurrences in peace process")

st.write('Peace process co-occurrence matrices are generated from a peace process BM by matrix multiplication operations.\
         The actor co-occurrence matrix provides the number of agreements to which a pair of actors are co-signatories.\
         The agreement co-occurrence matrices provides the number of actors that are co-signatories to a pair of agreements.')

st.write('Co-occurrence matrices are visualised below as heatmaps — the deeper the blue of a cell the greater the count of agreements or actors in the cell.')


actor_max = np.amax(actor_upper)
agreement_max = np.amax(agreement_upper)

f = plt.figure(figsize=(8,8))
plt.imshow(actor_upper,cmap=plt.cm.Blues)
ticks = range(0,actor_upper.shape[0])
plt.xticks(ticks,[])    
plt.yticks(ticks,[])    
plt.ylabel('Actor indices',fontsize='x-large')
plt.xlabel('Actor indices',fontsize='x-large')
plt.title('Actors co-occurrence matrix')
cbar = plt.colorbar()
cbar.set_label('Number of agreements',rotation=270,labelpad=15,fontsize='x-large')
st.pyplot(f)

f = plt.figure(figsize=(8,8))
plt.imshow(agreement_upper,cmap=plt.cm.Blues)
ticks = range(0,agreement_upper.shape[0])
plt.xticks(ticks,[])    
plt.yticks(ticks,[])    
plt.ylabel('Agreement indices',fontsize='x-large')
plt.xlabel('Agreement indices',fontsize='x-large')
plt.title('Agreements co-occurrence matrix')
cbar = plt.colorbar()
cbar.set_label('Number of actors',rotation=270,labelpad=15,fontsize='x-large')
st.pyplot(f)

st.write('Various operations on co-occurrence matrices are supported. Some examples are given below.')
st.write('1. Number of agreements to which a pair of actors are co-signatories.\
          The example below gives the pair of actors with the most agreements in common.')

# Actors with max agreements between them
actor_indices = np.unravel_index(np.argmax(actor_upper,axis=None),actor_upper.shape)
actors = [(pp_data_dict['pp_actor_ids'][index],\
           data_dict['vertices_dict'][pp_data_dict['pp_actor_ids'][index]][5]) for index in actor_indices]
s = actors[0][1] + ' (' + actors[0][0] + ') and ' + actors[1][1] + ' (' + actors[1][0] + ')'
st.caption(str(s))
s = 'Number of co-agreements: ' + str(actor_upper[actor_indices])
st.caption(str(s))

st.write('2. Number of actors who are co-signatories to a pair of agreements.\
          The example below gives the pair of agreements with the most co-signatories.')

# Agreements with max actors between them
agreement_indices = np.unravel_index(np.argmax(agreement_upper,axis=None),agreement_upper.shape)
agreements = [(pp_data_dict['pp_agreement_ids'][index],\
               data_dict['vertices_dict'][pp_data_dict['pp_agreement_ids'][index]][5]) for index in agreement_indices]
s = agreements[0][1] + ' (' + agreements[0][0] + ') and ' + agreements[1][1] + ' (' + agreements[1][0] + ')'
st.caption(str(s))
s = 'Number of co-signatories: ' + str(agreement_upper[agreement_indices])
st.caption(str(s))


# *********************************************************************************************************************
st.divider()

st.subheader("Exploring actor data")
st.write("Matrix-based access to actor data.")

with st.form("actors"):
 
    # Get actors in alpha order
    actor_options = [(vertex_id,data_dict['vertices_dict'][vertex_id][5]) for vertex_id in pp_data_dict['pp_actor_ids']]
    actor_options = sorted(actor_options,key=lambda t:t[1])
    actor_options = [t[0] + ': ' + t[1] for t in actor_options]

    actor_option = st.selectbox(
    'Select an actor:',
    actor_options)

    submitted = st.form_submit_button("Submit")
    if submitted:
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

