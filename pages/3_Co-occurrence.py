import streamlit as st

from streamlit_shared import *

data_dict = st.session_state["data_dict"]
pp_data_dict = st.session_state["pp_data_dict"]
adj_matrix = st.session_state["adj_matrix"]
adj_vertices = st.session_state["adj_vertices"]

# Get the co-occurrence matrices
co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
actor_upper = np.triu(co_matrices[0],k=1)
agreement_upper = np.triu(co_matrices[1],k=1)

# *********************************************************************************************************************

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


