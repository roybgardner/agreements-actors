import streamlit as st

from network_functions import *
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


twenty_distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',\
                          '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',\
                          '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff',\
                          '#000000']

st.markdown('<p class="maintitle">Signatories Network Analysis</p>', unsafe_allow_html=True)

# *********************************************************************************************************************

# Load data from CSV
data_path = './data/'
nodes_file = 'node_table.csv'
links_file = 'links_table.csv'
agreements_dict = 'agreements_dict.json'

data_dict = load_agreement_actor_data(nodes_file,links_file,agreements_dict,data_path)

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
st.divider()

st.header("Peace Process Analysis")

# Select a peace process
st.subheader("Select a peace process")
pp_names = get_peace_processes(data_dict)
pp_selection=st.selectbox("", pp_names, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose a Peace Process", disabled=False, label_visibility="visible")
pp_data_dict = get_peace_process_data(pp_selection,data_dict)

if pp_data_dict['pp_matrix'].shape[0] == 0 or pp_data_dict['pp_matrix'].shape[1] == 0:
    st.write('ISSUE: peace process submatrix is empty')
    raise Exception('error')

st.write('Number of agreements in peace process:',pp_data_dict['pp_matrix'].shape[0])
st.write('Number of actors in peace process:',pp_data_dict['pp_matrix'].shape[1])

st.subheader("Peace Process Agreement-Actor Network")

# Build peace process adjacency matrix and get adjacency matrix vertices and display graph using networkX
adj_matrix,adj_vertices = adjacency_from_biadjacency(pp_data_dict)
display_networkx_graph(adj_matrix,range(0,len(adj_vertices)),adj_vertices,data_dict)

# Get the co-occurrence matrices
co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
actor_upper = np.triu(co_matrices[0],k=1)
agreement_upper = np.triu(co_matrices[1],k=1)

# *********************************************************************************************************************
st.divider()
st.subheader("Actor signatory counts by stage")

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
st.subheader("Actor signatory counts by year")

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

st.subheader("Actor engagements in peace process over time")

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

