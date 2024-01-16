import streamlit as st
import networkx as nx
from networkx.readwrite import json_graph

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.spatial.distance import *

import json
import os
import csv
import sys

def depth_first_search(matrix,query_index,max_depth=1,depth=1,vertices=[],visited=[]):
    """
    Recursive function to visit all vertices that are reachable from a query vertex.
    param matrix: The adjacency matrix representation of a graph
    param query_index: The row/column index which defines the query vertex
    param max_depth: How deep to go (for a bipartite graph the maximum is 2)
    param depth: Keeps track of how deep we have gone
    param vertices: Store found vertices
    param visited: Store visited vertices so we have a terminating condition
    return list of vertices
    """
    visited.append(query_index)
    # Index row - find connected head vertices in the query index row. In other words,
    # find the vertices that the query vertex point to
    vertices.extend([i for i,v in enumerate(matrix[query_index]) if v > 0 and not i in visited])
    if depth < max_depth:
        for i in vertices:
            if i in visited:
                continue
            vertices = depth_first_search(matrix,i,max_depth=1,depth=1,vertices=vertices,visited=visited)
    return vertices

def binary_to_adjacency(pp_data_dict):
    """
    Convert a binary-valued peace process agreement-actor relations matrix to an adjacency matrix
    Rows and columns of the adjacency matrix are identical and
    are constructed from the binary-valued matrix in row-column order.
    The number of rows (and columns) in the adjacency matrix is therefore:
    binary_matix.shape[0] +  binary_matix.shape[1]
    param pp_data_dict: A dictionary containing peace process network data inclusing the BVRM
    return adjacency matrix and list of vertex labels. The latter is the concatenated lists of
    agreement and actor vertex labels
    """    
    binary_matrix = pp_data_dict['pp_matrix']
    size = binary_matrix.shape[0] + binary_matrix.shape[1]
    adjacency_matrix = np.zeros((size,size))
    
    # Get the range of the bin matrix rows to generate the upper triangle
    # of the adjacency matrix
    row_index = 0
    col_index = binary_matrix.shape[0]
    adjacency_matrix[row_index:row_index + binary_matrix.shape[0],\
           col_index:col_index + binary_matrix.shape[1]] = binary_matrix
    # Add in the lower triangle
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T    
    adj_vertices = []
    adj_vertices.extend(pp_data_dict['pp_agreement_ids'])
    adj_vertices.extend(pp_data_dict['pp_actor_ids'])

    return adjacency_matrix,adj_vertices

def get_query_matrix(query_indices,matrix,max_depth=1,operator='OR'):    
    """
    Query an adjacency matrix using depth-first search
    param query_indices: The indices of the query vertices
    param matrix: The adjacency matrix we are querying
    param max_depth: Max depth of the search. Defaults to 1. Agreement-actor graphs are bipartite so
    the maximum depth is 2.
    param operator: Boolean operator to use on found vertices. AND restricts the results to entities
    that have an edge to all query vertices.
    return: An adjacency matrix for the set of found vertices and the indices of the found vertices
    """    
    found_indices = []
    for i,query_index in enumerate(query_indices):
        vertices = depth_first_search(matrix,query_index,max_depth=max_depth,vertices=[],visited=[])
        if i == 0:
            found_indices.extend(vertices)
        else:
            if operator == 'OR':
                found_indices = list(set(found_indices).union(set(vertices)))
            else:
                found_indices = list(set(found_indices).intersection(set(vertices)))
    # Add the query vertex to the found vertices
    found_indices.extend(query_indices)    
    found_indices = sorted(found_indices)
    # Extract the sub-matrix containing only the found vertices
    query_matrix = matrix[np.ix_(found_indices,found_indices)]
    return query_matrix,found_indices

def display_networkx_graph(query_matrix,vertex_indices,adj_vertices,data_dict):
    node_labels = {i:adj_vertices[index] for i,index in enumerate(vertex_indices)}
    node_colors = [data_dict['color_map'][v.split('_')[0]] for _,v in node_labels.items()]
    graph = nx.from_numpy_array(query_matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(16,16))
    pos = nx.spring_layout(graph) 
    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,node_size=200,alpha=0.6)
    plt.grid(False)
    st.pyplot(f)
    
def get_peace_processes(data_dict):
    """
    Get list of peace process names 
    param data_dict: The application's data dictionary obtained from load_agreement_actor_data()
    return: list of process names in alpha order
    """
    processes = [row[data_dict['links_header'].index('PPName')].strip() for row in data_dict['links_data']]
    return sorted(list(set(processes)))

def get_peace_process_data(process_name,data_dict):
    
    # Peace process data are in the links table so collect all edges assigned to the process
    pp_edges = [row for row in data_dict['links_data'] if row[data_dict['links_header'].\
                                                              index('PPName')].strip()==process_name]
    
    # Now we want the indices of peace process agreements and actors so we can extract the peace process
    # sub-matrix
    pp_agreement_ids = list(set([row[data_dict['links_header'].index('from_node_id')] for row in pp_edges]))
    pp_agreement_indices = [data_dict['agreement_vertices'].index(agreement_id) for\
                            agreement_id in pp_agreement_ids]
    
    pp_actor_ids = list(set([row[data_dict['links_header'].index('to_node_id')] for row in pp_edges]))
    pp_actor_indices = [data_dict['actor_vertices'].index(actor_id) for actor_id in pp_actor_ids]

    pp_matrix = data_dict['matrix'][np.ix_(pp_agreement_indices,pp_actor_indices)]
    pp_matrix = np.array(pp_matrix)
    pp_data_dict = {}
    pp_data_dict['pp_actor_ids'] = pp_actor_ids
    pp_data_dict['pp_agreement_ids'] = pp_agreement_ids
    pp_data_dict['pp_matrix'] = pp_matrix    
    return pp_data_dict

def get_cooccurrence_matrices(matrix):
    # Actor-actor co-occurence matrix for a peace process
    V = np.matmul(matrix.T,matrix)
    # Agreement-agreement co-occurence matrix
    W = np.matmul(matrix,matrix.T)
    return (V,W)

def load_agreement_actor_data(nodes_file,links_file,agreements_dict,data_path):
    # Stash data in a dictionary
    data_dict = {}
    
    # Read the CSVs
    with open(data_path + nodes_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        # Get the header row
        nodes_header = next(reader)
        # Put the remaining rows into a list of lists
        nodes_data = [row for row in reader]

    with open(data_path + links_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        # Get the header row
        links_header = next(reader)
        # Put the remaining rows into a list of lists
        links_data = [row for row in reader]

    with open(data_path + agreements_dict) as f:
        agreements_dict = json.load(f)
    
    # Agreement are from vertices
    agreement_vertices = list(set([row[links_header.index('from_node_id')] for row in links_data]))
    # Actors are to vertices
    actor_vertices = list(set([row[links_header.index('to_node_id')] for row in links_data]))

    edge_dict = {}
    dates_dict = {}
    for row in links_data:
        if row[5] in edge_dict:
            edge_dict[row[5]].append(row[12])
        else:
            edge_dict[row[5]] = [row[12]]
        if not row[5] in dates_dict:
            a = row[1].split('/')
            dates_dict[row[5]] = int(''.join(a[::-1]))
    
    # Build a vertices dictionary with node_id as key and node row as the value
    vertices_dict = {row[nodes_header.index('node_id')]:row for row in nodes_data}

    # Collect all vertex types
    vertex_types = []
    for k,v in vertices_dict.items():
        type_ = v[nodes_header.index('type')]
        if len(type_) == 0:
            # This type is missing in node data
            type_ = 'AGT'
        vertex_types.append(type_)
    vertex_types = sorted(list(set(vertex_types)))

    # Build a colour map for types
    color_map = {type_:twenty_distinct_colors[i] for\
                 i,type_ in enumerate(vertex_types)}
    
    # Build the agreement-actor BVRM matrix - the core data structure
    matrix = []
    for agreement in agreement_vertices:
        row = [0]*len(actor_vertices)
        for i,actor in enumerate(actor_vertices):
            if actor in edge_dict[agreement]:
                row[i] = 1
        matrix.append(row)
    matrix = np.array(matrix)
    
    data_dict['agreements_dict'] = agreements_dict
    data_dict['dates_dict'] = dates_dict
    data_dict['nodes_data'] = nodes_data
    data_dict['nodes_header'] = nodes_header
    data_dict['links_data'] = links_data
    data_dict['links_header'] = links_header
    data_dict['agreement_vertices'] = agreement_vertices
    data_dict['actor_vertices'] = actor_vertices
    data_dict['vertices_dict'] = vertices_dict
    data_dict['color_map'] = color_map
    data_dict['matrix'] = matrix

    return data_dict

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
st.write('Analysis of signatory data based on binary-valued relation matrices (BVRMs). Includes:') 
st.write('1. Extraction of BVRMs containing data from individual peace processes.') 
st.write('2. Generation of adjacency matrices for querying and displaying peace process graphs.')  
st.write('3. Generation of co-occurrence matrices measuring, a)\
          the number of agreements to which a pair of actors are co-signatories, b)\
          the number of signatories a pair of agreements have in common.\
         The indices of entities in co-occurrence matrix cells can be recovered.') 
st.write('4. Unlocking metadata analysis within and across peace processes.') 

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
adj_matrix,adj_vertices = binary_to_adjacency(pp_data_dict)
display_networkx_graph(adj_matrix,range(0,len(adj_vertices)),adj_vertices,data_dict)

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

st.subheader("Actor engagements over time in selected peace process")

st.write('Actors are on y-axis ordered by first appearance in a peace process. The peace process is represented as a time-ordered set of agreements on the x-axis.\
          Actor, agreement, and date information are available but are not shown on this plot.')

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

st.subheader("Actor and agreement co-occurrences in peace process")

st.write('Peace process co-occurrence matrices are generated from a peace process BVRM by matrix multiplication operations.\
         The actor co-occurrence matrix provides the number of agreements to which a pair of actors are co-signatories.\
         The agreement co-occurrence matrices provides the number of actors that are co-signatories to a pair of agreements.')

st.write('Co-occurrence matrices are visualised below as heatmaps — the deeper the blue of a cell the greater the count of agreements or actors in the cell.')

co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
actor_upper = np.triu(co_matrices[0],k=1)
agreement_upper = np.triu(co_matrices[1],k=1)

actor_max = np.amax(actor_upper)
agreement_max = np.amax(agreement_upper)

f = plt.figure(figsize=(8,8))
plt.imshow(actor_upper,cmap=plt.cm.Blues)
plt.xticks(range(0,actor_upper.shape[0]),range(0,actor_upper.shape[0]),fontsize='x-large')    
plt.yticks(fontsize='x-large')    
plt.ylabel('Actor indices',fontsize='x-large')
plt.xlabel('Actor indices',fontsize='x-large')
plt.title('Actors co-occurrence matrix')
cbar = plt.colorbar()
cbar.set_label('Number of agreements',rotation=270,labelpad=15,fontsize='x-large')
st.pyplot(f)

f = plt.figure(figsize=(8,8))
plt.imshow(agreement_upper,cmap=plt.cm.Blues)
plt.xticks(fontsize='x-large')    
plt.yticks(fontsize='x-large')    
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

st.write('3. Obtaining the agreements from a cell in an actor co-occurrence matrix.\
          In this example, the agreements for the pair of actors with the most agreements in common (see above).')

# Get the row from the transpose of pp matrix
row1 = pp_data_dict['pp_matrix'].T[actor_indices[0]]
# Get the row from the pp BVRM
row2 = pp_data_dict['pp_matrix'].T[actor_indices[1]]
x = np.bitwise_and(row1,row2)
for index,value in enumerate(x): 
    if value == 1:
        s = pp_data_dict['pp_agreement_ids'][index] + ' ' + data_dict['vertices_dict'][pp_data_dict['pp_agreement_ids'][index]][5]
        st.caption(str(s))

# *********************************************************************************************************************
st.divider()
st.subheader("Actor signatory counts in selected peace process")

st.write('Number of agreements in peace process:',pp_data_dict['pp_matrix'].shape[0])
st.write('Number of actors in peace process:',pp_data_dict['pp_matrix'].shape[1])

# Get the actor co-occurrence matrix diagonal - it's equal to the columns marginal of the peace process matrix
actor_diag = np.diag(co_matrices[0])

# Plot
labels = [data_dict['vertices_dict'][v][5] for v in pp_data_dict['pp_actor_ids']]
z = list(zip(labels,actor_diag))
z = sorted(z,key=lambda t:t[1])

f = plt.figure(figsize=(8,32))
plt.barh(range(0,len(actor_diag)),[t[1] for t in z])
plt.yticks(range(0,len(actor_diag)),[t[0] for t in z],fontsize='large')
plt.xticks(fontsize='x-large')
plt.xlabel('Number of agreements to which actor is signatory',fontsize='x-large')
plt.margins(y=0)
st.pyplot(f)

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
