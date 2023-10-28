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

def load_agreement_actor_data(nodes_file,links_file,agreements_dict,data_path):
    """
    Load node and links data from CSV files, process, and add processed data to a data dictionary 
    The core data structure is a binary matrix defining the relationship between agreements and actors.
    Agreements are in rows, actors are in columns. Both agreements and actors are indexed sets.
    param nodes_file: The name of the CSV file containing node data
    param links_file: The name of the CSV file containing links data
    param agreements_dict: Dictionary from semantic work containing agreement metadata
    param data_path: The path to the folder containing the CSV files
    return: dictionary of processed data in data_dict
    """

    # Stash data in this dictionary
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

   # Agreement are the from vertices
    agreement_vertices = list(set([row[links_header.index('from_node_id')] for row in links_data]))
    
    # Actors are the to vertices
    actor_vertices = list(set([row[links_header.index('to_node_id')] for row in links_data]))

    # Build an edge dict (not persistent) with agreement as key and actor as value
    # Build an dates dict (persistent) with agreement as key and date as YYYYMMDD integer as value
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
    color_map = {type_:twenty_distinct_colors[i] for i,type_ in enumerate(vertex_types)}
    
    # This is the core data structure - the binary-valued relation matrix
    matrix = []
    for agreement in agreement_vertices:
        row = [0]*len(actor_vertices)
        for i,actor in enumerate(actor_vertices):
            if actor in edge_dict[agreement]:
                row[i] = 1
        matrix.append(row)
    matrix = np.array(matrix)

    # Agreements dictionary from semantic work
    data_dict['agreements_dict'] = agreements_dict
    # Agreement:signed date as YYYYMMDD integer
    data_dict['dates_dict'] = dates_dict
    # So we can index into node CSV rows using column names
    data_dict['nodes_header'] = nodes_header
    # Rows from the links CSV excluding the header
    data_dict['links_data'] = links_data
    # So we can index into links CSV rows using column names
    data_dict['links_header'] = links_header
    # List of agreement vertex IDs
    data_dict['agreement_vertices'] = agreement_vertices
    # List of actor vertex IDs
    data_dict['actor_vertices'] = actor_vertices
    # Vertex ID:vertex row - convenient way to lookup vertex data
    data_dict['vertices_dict'] = vertices_dict
    # Maps vertex types onto distinct colors
    data_dict['color_map'] = color_map
    # Binary-valued relation matrix with agreements in rows and actors in columns. This is where the work is done.
    data_dict['matrix'] = matrix

    return data_dict

def get_peace_processes(data_dict):
    """
    Get list of peace process names 
    param data_dict: The application's data dictionary obtained from load_agreement_actor_data()
    return: list of process names in alpha order
    """
    processes = [row[data_dict['links_header'].index('PPName')].strip() for row in data_dict['links_data']]
    return sorted(list(set(processes)))

def get_peace_process_data(process_name,data_dict):
    """
    Peace process data including the process agreement-actor relation matrix 
    param process_name: Name of peace process
    param data_dict: The application's data dictionary obtained from load_agreement_actor_data()
    return: peace process data dictionary
    """

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

    # Lift the process sub-matrix from the complete agreement-actor relation matrix
    pp_matrix = data_dict['matrix'][np.ix_(pp_agreement_indices,pp_actor_indices)]

    # Start populating the process data dictionary
    pp_data_dict = {}
    pp_data_dict['pp_actor_ids'] = pp_actor_ids
    pp_data_dict['pp_agreement_ids'] = pp_agreement_ids
    pp_data_dict['pp_matrix'] = pp_matrix
    
    # Build a graph from the process relation matrix
    linked_pairs = []
    for i,row in enumerate(pp_matrix):    
        linked_pairs.extend([(pp_agreement_ids[i],v,pp_actor_ids[j]) for j,v in enumerate(row) if v > 0])
    pp_graph = nx.Graph()

    vertices = []
    # Some simple integrity checks here
    vertices.extend([t[0] for t in linked_pairs if len(t[0])>0 and '_' in t[0]])
    vertices.extend([t[2] for t in linked_pairs if len(t[2])>0 and '_' in t[2]])
    vertices = list(set(vertices))

    pp_graph.add_nodes_from(vertices)
    for pair in linked_pairs:
        if len(pair[0])>0 and '_' in pair[0] and len(pair[2])>0 and '_' in pair[2]:
            pp_graph.add_edge(pair[0],pair[2],weight=pair[1])
    
    # Add the graph to the process data
    pp_data_dict['pp_graph'] = {}
    pp_data_dict['pp_graph'] = pp_graph
    pp_data_dict['pp_node_colors'] = [data_dict['color_map'][v.split('_')[0]] for v in vertices]

    return pp_data_dict

def query_graph(graph,query_vertices=[],operator='AND',depth=1):
    """
    Query a peace process graph using networkx depth-first search 
    param graph: A peace process graph
    param query_vertices: A list of actor and/or agreement vertex IDs from which we search
    param operator: Logical operator between found vertices of the query vertices
    param depth: depth of search. Default = 1
    return: dictionary containing found sub-graph and node colors
    """

    tree_list = []
    for v in query_vertices:
         tree_list.append(nx.dfs_tree(graph,source=v,depth_limit=depth))

    found_vertices = set(tree_list[0].nodes) 
    for tree in tree_list:
        if operator == 'AND':
            # Only include found vertices that are common to all query vertices
            found_vertices = found_vertices.intersection(set(tree.nodes))
        else:
            found_vertices = found_vertices.union(set(tree.nodes))

    found_vertices = list(found_vertices) 
    # Don't forget the query vertices  
    found_vertices.extend(query_vertices)

    # Edges between the found vertices
    found_edges = []
    for tree in tree_list:
        for e in tree.edges:
            if e[0] in found_vertices and e[1] in found_vertices:
                found_edges.append(e)

    found_graph = nx.Graph()
    found_graph.add_nodes_from(found_vertices)
    for e in found_edges:
        found_graph.add_edge(e[0],e[1],weight=1)
    node_colors = [data_dict['color_map'][v.split('_')[0]] for v in found_graph.nodes()]
    # Build a results dictionary
    results_dict = {}
    results_dict['graph'] = found_graph
    results_dict['node_colors'] = node_colors
    return results_dict

def display_graph(graph,node_colors):
    """
    Display a graph 
    param graph: A graph
    param node_colors: Colors for the graph nodes
    """
    f = plt.figure(figsize=(16,16))
    pos = nx.spring_layout(graph) 
    nx.draw_networkx(graph,pos,node_color=node_colors,font_size='8',alpha=0.8)
    plt.grid(False)
    st.pyplot(f)

def get_cooccurrence_matrices(matrix):
    """
    Get co-occurence matrices for a relation matrix 
    param matrix: A is an agreement-actor or process-actor relation matrix
    return: tuple of product matrices (A.T*A, A*A.T)
    """
    # Actor-actor co-occurence matrix for a peace process
    V = np.matmul(matrix.T,matrix)
    # Agreement-agreement co-occurence matrix
    W = np.matmul(matrix,matrix.T)
    return (V,W)

def display_cooccurrence_network(key,co_matrices,pp_data_dict,data_dict,threshold):
    """
    TODO - GENERALISE TO PROCESS-ACTOR MATRICES
    Display the uppoer triangle of a co-occurence matrix as a network 
    param key: Key for accessing actor or agreement vertex data
    param co_matrices: Key for accessing actor or agreement vertex data
    return: tuple of product matrices (A.T*A, A*A.T)
    """
    if key == 'actor':
        upper = np.triu(co_matrices[0],k=1)
        ids_key = 'pp_actor_ids'
    else:
        upper = np.triu(co_matrices[1],k=1)
        ids_key = 'pp_agreement_ids'

    # Upper triangle without diagonal
    linked_pairs = []
    for i,row in enumerate(upper): 
        linked_pairs.extend([(pp_data_dict[ids_key][i],v,pp_data_dict[ids_key][j]) for j,v in enumerate(row) if v >= threshold])
    actor_graph = nx.Graph()

    vertices = []
    vertices.extend([t[0] for t in linked_pairs])
    vertices.extend([t[2] for t in linked_pairs])
    vertices = list(set(vertices))
    actor_graph.add_nodes_from(vertices)
    for pair in linked_pairs:
        actor_graph.add_edge(pair[0],pair[2],weight=pair[1])

    vertex_labels = {v:v+'\n'+data_dict['vertices_dict'][v][5] for i,v in enumerate(pp_data_dict[ids_key]) if v in vertices}

    vertex_colors = [data_dict['color_map'][v.split('_')[0]] for v in actor_graph.nodes]

    f = plt.figure(figsize=(16,16))
    pos = nx.spring_layout(actor_graph) 

    nx.draw_networkx_nodes(actor_graph,pos,
                    nodelist=vertices,
                    node_size=1500,
                    node_color=vertex_colors,
                    alpha=0.8)
    nx.draw_networkx_edges(actor_graph,pos,
                        edgelist = [(t[0],t[2]) for t in linked_pairs],
                        width=[t[1] for t in linked_pairs],
                        edge_color='lightblue',
                        alpha=0.8)
    nx.draw_networkx_labels(actor_graph, pos,
                            labels=vertex_labels,
                            horizontalalignment='left',
                            font_color='black')
    nx.draw_networkx_edge_labels(actor_graph, pos,
                            edge_labels={(t[0],t[2]):t[1] for t in linked_pairs},
                            font_color='black')

    plt.grid(False)
    st.pyplot(f)

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
st.write('Analysis of signatory data based on binary-valued relation matrices. Includes:') 
st.write('1. Simple extraction of peace process data for network generation.') 
st.write('2. Querying of peace process networks.') 
st.write('3. Generation of co-occurrence networks measuring, for example, a) the number of agreements to which a pair of actors are co-signatories, b) the number of signatories a pair of agreements have in common.') 
st.write('4. Unlocking metadata analysis within and across peace processes.') 

st.subheader('Co-occurrence Matrices') 

st.write('The indices of the entities in a cell c(ij) of a co-occurrence matrix (C=A.TA or C=AA.T) can be retrieved by finding\
        indices containing non-zero values in the result of a bitwise AND operation between the ith row of the first matrix (A.T or A)\
        and the jth column of the second matrix (A or A.T).')


st.header("Peace Process Network Analysis")

# Select a peace process
st.subheader("Select a peace process")
pp_names = get_peace_processes(data_dict)
pp_selection=st.selectbox("", pp_names, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose a Peace Process", disabled=False, label_visibility="visible")
pp_data_dict = get_peace_process_data(pp_selection,data_dict)
st.write('Number of agreements in peace process:',pp_data_dict['pp_matrix'].shape[0])
st.write('Number of actors in peace process:',pp_data_dict['pp_matrix'].shape[1])

# Display peace process graph
pp_graph = pp_data_dict['pp_graph']
node_colors = pp_data_dict['pp_node_colors']
display_graph(pp_graph,node_colors)

# *********************************************************************************************************************

#Query vertices using depth-first search
with st.form("query"):
    st.subheader("Query peace process network")
    st.write('Critical UX/UI for formulating queries and providing users with insight into process actors and agreements.')
    st.write('Mix and match actors (in alpha order) with agreements (in date order) using the selectors below.')
 
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
    depth=st.slider("Select depth", min_value=1, max_value=3, value=1, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        options = [v.split(':')[0] for v in options_actor]
        options.extend([v.split(':')[0] for v in options_agreement])
        results_dict = query_graph(pp_graph,query_vertices=options,operator=select_operator,depth=depth)
        display_graph(results_dict['graph'],results_dict['node_colors'])

# *********************************************************************************************************************

# Co-occurrence networks
with st.form("cooccurrence"):
    st.subheader("Actor and agreement co-occurrences in peace process")
    st.write("NOTE: Co-occurrence networks can be queried to ask questions like 'What is the number of agreements to which countries X and Y were both co-signatories?'")
    st.write("It would also be possible to find actors who have never been co-signatories.")
    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    actor_upper = np.triu(co_matrices[0],k=1)
    agreement_upper = np.triu(co_matrices[1],k=1)

    # Get the states of the sliders right
    actor_disabled = False
    agreement_disabled = False
    actor_min = 1
    agreement_min = 1
    actor_default = 1
    agreement_default = 1

    actor_max = np.amax(actor_upper)
    agreement_max = np.amax(agreement_upper)
    if actor_max == 1:
        actor_min = 0
        actor_disabled = True
        actor_threshold = 1
    else:
        actor_default = math.ceil(actor_max/2)
    if agreement_max == 1:
        agreement_min = 0
        agreement_disabled = True
        agreement_threshold = 1
    else:
        agreement_default = math.ceil(agreement_max/2)

    actor_threshold=st.slider("Actor co-occurrence threshold", min_value=actor_min, max_value=actor_max, value=actor_default, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=actor_disabled, label_visibility="visible")
    agreement_threshold=st.slider("Agreement co-occurrence threshold", min_value=agreement_min, max_value=agreement_max, value=agreement_default, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=agreement_disabled, label_visibility="visible")

    # Every form must have a submit button.
    submitted_cooccur = st.form_submit_button("Submit")
    if submitted_cooccur:
        st.write('Edge values are the number of agreements to which a pair of actors are co-signatories.')
        display_cooccurrence_network('actor',co_matrices,pp_data_dict,data_dict,actor_threshold)
        st.write('Edge values are the number of signatories a pair of agreements have in common.')
        display_cooccurrence_network('agreement',co_matrices,pp_data_dict,data_dict,agreement_threshold)

# *********************************************************************************************************************

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


# *********************************************************************************************************************

st.subheader("Actor engagements over time in selected peace process")

st.write('Actors on y-axis ordered by first appearance in a peace process. The peace process is represented as a time-ordered set of agreements on the x-axis.\
          Necessary actor, agreement, and date information are available.')

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
# *********************************************************************************************************************

st.header("Analysis - All Agreements")

st.write('Number of agreements in full data set:',data_dict['matrix'].shape[0])
st.write('Number of actors in full data set:',data_dict['matrix'].shape[1])

st.write("NOTE: Co-occurrence networks can be queried to ask questions like 'What is the number of peace processes in which countries X and Y were both signatory actors?'")
st.subheader("Peace process - actor relation matrix")

# THIS IS HORRIBLY INLINE - NEEDS TO BE TIDIED AND REALITY CHECKED

st.write('Can use the process-actor relation matrix (visualised below) to examine process and actor co-occurence networks.')
st.write('Scope for analysis by process attributes (e.g., number of agreements, duration, messiness) and metadata, and actor attributes.')
st.write('')

# Build a process-actor matrix
process_matrix = np.zeros((len(pp_names),len(data_dict['actor_vertices'])), dtype=int)

for i,process_name in enumerate(pp_names):
    process_data = get_peace_process_data(process_name,data_dict)
    actor_marginal = [sum(row) for row in process_data['pp_matrix'].T]
    actor_indices = [data_dict['actor_vertices'].index(actor_id) for actor_id in process_data['pp_actor_ids']]
    for j,v in enumerate(actor_marginal):
        if v > 0:
            process_matrix[i][actor_indices[j]] = 1

f = plt.figure(figsize=(8,8))
plt.imshow(process_matrix,cmap=plt.cm.Blues)
plt.ylabel('Peace process')
plt.xlabel('Actor')
st.pyplot(f)


proc_co_matrices = get_cooccurrence_matrices(process_matrix)
upper = np.triu(proc_co_matrices[0],k=1)
ids_key = 'actor_vertices'
threshold = math.ceil(np.amax(upper)/2)

# *********************************************************************************************************************

st.write('Network diagram showing pairs of actors with', threshold, 'or more peace processes in common.')

proc_linked_pairs = []
for i,row in enumerate(upper): 
    proc_linked_pairs.extend([(data_dict[ids_key][i],v,data_dict[ids_key][j]) for j,v in enumerate(row) if v >= threshold])

proc_graph = nx.Graph()

proc_vertices = []
proc_vertices.extend([t[0] for t in proc_linked_pairs])
proc_vertices.extend([t[2] for t in proc_linked_pairs])
proc_vertices = list(set(proc_vertices))
proc_graph.add_nodes_from(proc_vertices)
for pair in proc_linked_pairs:
    proc_graph.add_edge(pair[0],pair[2],weight=pair[1])

proc_vertex_labels = {v:v+'\n'+data_dict['vertices_dict'][v][5] for i,v in enumerate(data_dict[ids_key]) if v in proc_vertices}

proc_vertex_colors = [data_dict['color_map'][v.split('_')[0]] for v in proc_graph.nodes]

f = plt.figure(figsize=(16,16))
pos = nx.spring_layout(proc_graph) 

nx.draw_networkx_nodes(proc_graph,pos,
                nodelist=proc_vertices,
                node_size=1500,
                node_color=proc_vertex_colors,
                alpha=0.8)
nx.draw_networkx_edges(proc_graph,pos,
                    edgelist = [(t[0],t[2]) for t in proc_linked_pairs],
                    width=[t[1] for t in proc_linked_pairs],
                    edge_color='lightblue',
                    alpha=0.8)
nx.draw_networkx_labels(proc_graph, pos,
                        labels=proc_vertex_labels,
                        horizontalalignment='left',
                        font_color='black')
nx.draw_networkx_edge_labels(proc_graph, pos,
                        edge_labels={(t[0],t[2]):t[1] for t in proc_linked_pairs},
                        font_color='black')

plt.grid(False)
st.pyplot(f)


upper = np.triu(proc_co_matrices[1],k=1)
threshold = math.ceil(np.amax(upper)/2)

# *********************************************************************************************************************

st.write('Network diagram showing pairs of peace processes with', threshold ,'or more actors in common.')

proc_linked_pairs = []
for i,row in enumerate(upper): 
    proc_linked_pairs.extend([(pp_names[i],v,pp_names[j]) for j,v in enumerate(row) if v >= threshold])

proc_graph = nx.Graph()

proc_vertices = []
proc_vertices.extend([t[0] for t in proc_linked_pairs])
proc_vertices.extend([t[2] for t in proc_linked_pairs])
proc_vertices = list(set(proc_vertices))
proc_graph.add_nodes_from(proc_vertices)
for pair in proc_linked_pairs:
    proc_graph.add_edge(pair[0],pair[2],weight=pair[1])

proc_vertex_labels = {v:v for v in pp_names if v in proc_vertices}


f = plt.figure(figsize=(16,16))
pos = nx.spring_layout(proc_graph) 

nx.draw_networkx_nodes(proc_graph,pos,
                nodelist=proc_vertices,
                node_size=1500,
                node_color='pink',
                alpha=0.8)
nx.draw_networkx_edges(proc_graph,pos,
                    edgelist = [(t[0],t[2]) for t in proc_linked_pairs],
                    width=[t[1] for t in proc_linked_pairs],
                    edge_color='lightblue',
                    alpha=0.8)
nx.draw_networkx_labels(proc_graph, pos,
                        labels=proc_vertex_labels,
                        horizontalalignment='left',
                        font_color='black')
nx.draw_networkx_edge_labels(proc_graph, pos,
                        edge_labels={(t[0],t[2]):t[1] for t in proc_linked_pairs},
                        font_color='black')

plt.grid(False)
st.pyplot(f)

# *********************************************************************************************************************


st.subheader("Distribution of number of agreements signed across actors")

st.write('THIS IS ILLUSTRATIVE ONLY. THERE ARE MANY OPPORTUNITIES FOR MORE MEANINGFUL ANALYSES.')

# Get the column marginals
col_marginals = []
for row in data_dict['matrix'].T:
    col_marginals.append(sum(row))


f = plt.figure(figsize=(16,16))
plt.plot(range(0,len(col_marginals)),col_marginals)
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.xlabel('Actor index',fontsize='xx-large')
plt.ylabel('Number of agreements signed',fontsize='xx-large')
st.pyplot(f)

# Top-scoring 10 actors
st.write('Top scoring actors across all agreements.')
x = np.argsort(col_marginals)[::-1][:10]
for index in x:
    actor = data_dict['actor_vertices'][index]
    st.write(col_marginals[index],actor,data_dict['vertices_dict'][actor][5])

st.header("Caveats")

st.write('1. Following the happy path.')
st.write('2. Data integrity checks required.')
st.write('3. Systematic testing required.')

