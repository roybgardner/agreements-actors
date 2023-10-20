import streamlit as st
import networkx as nx
from networkx.readwrite import json_graph

import numpy as np
import matplotlib.pyplot as plt

import json
import os
import csv
import sys

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

    # Build an edge dict with agreement as key and actor as value
    edge_dict = {}
    for row in links_data:
        if row[5] in edge_dict:
            edge_dict[row[5]].append(row[12])
        else:
            edge_dict[row[5]] = [row[12]]
    
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
    
    matrix = []
    for agreement in agreement_vertices:
        row = [0]*len(actor_vertices)
        for i,actor in enumerate(actor_vertices):
            if actor in edge_dict[agreement]:
                row[i] = 1
        matrix.append(row)
    matrix = np.array(matrix)
    
    #data_dict['nodes_data'] = nodes_data - DON'T NEED THIS
    data_dict['nodes_header'] = nodes_header
    data_dict['links_data'] = links_data
    data_dict['links_header'] = links_header
    data_dict['agreement_vertices'] = agreement_vertices
    data_dict['actor_vertices'] = actor_vertices
    data_dict['vertices_dict'] = vertices_dict
    data_dict['color_map'] = color_map
    data_dict['matrix'] = matrix

    return data_dict

def get_peace_processes(data_dict):
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
    pp_data_dict = {}
    pp_data_dict['pp_actor_ids'] = pp_actor_ids
    pp_data_dict['pp_agreement_ids'] = pp_agreement_ids
    pp_data_dict['pp_matrix'] = pp_matrix
    
    # Build a graph
    linked_pairs = []
    for i,row in enumerate(pp_matrix):    
        linked_pairs.extend([(pp_agreement_ids[i],v,pp_actor_ids[j]) for j,v in enumerate(row) if v > 0])
    pp_graph = nx.Graph()

    vertices = []
    vertices.extend([t[0] for t in linked_pairs])
    vertices.extend([t[2] for t in linked_pairs])
    vertices = list(set(vertices))

    pp_graph.add_nodes_from(vertices)
    for pair in linked_pairs:
        pp_graph.add_edge(pair[0],pair[2],weight=pair[1])
    
    pp_data_dict['pp_graph'] = {}
    pp_data_dict['pp_graph'] = pp_graph
    pp_data_dict['pp_node_colors'] = [data_dict['color_map'][v.split('_')[0]] for v in vertices]

    return pp_data_dict

def query_graph(graph,query_vertices=[],operator='AND',depth=1):
    tree_list = []
    for v in query_vertices:
         tree_list.append(nx.dfs_tree(graph,source=v,depth_limit=depth))

    found_vertices = set(tree_list[0].nodes) 
    for tree in tree_list:
        if operator == 'AND':
            found_vertices = found_vertices.intersection(set(tree.nodes))
        else:
            found_vertices = found_vertices.union(set(tree.nodes))

    found_vertices = list(found_vertices)    
    found_vertices.extend(query_vertices)

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
    results_dict = {}
    results_dict['graph'] = found_graph
    results_dict['node_colors'] = node_colors
    return results_dict

def display_graph(graph,node_colors):
    f = plt.figure(figsize=(16,16))
    pos = nx.spring_layout(graph) 
    nx.draw_networkx(graph,pos,node_color=node_colors,font_size='8',alpha=0.8)
    plt.grid(False)
    st.pyplot(f)

def get_cooccurrence_matrices(pp_matrix):
    # Actor-actor co-occurence matrix for a peace process
    V = np.matmul(pp_matrix.T,pp_matrix)
    # Agreement-agreement co-occurence matrix
    W = np.matmul(pp_matrix,pp_matrix.T)
    return (V,W)

def display_cooccurrence_network(key,co_matrices,pp_data_dict,data_dict,threshold):
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
                    alpha=0.7)
    nx.draw_networkx_edges(actor_graph,pos,
                        edgelist = [(t[0],t[2]) for t in linked_pairs],
                        width=[t[1] for t in linked_pairs],
                        edge_color='lightblue',
                        alpha=0.6)
    nx.draw_networkx_labels(actor_graph, pos,
                            labels=vertex_labels,
                            horizontalalignment='left',
                            font_color='black')
    nx.draw_networkx_edge_labels(actor_graph, pos,
                            edge_labels={(t[0],t[2]):t[1] for t in linked_pairs},
                            font_color='black')

    plt.grid(False)
    st.pyplot(f)


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

# Load data from CSV
data_path = './data/'
nodes_file = 'node_table.csv'
links_file = 'links_table.csv'
agreements_dict = 'agreements_dict.json'
data_dict = load_agreement_actor_data(nodes_file,links_file,agreements_dict,data_path)


st.write('Analysis of Niamh Henry\'s signatory data. Objectives are to support:') 
st.write('1. Simple extraction of peace process data.') 
st.write('2. Querying of peace process networks.') 
st.write('3. Generation of co-occurrence networks measuring a) the number of agreements to which a pair of actors are co-signatories, b) the number of signatories a pair of agreements have in common.') 
st.write('4. Access to descriptive statistics.') 


st.header("Peace Process Network Analysis")


# Select a peace process
st.subheader("Select a peace process")
pp_names = get_peace_processes(data_dict)
pp_selection=st.selectbox("", pp_names, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose a Peace Process", disabled=False, label_visibility="visible")
pp_data_dict = get_peace_process_data(pp_selection,data_dict)
# Display peace process graph
pp_graph = pp_data_dict['pp_graph']
node_colors = pp_data_dict['pp_node_colors']
display_graph(pp_graph,node_colors)


#Query vertices using depth-first search
with st.form("query"):
    st.subheader("Query peace process network")
    # Build the options - NEED TO WORK ON THIS
    option_list = []
    option_list.extend(pp_data_dict['pp_actor_ids'])
    option_list.extend(pp_data_dict['pp_agreement_ids'])
    option_list = sorted(option_list,reverse=True)
    option_list = [vertex_id + ': ' + data_dict['vertices_dict'][vertex_id][5] for vertex_id in option_list]

    options = st.multiselect(
    'Select actors and/or agreements',
    option_list,
    [])

    operator=["AND", "OR"]
    select_operator=st.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, horizontal=False, captions=None, label_visibility="visible")
    depth=st.slider("Select depth", min_value=1, max_value=3, value=1, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

   # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        options = [v.split(':')[0] for v in options]
        results_dict = query_graph(pp_graph,query_vertices=options,operator=select_operator,depth=depth)
        display_graph(results_dict['graph'],results_dict['node_colors'])


# Co-occurrence networks
with st.form("cooccurrence"):
    st.subheader("Actor and agreement co-occurrences in peace process")
    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    actor_upper = np.triu(co_matrices[0],k=1)
    agreement_upper = np.triu(co_matrices[1],k=1)
    actor_threshold=st.slider("Actor co-occurrence threshold", min_value=np.amin(actor_upper), max_value=np.amax(actor_upper), value=1, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    agreement_threshold=st.slider("Agreement co-occurrence threshold", min_value=np.amin(agreement_upper), max_value=np.amax(agreement_upper), value=1, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

   # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write('Edge values are the number of agreements to which a pair of actors are co-signatories.')
        display_cooccurrence_network('actor',co_matrices,pp_data_dict,data_dict,actor_threshold)
        st.write('Edge values are the number of signatories a pair of agreements have in common.')
        display_cooccurrence_network('agreement',co_matrices,pp_data_dict,data_dict,agreement_threshold)

st.subheader("Actor signatory counts in selected peace process")

st.write(pp_data_dict['pp_matrix'].shape[0])
st.write(pp_data_dict['pp_matrix'].shape[1])

# Get the actor co-occurrence matrix diagonal - it's equal to the columns marginal of the peace process matrix
actor_diag = np.diag(co_matrices[0])

# Plot
labels = [data_dict['vertices_dict'][v][5] for v in pp_data_dict['pp_actor_ids']]
z = list(zip(labels,actor_diag))
z = sorted(z,key=lambda t:t[1])

f = plt.figure(figsize=(8,32))
plt.barh(range(0,len(actor_diag)),[t[1] for t in z])
plt.yticks(range(0,len(actor_diag)),[t[0] for t in z],fontsize='large')
plt.xlabel('Number of agreements to which actor is signatory')
st.pyplot(f)

st.header("Distribution of number of agreements signed across actors")

