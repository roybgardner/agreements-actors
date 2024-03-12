import streamlit as st

from streamlit_shared import *

if "data_dict" not in st.session_state:
    st.session_state["data_dict"] = {}
if "pp_data_dict" not in st.session_state:
    st.session_state["pp_data_dict"] = {}
if "adj_matrix" not in st.session_state:
    st.session_state["adj_matrix"] = []
if "adj_vertices" not in st.session_state:
    st.session_state["adj_vertices"] = []       

if "selected_actors" not in st.session_state:
    st.session_state["selected_actors"] = []       
if "selected_agreements" not in st.session_state:
    st.session_state["selected_agreements"] = []       




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

    
if len(st.session_state["data_dict"]) == 0:
    data_dict = load_agreement_actor_data(nodes_file,links_file,agreements_dict,data_path)
    st.session_state["data_dict"] = data_dict
else:
    data_dict = st.session_state["data_dict"]
    

st.header('Credits') 
st.write('Signatory data: Niamh Henry and Sanja Badanjak') 
st.write('Analysis/coding: Roy Gardner') 

st.header("Introduction")

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
