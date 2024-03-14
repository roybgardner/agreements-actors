import streamlit as st

from streamlit_shared import *

# Main data_dict for all pages that need it
if "data_dict" not in st.session_state:
    st.session_state["data_dict"] = {}

# State for selected peace process
if "pp_data_dict" not in st.session_state:
    st.session_state["pp_data_dict"] = {}
if "adj_matrix" not in st.session_state:
    st.session_state["adj_matrix"] = []
if "adj_vertices" not in st.session_state:
    st.session_state["adj_vertices"] = []       

# State for the query page
if "selected_actors" not in st.session_state:
    st.session_state["selected_actors"] = []       
if "selected_agreements" not in st.session_state:
    st.session_state["selected_agreements"] = []       

# State for the actor signatory page
if "selected_data_actor" not in st.session_state:
    st.session_state["selected_data_actor"] = ''      

# State for the actor metadata page
if "selected_metadata_actor" not in st.session_state:
    st.session_state["selected_metadata_actor"] = ''      

# State for the agreement data page
if "selected_data_agreement" not in st.session_state:
    st.session_state["selected_data_agreement"] = ''      



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

st.subheader('Demonstration') 

st.write('This demonstration illustrates a few of the many ways in which\
          agreement-actor signatory data can be analysed and displayed.\
          This demo serves to give an idea of its potential.')

st.write('The demonstrator applies the methodology describe below in order to interrogate, analyse,\
          and visualise agreement-actor signatory data.\
         Network representations of the data and the ability to query these networks are a core feature of the methodology.\
         Numerical values can be displayed in tables and used for statistical analysis.')

st.write('The demo is limited in scope, especially in terms of user interaction with networks and other visualisations.\
          The potential for such interaction is described in ‘Potential function’ sections.')

st.subheader('Methodology') 

st.write('Agreement-actor signatory data are an example of an undirected bipartite graph, i.e.,\
          there are edges (links) between agreements and actors, but not between agreements or between actors.') 
st.write('Agreement-actor bipartite graphs are represented by binary-valued biadjacency matrices where\
          the rows of the matrix correspond to agreements and the columns to actors. Cell values contain\
          the value 1 if an actor is a signatory to an agreement, otherwise cells values are 0.') 
st.write('Agreement-actor biadjacency matrices, together with actor and agreement data, provide the basis of peace process network analysis as follows:') 
st.write('1. Extraction of biadjacency matrices containing data from individual peace processes.') 
st.write('2. Generation of full adjacency matrices used for depth-first search network queries. \
          Adjacency matrices can be passed to network packages for network visualisation.') 
st.write('3. Generation of co-occurrence matrices measuring, a)\
          the number of agreements to which a pair of actors are co-signatories, b)\
          the number of signatories a pair of agreements have in common.\
         The entities in co-occurrence matrix cells can be recovered.') 
st.write('4. Support for metadata-based analysis within and across peace processes.') 
