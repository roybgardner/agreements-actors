# Agreement-actor networks - DEPRECATED

29/07/2024. Analysis here is now superceded by analysis in paa-x repo.

Port of Jupyter notebook into Streamlit app. Supports:

1. Signatory data load into a data dictionary.
2. Generation of complete agreement-actor matrix (not an adjacency matrix). Columns marginal provides counts of actors over all agreements.
3. Extraction of peace process sub-matrices and generation of adjacency matrix representations of graphs.
4. Querying peace process adjacency matrices using depth-first search to generate adjacency matrices containing only the found vertices and the query vertices.
5. Use of a peace process sub-matrix $A$ to generate actor and agreement co-occurrence matrices.
   Actor co-occurrence matrix $V=A^TA$, agreement co-occurrence matrix $V=AA^T$.
   The co-occurrence matrices diagonals provide the column and row marginals of a peace process sub-matrix.
   Only need the upper triangle (excluding the diagonal) of co-occurrence matrices for graph generation.
   Recovery of coocurring entities.
6. Various visualisations based on metadata attribute values.


The app can be viewed here: [https://agreements-actors-jm3mmkozifmylnrtgjy4nb.streamlit.app](https://agreements-actors-jm3mmkozifmylnrtgjy4nb.streamlit.app/)
