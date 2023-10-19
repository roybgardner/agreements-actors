# Agreement-actor networks

Port of Jupyter notebook into Streamlit. Supports:

1. Signatory data load.
2. Generation of complete agreement-actor matrix (not an adjacency matrix). Column marginal provides counts of actors over all agreements.
3. Extraction of peace process sub-matrices and generation of graphs.
4. Querying the peace process graph by agreements and/or actors.
5. Use of a peace process sub-matrix $A$ to generate actor and agreement co-occurrence matrices.
   Actor co-occurrence matrix $V=A^TA$, agreement co-occurrence matrix $V=AA^T$.
   The co-occurrence matrices diagonals provide the column and row marginals of peace process sub-matrix.
   Only need the upper triangle of the co-occurrence matrices for graph generation.
