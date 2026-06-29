import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from umap import UMAP
import sklearn
import seaborn as sns
import os
os.chdir('COSMOS-main/')
import COSMOS
from COSMOS import cosmos
from COSMOS.pyWNN import pyWNN
import warnings
warnings.filterwarnings('ignore')
random_seed = 20
import scanpy as sc
import scvi
import squidpy as sq

sc.pp.normalize_total(adata1)
sc.pp.log1p(adata1)
sc.pp.normalize_total(adata2)
sc.pp.log1p(adata2)

loc = adata1.obsm['spatial_fov']
adata1.obs['x_pos'] = np.array(loc)[:, 0]
adata1.obs['y_pos'] = np.array(loc)[:, 1]
adata2.obs['x_pos'] = np.array(loc)[:, 0]
adata2.obs['y_pos'] = np.array(loc)[:, 1]

cosmos_comb = cosmos.Cosmos(adata1=adata1, adata2=adata2)
cosmos_comb.preprocessing_data(n_neighbors=10)
cosmos_comb.train(
    spatial_regularization_strength=0.05, 
    z_dim=50,
    lr=1e-3, 
    wnn_epoch=500, 
    total_epoch=1000, 
    max_patience_bef=10,
    max_patience_aft=30, 
    min_stop=200,
    random_seed=random_seed, 
    gpu=0, 
    regularization_acceleration=True, 
    edge_subset_sz=1000000
)

df_embedding = pd.DataFrame(cosmos_comb.embedding)
embedding_adata = sc.AnnData(df_embedding)
embedding_adata.obsm["spatial_fov"] = adata1.obsm['spatial_fov']

cosmos_output_path = os.path.join(cosmos_output_folder, f"embedding_adata.h5ad")
sc.write(cosmos_output_path, embedding_adata)