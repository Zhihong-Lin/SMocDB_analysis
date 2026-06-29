import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
sc.set_figure_params(dpi=100)
sc.settings.verbosity = 0
plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["pdf.fonttype"] = 42
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

class SpatialDataAligner:
    def __init__(self, adata_RNA, adata_pro):
        self.adata_RNA = adata_RNA
        self.adata_pro = adata_pro
        
        self.original_RNA_coords = adata_RNA.obsm['spatial_fov'].copy()
        self.original_pro_coords = adata_pro.obsm['spatial_fov'].copy()
        
        self.params = {
            'translate_x': 0,
            'translate_y': 0,
            'rotation': 0,
            'scale_x': 1.0,
            'scale_y': 1.0,
            'flip_x': False,
            'flip_y': False
        }
    
    def transform_coordinates(self, coords: np.ndarray) -> np.ndarray:
        transformed = coords.copy()
        
        if self.params['flip_x']:
            transformed[:, 0] = -transformed[:, 0]
        if self.params['flip_y']:
            transformed[:, 1] = -transformed[:, 1]
        
        transformed[:, 0] *= self.params['scale_x']
        transformed[:, 1] *= self.params['scale_y']
        
        if self.params['rotation'] != 0:
            theta = np.radians(self.params['rotation'])
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            x_rot = transformed[:, 0] * cos_theta - transformed[:, 1] * sin_theta
            y_rot = transformed[:, 0] * sin_theta + transformed[:, 1] * cos_theta
            transformed[:, 0] = x_rot
            transformed[:, 1] = y_rot
        
        transformed[:, 0] += self.params['translate_x']
        transformed[:, 1] += self.params['translate_y']
        
        return transformed
    
    def set_parameters(self, 
                       translate_x: float = None,
                       translate_y: float = None,
                       rotation: float = None,
                       scale_x: float = None,
                       scale_y: float = None,
                       flip_x: bool = None,
                       flip_y: bool = None):
        if translate_x is not None:
            self.params['translate_x'] = translate_x
        if translate_y is not None:
            self.params['translate_y'] = translate_y
        if rotation is not None:
            self.params['rotation'] = rotation
        if scale_x is not None:
            self.params['scale_x'] = scale_x
        if scale_y is not None:
            self.params['scale_y'] = scale_y
        if flip_x is not None:
            self.params['flip_x'] = flip_x
        if flip_y is not None:
            self.params['flip_y'] = flip_y
    
    def plot_alignment(self, 
                      figsize: Tuple[int, int] = (10, 10),
                      alpha_RNA: float = 0.3,
                      alpha_pro: float = 1,
                      s_RNA: float = 5,
                      s_pro: float = 5,
                      title: str = "Spatial Alignment",
                      show: bool = True):
        transformed_RNA_coords = self.transform_coordinates(self.original_RNA_coords)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(self.original_pro_coords[:, 0], 
                   self.original_pro_coords[:, 1], 
                   c='#f9d5a2', alpha=alpha_pro, s=s_pro, 
                   label='Protein (reference)', edgecolors='none', linewidths=0.5)
        
        ax.scatter(transformed_RNA_coords[:, 0], 
                   transformed_RNA_coords[:, 1], 
                   c='#729DAD', alpha=0.1, s=s_RNA, 
                   label='RNA (transformed)', edgecolors='none', linewidths=0.5)
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, ax
    
    def get_current_parameters(self) -> dict:
        return self.params.copy()
    
    def get_transformed_coordinates(self) -> np.ndarray:
        return self.transform_coordinates(self.original_RNA_coords)
    
    def save_alignment(self, output_path: str):
        import pickle
        
        alignment_data = {
            'parameters': self.params,
            'original_RNA_coords': self.original_RNA_coords,
            'original_pro_coords': self.original_pro_coords,
            'transformed_RNA_coords': self.get_transformed_coordinates()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(alignment_data, f)
        print(f"Alignment saved to {output_path}")


def find_nearest_neighbors(adata_RNA, adata_pro):
    RNA_coords = adata_RNA.obsm['spatial_fov']
    pro_coords = adata_pro.obsm['spatial_fov']
    
    RNA_cell_names = adata_RNA.obs_names.values
    pro_cell_names = adata_pro.obs_names.values

    pro_tree = KDTree(pro_coords)
    distances, indices = pro_tree.query(RNA_coords, k=1)
    
    matching_matrix = pd.DataFrame({
        'RNA_cell': RNA_cell_names,
        'Protein_cell': pro_cell_names[indices],
        'distance': distances
    })
    
    print(f"\nMatching completed! Find the {len(matching_matrix)} matching pairs")
    print(f"mean distance: {distances.mean():.4f}")
    
    return matching_matrix


def create_matched_anndata_with_duplication(adata_RNA, adata_pro, matching_matrix):
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    
    valid_matches = matching_matrix.dropna(subset=['RNA_cell', 'Protein_cell']).copy()
    
    print(f"Total matching pairs: {len(valid_matches)}")
    
    protein_match_counts = valid_matches['Protein_cell'].value_counts()
    multiply_matched_proteins = protein_match_counts[protein_match_counts > 1]
    
    print(f"\nMatching statistics:")
    print(f"Unique RNA cells: {valid_matches['RNA_cell'].nunique()}")
    print(f"Unique protein cells: {valid_matches['Protein_cell'].nunique()}")
    print(f"Protein cells matched by multiple RNAs: {len(multiply_matched_proteins)}")
    
    if len(multiply_matched_proteins) > 0:
        print("\nMultiply matched protein cells:")
        for protein, count in multiply_matched_proteins.head(10).items():
            print(f"  {protein}: matched by {count} RNA cells")
        if len(multiply_matched_proteins) > 10:
            print(f"  ... and {len(multiply_matched_proteins) - 10} more")
    
    RNA_indices = []
    pro_indices = []
    original_protein_names = []
    
    for _, row in valid_matches.iterrows():
        RNA_cell = row['RNA_cell']
        pro_cell = row['Protein_cell']
        
        RNA_idx = np.where(adata_RNA.obs_names == RNA_cell)[0]
        if len(RNA_idx) == 0:
            print(f"Warning: RNA cell {RNA_cell} not found in original data, skipping")
            continue
        RNA_indices.append(RNA_idx[0])
        
        pro_idx = np.where(adata_pro.obs_names == pro_cell)[0]
        if len(pro_idx) == 0:
            print(f"Warning: Protein cell {pro_cell} not found in original data, skipping")
            continue
        
        pro_indices.append(pro_idx[0])
        original_protein_names.append(pro_cell)
    
    print(f"\nSuccessfully processed matching pairs: {len(RNA_indices)}")
    
    adata_RNA_matched = adata_RNA[RNA_indices].copy()
    adata_pro_matched = adata_pro[pro_indices].copy()
    
    new_obs_names = [f"matched_pair_{i:05d}" for i in range(len(RNA_indices))]
    adata_RNA_matched.obs_names = new_obs_names
    adata_pro_matched.obs_names = new_obs_names
    
    adata_RNA_matched.obs['original_RNA_name'] = valid_matches['RNA_cell'].values[:len(RNA_indices)]
    adata_RNA_matched.obs['matched_Protein_name'] = original_protein_names
    adata_RNA_matched.obs['matching_distance'] = valid_matches['distance'].values[:len(RNA_indices)]
    
    adata_pro_matched.obs['original_Protein_name'] = original_protein_names
    adata_pro_matched.obs['matched_RNA_name'] = valid_matches['RNA_cell'].values[:len(RNA_indices)]
    adata_pro_matched.obs['matching_distance'] = valid_matches['distance'].values[:len(RNA_indices)]
    
    if len(multiply_matched_proteins) > 0:
        is_duplicate_protein = adata_pro_matched.obs['original_Protein_name'].duplicated(keep='first')
        adata_pro_matched.obs['is_duplicate_protein'] = is_duplicate_protein
        adata_RNA_matched.obs['is_duplicate_protein'] = is_duplicate_protein
        
        print(f"\nCreated protein cell duplicates: {is_duplicate_protein.sum()}")

    
    return adata_RNA_matched, adata_pro_matched

adata1=sc.read_h5ad("001_RNA.h5ad")
adata2=sc.read_h5ad("001_pro.h5ad")
if "spatial_fov" not in adata1.obsm:
    adata1.obsm["spatial_fov"] = adata1.obsm["spatial"]    
if "spatial_fov" not in adata2.obsm:
    adata2.obsm["spatial_fov"] = adata2.obsm["spatial"]    

adata_RNA=adata1
adata_pro=adata2
# 1. Alignment
aligner = SpatialDataAligner(adata_RNA, adata_pro)
aligner.set_parameters(
    translate_x=2800,  
    translate_y=4580,   
    rotation=200,      
    scale_x=2.135,     
    scale_y=2.135,      
    flip_x=False,    
    flip_y=False      
)
aligner.plot_alignment()
params = aligner.get_current_parameters()
params_df = pd.DataFrame([params]) 
params_df.to_csv('001/alignment_parameters.csv', index=False)
# 2.transformed_coords
transformed_coords = aligner.get_transformed_coordinates()
adata_RNA.obsm['spatial_fov'] = transformed_coords
# 3.validation and save
aligner = SpatialDataAligner(adata_RNA, adata_pro)
aligner.set_parameters(
    translate_x=0,   
    translate_y=0,  
    rotation=0,      
    scale_x=1,       
    scale_y=1,       
    flip_x=False,     
    flip_y=False      
)
plt.figure(figsize=(8, 8))
fig, ax = aligner.plot_alignment(show=False)
fig.savefig("001/alignment_plot.pdf", format="pdf", dpi=300)
plt.figure(figsize=(8, 8))
fig, ax = aligner.plot_alignment(show=False)
fig.savefig("001/alignment_plot.png", format="png", dpi=200)
# 4.matching matrix
matching_matrix = find_nearest_neighbors(adata_RNA, adata_pro)
matching_matrix.to_csv('001/matching_matrix.csv', index=False)
# 5.create matched anndata
adata_RNA_matched, adata_pro_matched = create_matched_anndata_with_duplication(
    adata_RNA, 
    adata_pro, 
    matching_matrix
)
adata_pro_matched.obsm["spatial_fov"]=adata_RNA_matched.obsm["spatial_fov"]
sc.write("001/adata_RNA_matched.h5ad", adata_RNA_matched)
sc.write("001/adata_pro_matched.h5ad", adata_pro_matched)