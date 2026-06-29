import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import cellcharter as cc
from scimilarity import CellAnnotation
from scimilarity.utils import align_dataset, lognorm_counts
model_path = "./model_v1.1/"
ca = CellAnnotation(model_path=model_path)
import matplotlib
import matplotlib.pyplot as plt
from umap import UMAP
import sklearn
import seaborn as sns
import os
import scanpy as sc
import scvi
import squidpy as sq

random_seed = 20
input_dir = "/2.Result/Database/CosMx_v2/"
output_dir = "/2.Result/Database/2.Downstream/1107/"
output_file= "/2.Result/Database/2.Downstream/1107/spatial_analysis_log.txt"
for file_name in file_names:
    with open(output_file, 'a') as f:  
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{current_time}] ------------- {file_name} start! -------------\n")
    # ======== Analysis Program ===========
    input_path = os.path.join(input_dir, file_name)  
    adata_output_folder = os.path.join(output_dir,"1.AnndataObject/")  
    os.makedirs(adata_output_folder, exist_ok=True)  
    figures_output_folder = os.path.join(output_dir,"2.DownstreamFigures/", file_name.replace(".h5ad", "")) 
    os.makedirs(figures_output_folder, exist_ok=True)  

    try:
        adata = sc.read_h5ad(input_path)
        if "spatial_fov" not in adata.obsm:
            adata.obsm["spatial_fov"] = adata.obsm["spatial"]    
        adata.layers["counts"] = adata.X.copy()


        # ------------1. Spatial domain detection-----------
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=30)  
        adata.obsm['X_pca'] = adata.obsm['X_pca'].astype(np.float32)
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial_fov', percentile=99)
        cc.gr.remove_long_links(adata)
        cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_pca', out_key='X_cellcharter', sample_key='sample')
    
        autok = cc.tl.ClusterAutoK(n_clusters=(2,15), max_runs=10, convergence_tol=0.001)
        autok.fit(adata, use_rep='X_cellcharter')
        cc.pl.autok_stability(autok)
        plt.savefig(os.path.join(figures_output_folder, "1.1autok_stability_plot.pdf"), format="pdf", dpi=300)
        plt.savefig(os.path.join(figures_output_folder, "1.1autok_stability_plot.png"), format="png", dpi=300)
        plt.close()
        
        adata.obs['cluster_cellcharter'] = autok.predict(adata, use_rep='X_cellcharter')
        sq.pl.spatial_scatter(adata, shape=None, color=["cluster_cellcharter"], wspace=0.4, library_id='spatial', spatial_key='spatial_fov')
        plt.savefig(os.path.join(figures_output_folder, "1.2cluster_cellcharter_plot.pdf"), format="pdf", dpi=300)
        plt.savefig(os.path.join(figures_output_folder, "1.2cluster_cellcharter_plot.png"), format="png", dpi=300)
        plt.close()

        # ------------2. cell type annotation-----------
        adata_for_annotation  = align_dataset(adata, ca.gene_order,gene_overlap_threshold=0)
        adata_for_annotation  = lognorm_counts(adata_for_annotation )
        adata_for_annotation .obsm["X_scimilarity"] = ca.get_embeddings(adata_for_annotation .X)
        sc.pp.neighbors(adata_for_annotation , use_rep="X_scimilarity")
        sc.tl.umap(adata_for_annotation )
        target_celltypes = [
            "B cell",
            "plasma cell",
            "CD4-positive, alpha-beta T cell",
            "CD8-positive, alpha-beta T cell",
            "endothelial cell",
            "epithelial cell",
            "fibroblast",
            "mast cell",
            "myeloid cell"
        ]
        ca.safelist_celltypes(target_celltypes)
        # The ca.annotate_dataset function will now only classify to the safelisted cell types
        adata_for_annotation  = ca.annotate_dataset(adata_for_annotation )
        sq.pl.spatial_scatter(adata_for_annotation , shape=None, color=["celltype_hint"], wspace=0.4, library_id='spatial', size=0.01, spatial_key='spatial_fov')
        plt.savefig(os.path.join(figures_output_folder, "assigned_cell_type_plot.pdf"), format="pdf", dpi=300)
        plt.savefig(os.path.join(figures_output_folder, "assigned_cell_type_plot.png"), format="png", dpi=300)
        plt.close()

        adata.obs["min_dist"] = adata_for_annotation.obs["min_dist"]
        adata.obs["celltype_hint"] = adata_for_annotation.obs["celltype_hint"]
        adata.uns["celltype_hint_colors"] = adata_for_annotation.uns["celltype_hint_colors"]
        adata.obsm["X_umap"] = adata_for_annotation.obsm["X_umap"]

        # ------------3. UMAP and marker gene plot for cell type -----------
        sc.pl.umap(adata, color=["celltype_hint"], frameon=False,show=False)
        plt.savefig(os.path.join(figures_output_folder, "umap_plot.png"), format="png", dpi=100,bbox_inches='tight')
        plt.savefig(os.path.join(figures_output_folder, "umap_plot.pdf"), format="pdf", dpi=300,bbox_inches='tight')
        plt.close()
        marker_genes = {
            "CD8+ T Cells": ["CD3D", "CD3E", "CD8A"],
            "CD4+ T Cells": ["CD3D", "CD3E", "CD4"],
            "B cells": ["CD19", "BANK1", "MS4A1"],
            "Plasma cells": ["IGHG1", "IGKC"],
            "Myeloid cell": ["CD33","CD68", "CD14","CD11c"],
            "Fibroblasts": ["DCN","FAP","COL1A1"],
            "Endothelial cells": ["PECAM1", "VWF"],
            "Tumour cell": ["EPCAM", "KRT19", "PROM1", "ALDH1A1", "CD24"],
            "Mast": ["CPA3", "CST3", "KIT", "TPSAB1", "TPSB2", "MS4A2"]
        }
        marker_genes_filtered = {
            cell_type: [gene for gene in genes if gene in adata.var_names]
            for cell_type, genes in marker_genes.items()
        }
        sc.pl.dotplot(adata, marker_genes_filtered, groupby="celltype_hint", standard_scale="var",show=False)
        plt.savefig(os.path.join(figures_output_folder, "marker_gene.pdf"), format="pdf", dpi=300,bbox_inches='tight')
        plt.savefig(os.path.join(figures_output_folder, "marker_gene.png"), format="png", dpi=300,bbox_inches='tight')
        plt.close() 

         # ------------5. SVG-----------
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial_fov', percentile=99)
        sq.gr.spatial_autocorr(
            adata,
            mode="moran",
            n_perms=100,
            n_jobs=1,
        )
        SVG_top50 = adata.uns["moranI"].head(50)
        SVG_top50.to_csv(os.path.join(figures_output_folder, "SVG_top50.csv"))

        # ------------Save the processed H5AD file-----------
        output_h5ad_path = os.path.join(adata_output_folder, file_name)
        adata.X=adata.layers["counts"]
        sc.write(output_h5ad_path,adata)
    except Exception as e:
        # ======== Analysis Program ===========
        with open(output_file, 'a') as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] {file_name} fails!\n\n")
with open(output_file, 'a') as f:
    f.write(f"Analysis completed. Log saved to: {output_file}")
    
print(f"Analysis completed. Log saved to: {output_file}")