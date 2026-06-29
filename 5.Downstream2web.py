import os
import traceback
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import anndata as ad
import scanpy as sc
import squidpy as sq
import cellcharter as cc
import scvi
from lightning.pytorch import seed_everything
seed_everything(12345)
scvi.settings.seed = 12345

def create_expression_dataframe(adata, coords_key, coord_names):
    coords = adata.obsm[coords_key][:, :len(coord_names)]
    
    if hasattr(adata.X, "toarray"):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X.copy()
    
    df = pd.DataFrame(
        data=np.column_stack([coords, expr_matrix]),
        columns=coord_names + adata.var_names.tolist(),
        index=adata.obs_names.tolist()
    )
    return df

def get_point_size(n_cells):
    if n_cells < 200000:
        return 3
    elif n_cells < 400000:
        return 1
    else:
        return 0.5

def create_scatter_plot(df, x_col, y_col, color_col, cmap_or_palette, point_size=1, 
                        alpha=0.7, vmax=None, save_path=None, title=None, 
                        xlabel=None, ylabel=None, legend_title=None, 
                        hide_ticks=False, legend_labels=None, colors=None):
    plt.figure(figsize=(10, 8))
    
    is_categorical = df[color_col].dtype.name in ['category', 'object', 'bool'] or \
                     df[color_col].dtype.name.startswith('int') and len(df[color_col].unique()) <= 20
    
    if is_categorical:
        if isinstance(cmap_or_palette, dict):
            palette = cmap_or_palette
        else:
            palette = cmap_or_palette
        
        scatter = sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=color_col,
            palette=palette, s=point_size, alpha=alpha,
            edgecolor="none", legend=False
        )
        
        if legend_labels is not None:
            if colors is None:
                n_categories = len(legend_labels)
                colors = sns.color_palette("tab20", n_categories)
            
            legend_handles = [
                Line2D([0], [0], marker="o", color=c, markersize=8, 
                       linestyle="", alpha=alpha)
                for c in colors
            ]
            plt.legend(legend_handles, legend_labels, title=legend_title,
                      bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    else:
        scatter_kwargs = {
            'x': df[x_col], 'y': df[y_col], 'c': df[color_col],
            'cmap': cmap_or_palette, 's': point_size, 'alpha': alpha,
            'edgecolor': "none"
        }
        if vmax is not None:
            scatter_kwargs['vmax'] = vmax
        
        scatter = plt.scatter(**scatter_kwargs)
        
        cbar = plt.colorbar(scatter, shrink=0.8)
        cbar.set_label(f"{color_col} Expression", fontsize=12)
    
    if title:
        plt.title(title, fontsize=14)
    if xlabel:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)
    
    if hide_ticks:
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

def batch_plot_gene_expression(df, gene_columns, coord_names, plot_title_prefix, 
                               save_dir, point_size, vmax_percentile=95):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    x_col, y_col = coord_names[0], coord_names[1]
    is_spatial = plot_title_prefix.lower().startswith("spatial")
    
    for i, gene in enumerate(gene_columns):
        
        expr_values = df[gene].values
        vmax_90 = np.percentile(expr_values, vmax_percentile)
        vmax = max(vmax_90, 1)
        
        safe_name = gene.replace('/', '_').replace(' ', '_').replace(',', '').replace('-', '_')
        save_path = os.path.join(save_dir, f"{safe_name}.jpg")
        
        create_scatter_plot(
            df=df,
            x_col=x_col,
            y_col=y_col,
            color_col=gene,
            cmap_or_palette="RdPu",
            point_size=point_size,
            alpha=0.7,
            vmax=vmax,
            save_path=save_path,
            title=f"{plot_title_prefix}: {gene}",
            xlabel=x_col,
            ylabel=y_col,
            hide_ticks=is_spatial
        )
        
        if (i+1) % 50 == 0 or (i+1) == len(gene_columns):
            print(f" {i+1}/{len(gene_columns)} ({((i+1)/len(gene_columns)*100):.1f}%)")


def plot_celltype_distribution(data, x_col, y_col, title_prefix, save_dir, point_size=1):

    os.makedirs(save_dir, exist_ok=True)
    
    all_types = data["assigned_cell_type"].unique()
    colors = sns.color_palette("tab20", len(all_types))
    
    save_path_all = os.path.join(save_dir, f"{title_prefix.lower()}_all.png")
    create_scatter_plot(
        df=data,
        x_col=x_col,
        y_col=y_col,
        color_col="assigned_cell_type",
        cmap_or_palette="tab20",
        point_size=point_size,
        alpha=0.7,
        save_path=save_path_all,
        title=f"{title_prefix} of All Cell Types",
        xlabel=x_col,
        ylabel=y_col,
        legend_title="Cell Type",
        legend_labels=all_types,
        colors=colors,
        hide_ticks=(x_col == "x")
    )

    
    for i, ct in enumerate(all_types):
        plot_data = data.copy()
        plot_data['color_group'] = plot_data['assigned_cell_type'].apply(
            lambda x: ct if x == ct else 'Other'
        )
        
        color_dict = {ct: colors[i % 20], 'Other': '#CCCCCC'}
        legend_labels = [ct, 'Other cells']
        legend_colors = [colors[i % 20], '#CCCCCC']
        
        save_path = os.path.join(save_dir, f"{ct.replace('/', '_').replace(' ', '_').replace(',', '')}.png")
        
        create_scatter_plot(
            df=plot_data,
            x_col=x_col,
            y_col=y_col,
            color_col="color_group",
            cmap_or_palette=color_dict,
            point_size=point_size,
            alpha=0.7,
            save_path=save_path,
            title=f"{title_prefix}: {ct}",
            xlabel=x_col,
            ylabel=y_col,
            legend_title="Cell Type",
            legend_labels=legend_labels,
            colors=legend_colors,
            hide_ticks=(x_col == "x")
        )
        
        if (i+1) % 10 == 0 or (i+1) == len(all_types):
            print(f"  done {i+1}/{len(all_types)}")
    

def plot_spatial_clustering(df, cluster_col, k, save_path, point_size=1):

    df[cluster_col] = df[cluster_col].astype(int) 
    
    cluster_labels = sorted(df[cluster_col].unique())
    n_clusters = len(cluster_labels)
    
    colors = sns.color_palette("tab20", n_clusters)
    color_dict = {label: colors[i] for i, label in enumerate(cluster_labels)}
    
    df_plot = df.copy()
    df_plot['cluster_ordered'] = pd.Categorical(
        df_plot[cluster_col], 
        categories=cluster_labels, 
        ordered=True
    )
    
    create_scatter_plot(
        df=df_plot,
        x_col="x",
        y_col="y",
        color_col="cluster_ordered",
        cmap_or_palette=color_dict,
        point_size=point_size,
        alpha=0.7,
        save_path=save_path,
        title=f"Spatial Domain (k={k})",
        xlabel="X Coordinate",
        ylabel="Y Coordinate",
        legend_title="Spatial domain",
        legend_labels=[str(label) for label in cluster_labels],  
        colors=colors,
        hide_ticks=True
    )

infro_df = pd.read_excel("./1.infro_df.xlsx")

ANNDATA_ROOT = "./1.AnndataObject"
OUTPUT_ROOT = "./downstream_results/"
LOG_DIR = OUTPUT_ROOT
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"batch_run_{timestamp}.log")
summary_file = os.path.join(LOG_DIR, f"batch_run_summary_{timestamp}.csv")

logger = logging.getLogger("batch_run")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def run_single_sample(ID, Data_name):
    logger.info("=" * 80)
    logger.info(f"Start processing | ID={ID} | Data_name={Data_name}")

    adata_path = os.path.join(ANNDATA_ROOT, Data_name)
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"h5ad file not found: {adata_path}")

    adata = ad.read(adata_path)

    required_obs = ['celltype_hint']
    required_obsm = ['spatial_fov', 'X_umap']

    for col in required_obs:
        if col not in adata.obs.columns:
            raise ValueError(f"{col} not found in adata.obs")

    for key in required_obsm:
        if key not in adata.obsm:
            raise ValueError(f"{key} not found in adata.obsm")

    n_cells = adata.n_obs
    point_size = get_point_size(n_cells)
    logger.info(f"n_cells={n_cells}, point_size={point_size}")

    logger.info("Step 1: Draw a distribution map of gene expression")

    df_spatial = create_expression_dataframe(adata, "spatial_fov", ["x", "y"])
    save_dir_spatial = os.path.join(OUTPUT_ROOT, ID, "1.Gene expression", "1.Spatial distribution")
    gene_columns_spatial = [col for col in df_spatial.columns if col not in ['x', 'y']]
    logger.info(f"Spatial genes found: {len(gene_columns_spatial)}")

    batch_plot_gene_expression(
        df=df_spatial,
        gene_columns=gene_columns_spatial,
        coord_names=["x", "y"],
        plot_title_prefix="Spatial Expression of",
        save_dir=save_dir_spatial,
        point_size=point_size
    )

    df_umap = create_expression_dataframe(adata, "X_umap", ["UMAP_1", "UMAP_2"])
    save_dir_umap = os.path.join(OUTPUT_ROOT, ID, "1.Gene expression", "2.UMAP_distribution")
    gene_columns_umap = [col for col in df_umap.columns if col not in ['UMAP_1', 'UMAP_2']]
    logger.info(f"UMAP genes found: {len(gene_columns_umap)}")

    batch_plot_gene_expression(
        df=df_umap,
        gene_columns=gene_columns_umap,
        coord_names=["UMAP_1", "UMAP_2"],
        plot_title_prefix="UMAP: Expression Pattern of",
        save_dir=save_dir_umap,
        point_size=point_size
    )

    if "moranI" not in adata.uns:
        raise ValueError("moranI not found in adata.uns")

    SVG_top50 = adata.uns["moranI"].head(50)
    save_dir_gene = os.path.join(OUTPUT_ROOT, ID, "1.Gene expression")
    os.makedirs(save_dir_gene, exist_ok=True)
    SVG_top50.to_csv(os.path.join(save_dir_gene, "SVG_top50.csv"))
    logger.info("Saved SVG_top50.csv")

    logger.info("Step 2: Draw cell type distribution maps")

    cell_type_spatial = pd.DataFrame({
        'x': adata.obsm['spatial_fov'][:, 0],
        'y': adata.obsm['spatial_fov'][:, 1],
        'assigned_cell_type': adata.obs['celltype_hint'].tolist()
    }, index=adata.obs_names)

    cell_type_umap = pd.DataFrame({
        "UMAP_1": adata.obsm["X_umap"][:, 0],
        "UMAP_2": adata.obsm["X_umap"][:, 1],
        "assigned_cell_type": adata.obs["celltype_hint"].tolist()
    }, index=adata.obs_names)

    base_path_celltype = os.path.join(OUTPUT_ROOT, ID, "2.Cell type annotation")
    plot_celltype_distribution(
        cell_type_spatial, "x", "y", "Spatial Distribution",
        os.path.join(base_path_celltype, "1.Spatial distribution"), point_size
    )
    plot_celltype_distribution(
        cell_type_umap, "UMAP_1", "UMAP_2", "UMAP",
        os.path.join(base_path_celltype, "2.umap"), point_size
    )

    marker_genes = {
        "CD8+ T Cells": ["CD3D", "CD3E", "CD8A"],
        "CD4+ T Cells": ["CD3D", "CD3E", "CD4"],
        "B cells": ["CD19", "BANK1", "MS4A1"],
        "Plasma cells": ["IGHG1", "IGKC"],
        "Myeloid cell": ["CD33", "CD68", "CD14", "CD11c"],
        "Fibroblasts": ["DCN", "FAP", "COL1A1"],
        "Endothelial cells": ["PECAM1", "VWF"],
        "Tumour cell": ["EPCAM", "KRT19", "PROM1", "ALDH1A1", "CD24"],
        "Mast": ["CPA3", "CST3", "KIT", "TPSAB1", "TPSB2", "MS4A2"]
    }

    marker_genes_filtered = {
        cell_type: [gene for gene in genes if gene in adata.var_names]
        for cell_type, genes in marker_genes.items()
    }

    sc.pl.dotplot(
        adata,
        marker_genes_filtered,
        groupby="celltype_hint",
        standard_scale="var",
        show=False
    )
    os.makedirs(base_path_celltype, exist_ok=True)
    plt.savefig(
        os.path.join(base_path_celltype, "marker_gene.png"),
        format="png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    logger.info("Saved marker_gene.png")

    logger.info("Step 3: Spatial clustering analysis")
    sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial_fov', percentile=99)
    cc.gr.remove_long_links(adata)
    cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_pca', out_key='X_cellcharter', sample_key='sample')
    autok = cc.tl.ClusterAutoK(n_clusters=(2, 15), max_runs=10, convergence_tol=0.001)
    autok.fit(adata, use_rep='X_cellcharter')

    base_path_cluster = os.path.join(OUTPUT_ROOT, ID, "3.Spatial clustering")

    if not hasattr(pd.Series, "nonzero"):
        pd.Series.nonzero = lambda self: np.nonzero(self.values)

    sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)

    for k in range(2, 16):
        logger.info(f"Processing k={k}")

        save_path = os.path.join(base_path_cluster, f"{k-1}.k_{k}")
        os.makedirs(save_path, exist_ok=True)

        cc.pl.autok_stability(autok)
        plt.savefig(os.path.join(save_path, "0.autok_stability_plot.png"), format="png", dpi=300)
        plt.close()

        adata.obs[f'cluster_cellcharter_k{k}'] = autok.predict(adata, use_rep='X_cellcharter', k=k)

        df_cluster = pd.DataFrame({
            'x': adata.obsm['spatial_fov'][:, 0],
            'y': adata.obsm['spatial_fov'][:, 1],
            f'cluster_cellcharter_k{k}': adata.obs[f'cluster_cellcharter_k{k}'].tolist()
        }, index=adata.obs_names)

        plot_spatial_clustering(
            df=df_cluster,
            cluster_col=f'cluster_cellcharter_k{k}',
            k=k,
            save_path=os.path.join(save_path, "1.spatial_clustering.png"),
            point_size=point_size
        )

        cc.gr.enrichment(adata, group_key=f'cluster_cellcharter_k{k}', label_key='celltype_hint')
        df_cell_type_enrichment = adata.uns[f"cluster_cellcharter_k{k}_celltype_hint_enrichment"]["enrichment"]
        df_cell_type_enrichment.to_csv(
            os.path.join(save_path, "2.cell_type_enrichment.csv"),
            index=True,
            encoding="utf-8"
        )

        adata.obs[f'cluster_cellcharter_k{k}'] = adata.obs[f'cluster_cellcharter_k{k}'].astype('category')
        cc.gr.nhood_enrichment(adata, cluster_key=f'cluster_cellcharter_k{k}')
        cluster_nhood_enrichment = adata.uns[f"cluster_cellcharter_k{k}_nhood_enrichment"]["enrichment"]
        cluster_nhood_enrichment.to_csv(
            os.path.join(save_path, "3.nhood_enrichment.csv"),
            index=True,
            encoding="utf-8"
        )

        logger.info(f"Completed k={k}, results saved to {save_path}")

    logger.info(f"Finished successfully | ID={ID} | Data_name={Data_name}")



required_columns = ["ID", "Data_name"]
for col in required_columns:
    if col not in infro_df.columns:
        raise ValueError(f"{col} not found in infro_df")

results = []

logger.info(f"Total samples to process: {len(infro_df)}")

for idx, row in infro_df.iterrows():
    ID = str(row["ID"])
    Data_name = str(row["Data_name"])

    try:
        run_single_sample(ID, Data_name)
        results.append({
            "row_index": idx,
            "ID": ID,
            "Data_name": Data_name,
            "status": "success",
            "error_message": ""
        })
        logger.info(f"[SUCCESS] {ID} | {Data_name}")

    except Exception as e:
        err_msg = str(e)
        tb = traceback.format_exc()

        results.append({
            "row_index": idx,
            "ID": ID,
            "Data_name": Data_name,
            "status": "failed",
            "error_message": err_msg
        })

        logger.error(f"[FAILED] {ID} | {Data_name} | Error: {err_msg}")
        logger.error(tb)

results_df = pd.DataFrame(results)
results_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

success_count = (results_df["status"] == "success").sum()
failed_count = (results_df["status"] == "failed").sum()

logger.info("=" * 80)
logger.info("Batch run finished")
logger.info(f"Success: {success_count}")
logger.info(f"Failed : {failed_count}")
logger.info(f"Summary saved to: {summary_file}")
logger.info(f"Detailed log saved to: {log_file}")
logger.info("=" * 80)

print("All tasks completed!")
print(f"Detailed log: {log_file}")
print(f"Run summary: {summary_file}")