import squidpy as sq
import scanpy as sc
import spatialdata as sd
from spatialdata_io import xenium
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import pandas as pd
import tifffile as tiff
import skimage
from skimage.draw import polygon
import numpy as np
import tifffile

adata=sc.read_h5ad('Xenium_Prime_Human_Prostate_FFPE_outs.h5ad')
cell_boundaries = pd.read_csv('/Xenium_Prime_Human_Prostate_FFPE_outs/cell_boundaries.csv')
file_path = '/Xenium_Prime_Human_Prostate_FFPE_outs/morphology_focus/morphology_focus_0001.ome.tif'
with tiff.TiffFile(file_path) as tif:
    pages = tif.pages
    image_0001 = pages[0].asarray()
file_path = '/Xenium_Prime_Human_Prostate_FFPE_outs/morphology_focus/morphology_focus_0003.ome.tif'
with tiff.TiffFile(file_path) as tif:
    pages = tif.pages
    image_0003 = pages[0].asarray()

cell_boundaries['pixel_x'] = (cell_boundaries['vertex_x'] / 0.2125).round().astype(int)
cell_boundaries['pixel_y'] = (cell_boundaries['vertex_y'] / 0.2125).round().astype(int)

results = []
for label in cell_boundaries['label_id'].unique():
    filtered_data = cell_boundaries[cell_boundaries['label_id'] == label]
    r = filtered_data['pixel_y'].to_numpy()  
    c = filtered_data['pixel_x'].to_numpy()  
    mask = skimage.draw.polygon(r, c)
    polygon_pixels = image_0001[mask]
    mean_gray_value = np.mean(polygon_pixels)
    max_gray_value = np.max(polygon_pixels)
    results.append({
        'cell_labels': label,
        'Mean_0001': mean_gray_value,
        'Max_0001': max_gray_value
    })
results_df_0001 = pd.DataFrame(results)

results = []
for label in cell_boundaries['label_id'].unique():
    filtered_data = cell_boundaries[cell_boundaries['label_id'] == label]
    r = filtered_data['pixel_y'].to_numpy()  
    c = filtered_data['pixel_x'].to_numpy() 
    mask = skimage.draw.polygon(r, c)
    polygon_pixels = image_0003[mask]
    mean_gray_value = np.mean(polygon_pixels)
    max_gray_value = np.max(polygon_pixels)
    results.append({
        'cell_labels': label,
        'Mean_0003': mean_gray_value,
        'Max_0003': max_gray_value
    })
results_df_0003 = pd.DataFrame(results)
protein = pd.merge(results_df_0001, results_df_0003, on='cell_labels', how='inner')