import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import importlib
import utils
import scipy
from scipy.stats import norm
import pandas as pd

# folder to store results
results_dir = "results"
figures_dir = "figures"
data_dir = "data"

# path to volume pipeline multiverse outputs
raw_data_path = os.path.join(data_dir, "multiverse_outputs")

#######
# GET PATH OF RAW DATA
#######
# load .linear files
surface_file_paths = glob.glob(os.path.join(raw_data_path, "*.linear"))
surface_file_paths.sort()

# create Z maps of the volumes:
z_maps_flatten = []
z_maps_names = [surface_file_paths[0].split('/')[-1][:-7], surface_file_paths[1].split('/')[-1][:-7]] # remove '.linear' from name

# load surface Z values
for i, file_path in enumerate(surface_file_paths):
	print(file_path)
	df = pd.read_csv(file_path, delimiter="\s+", header=0)
	z_maps_flatten.append(df['Z'].values)
z_maps_flatten = numpy.array(z_maps_flatten)
#######
# PLOT CORRELATION MATRICES
#######
def plot_corr_matrix(data, names, title, saving_path):
	Q = numpy.corrcoef(data)
	# Create a heatmap of the correlation matrix
	plt.figure(figsize=(8, 6))
	sns.heatmap(Q, annot=True, cmap='coolwarm', xticklabels=names, yticklabels=names, cbar=True)

	# Set titles and labels
	plt.title(title)
	plt.xlabel('Conditions')
	plt.xticks(rotation=70)
	plt.ylabel('Conditions')

	# Show the plot
	plt.tight_layout()
	plt.savefig(saving_path)
	plt.close('all')
# between Z values per map
plot_corr_matrix(z_maps_flatten, z_maps_names, 'correlation surface z values', os.path.join(figures_dir, 'correlation_surface_z_values.png'))

#######
# COMPUTE SDMA STOUFFER
#######

SDMA_Stouffer_outputs = utils.SDMA_Stouffer(z_maps_flatten)
# store results for SDMA Stouffer
SDMA_Stouffer_Zmap = SDMA_Stouffer_outputs[0]
SDMA_Stouffer_pmap = SDMA_Stouffer_outputs[1]
SDMA_Stouffer_significant_pmap = SDMA_Stouffer_pmap.copy()
SDMA_Stouffer_significant_pmap[SDMA_Stouffer_significant_pmap>0.05] = 0 # non corrected
SDMA_Stouffer_percentage_significance = len(SDMA_Stouffer_significant_pmap[SDMA_Stouffer_significant_pmap != 0]) / len(SDMA_Stouffer_significant_pmap)

#######
# COMPUTE SDMA GLS
#######
SDMA_GLS_outputs = utils.SDMA_GLS(z_maps_flatten)
# store results for SDMA GLS
SDMA_GLS_Zmap = SDMA_GLS_outputs[0]
SDMA_GLS_pmap = SDMA_GLS_outputs[1]
SDMA_GLS_significant_pmap = SDMA_GLS_pmap.copy()
SDMA_GLS_significant_pmap[SDMA_GLS_significant_pmap>0.05] = 0 # non corrected
SDMA_GLS_percentage_significance = len(SDMA_GLS_significant_pmap[SDMA_GLS_significant_pmap != 0]) / len(SDMA_GLS_significant_pmap)

# save results in DF
df_SDMA_results = pd.DataFrame(columns=['SDMA_Stouffer_Z', 'SDMA_Stouffer_p', 'SDMA_Stouffer_significant_p', 'SDMA_GLS_Z', 'SDMA_GLS_p', 'SDMA_GLS_significant_p'])
df_SDMA_results['SDMA_Stouffer_Z'] = SDMA_Stouffer_Zmap
df_SDMA_results['SDMA_Stouffer_p'] = SDMA_Stouffer_pmap
df_SDMA_results['SDMA_Stouffer_significant_p'] = SDMA_Stouffer_significant_pmap
df_SDMA_results['SDMA_GLS_Z'] = SDMA_GLS_Zmap
df_SDMA_results['SDMA_GLS_p'] = SDMA_GLS_pmap
df_SDMA_results['SDMA_GLS_significant_p'] = SDMA_GLS_significant_pmap


df_SDMA_results.to_csv(os.path.join(results_dir, "results_SDMA_surface.linear"), sep=' ', index=False)