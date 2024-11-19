import glob
import os
import nibabel
from nilearn import masking
from nilearn.input_data import NiftiMasker
from nilearn import image
from nilearn import surface, plotting, datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import importlib
import utils
from nilearn.datasets import load_mni152_brain_mask
import scipy
from scipy.stats import norm
from nilearn.image import resample_img


# reload utils, useful if modifications were made in utils file
importlib.reload(utils)

# folder to store results
results_dir = "results"
figures_dir = "figures"
data_dir = "data"


# path to volume pipeline multiverse outputs
raw_data_path = os.path.join(data_dir, "multiverse_outputs")

#######
# GET PATH OF RAW DATA
#######
# store Z value raw maps
raw_z_value_maps = glob.glob(os.path.join(raw_data_path, "*_Z_MNI.nii.gz"))
raw_z_value_maps.sort()
# store p value raw maps
raw_p_value_maps = glob.glob(os.path.join(raw_data_path, "*_p.nii.gz"))
raw_p_value_maps.sort()
assert len(raw_z_value_maps) == len(raw_p_value_maps) == 3

# check size of maps and visualize maps
for i in range(3):
	print("Checking shape of files")
	print(raw_z_value_maps[i])
	print(nibabel.load(raw_z_value_maps[i]).get_fdata().mean())
	print(nibabel.load(raw_p_value_maps[i]).get_fdata().mean())
	assert nibabel.load(raw_z_value_maps[i]).get_fdata().shape == (91, 109, 91)
	assert nibabel.load(raw_p_value_maps[i]).get_fdata().shape == (91, 109, 91)


#######
# COMPUTE MASK
#######
# compute mask using the z value maps 
masks = []
print("Computing mask...")
for unthreshold_map in raw_z_value_maps:
	mask = masking.compute_background_mask(unthreshold_map)
	masks.append(mask)
multiverse_outputs_mask = masking.intersect_masks(masks, threshold=1, connected=False)
nibabel.save(multiverse_outputs_mask, os.path.join(data_dir, "masking", "multiverse_outputs_mask.nii"))

# load mask for inverse transform
masker = NiftiMasker(
    mask_img=multiverse_outputs_mask)

#######
# RESAMPLE RAW DATA WITH MASK
#######
# masking data with the created mask
def masking_raw_z_maps(raw_maps, mask):
	# computing mask
	resampled_data_path = os.path.join(data_dir, "mutliverse_outputs_resampled")
	resampled_maps = {} #storing the resampled maps
	for unthreshold_map in raw_maps:
		name = unthreshold_map.split('/')[-1].split('_')[0] + '_resampled_Z_MNI' 
		# resample MNI
		resampled_map = image.resample_to_img(
					unthreshold_map,
					mask,
					interpolation='nearest')
		resampled_maps[name] = resampled_map
		assert resampled_map.get_fdata().shape == multiverse_outputs_mask.get_fdata().shape
		resampled_map = None # emptying RAM memory
	# saving
	for key in resampled_maps.keys():
		nibabel.save(resampled_maps[key], os.path.join(data_dir, "multiverse_outputs_resampled", "{}.nii".format(key)))
	return resampled_maps

masked_z_maps = masking_raw_z_maps(raw_z_value_maps, multiverse_outputs_mask)
for key in masked_z_maps.keys(): # save data
	nibabel.save(masked_z_maps[key], os.path.join(data_dir, "multiverse_outputs_resampled", "{}.nii".format(key))),
masked_z_maps_flatten = masker.fit_transform(masked_z_maps.values())

# compute Z into p to check diff with SDMA outputs and 3 inputs significant values:
masked_p_maps_flatten = scipy.stats.norm.sf(masked_z_maps_flatten)
masked_significant_p_maps_flatten = masked_p_maps_flatten.copy()
masked_significant_p_maps_flatten[masked_significant_p_maps_flatten>0.05] = 0

CAT12_percentage_significance = len(masked_significant_p_maps_flatten[0][masked_significant_p_maps_flatten[0] != 0]) / len(masked_significant_p_maps_flatten[0])
FSLVBM_percentage_significance = len(masked_significant_p_maps_flatten[1][masked_significant_p_maps_flatten[1] != 0]) / len(masked_significant_p_maps_flatten[1])
FSLANAT_percentage_significance = len(masked_significant_p_maps_flatten[2][masked_significant_p_maps_flatten[2] != 0]) / len(masked_significant_p_maps_flatten[2])

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
plot_corr_matrix(masked_z_maps_flatten, masked_z_maps.keys(), 'correlation z values', os.path.join(figures_dir, 'correlation_z_values.png'))
# between p values per mask
plot_corr_matrix(masked_p_maps_flatten, masked_z_maps.keys(), 'correlation p values', os.path.join(figures_dir, 'correlation_p_values.png'))

#######
# COMPUTE SDMA STOUFFER
#######

SDMA_Stouffer_outputs = utils.SDMA_Stouffer(masked_z_maps_flatten)
# store results for SDMA Stouffer
SDMA_Stouffer_Zmap = SDMA_Stouffer_outputs[0]
SDMA_Stouffer_pmap = SDMA_Stouffer_outputs[1]
SDMA_Stouffer_significant_pmap = SDMA_Stouffer_pmap.copy()
SDMA_Stouffer_significant_pmap[SDMA_Stouffer_significant_pmap>0.05] = 0 # non corrected
SDMA_Stouffer_percentage_significance = len(SDMA_Stouffer_significant_pmap[SDMA_Stouffer_significant_pmap != 0]) / len(SDMA_Stouffer_significant_pmap)

#save results
nibabel.save(masker.inverse_transform(SDMA_Stouffer_Zmap), os.path.join(results_dir , "SDMA_Stouffer_Zmap.nii"))
nibabel.save(masker.inverse_transform(SDMA_Stouffer_pmap), os.path.join(results_dir , "SDMA_Stouffer_pmap.nii"))
nibabel.save(masker.inverse_transform(SDMA_Stouffer_significant_pmap), os.path.join(results_dir , "SDMA_Stouffer_significant_pmap.nii"))


#######
# COMPUTE SDMA GLS
#######
SDMA_GLS_outputs = utils.SDMA_GLS(masked_z_maps_flatten)
# store results for SDMA GLS
SDMA_GLS_Zmap = SDMA_GLS_outputs[0]
SDMA_GLS_pmap = SDMA_GLS_outputs[1]
SDMA_GLS_significant_pmap = SDMA_GLS_pmap.copy()
SDMA_GLS_significant_pmap[SDMA_GLS_significant_pmap>0.05] = 0 # non corrected
SDMA_GLS_percentage_significance = len(SDMA_GLS_significant_pmap[SDMA_GLS_significant_pmap != 0]) / len(SDMA_GLS_significant_pmap)
#save results
nibabel.save(masker.inverse_transform(SDMA_GLS_Zmap), os.path.join(results_dir , "SDMA_GLS_Zmap.nii"))
nibabel.save(masker.inverse_transform(SDMA_GLS_pmap), os.path.join(results_dir , "SDMA_GLS_pmap.nii"))
nibabel.save(masker.inverse_transform(SDMA_GLS_significant_pmap), os.path.join(results_dir , "SDMA_GLS_significant_pmap.nii"))

#######
# SAVE ORIGINAL ZVALUE AND PVALUES FOR EACH PIPELINE (to compare with SDMA results)
#######
CAT12_significant_p = masker.inverse_transform(masked_significant_p_maps_flatten[0])
FSLVBM_significant_p = masker.inverse_transform(masked_significant_p_maps_flatten[1])
FSLANAT_significant_p = masker.inverse_transform(masked_significant_p_maps_flatten[2])

nibabel.save(CAT12_significant_p, os.path.join(results_dir , "CAT12_significant_p.nii"))
nibabel.save(FSLVBM_significant_p, os.path.join(results_dir , "FSLVBM_significant_p.nii"))
nibabel.save(FSLANAT_significant_p, os.path.join(results_dir , "FSLANAT_significant_p.nii"))


# Define the second map list
map_list_column_1 = [
    os.path.join(data_dir, "multiverse_outputs_resampled", "CAT12_resampled_Z_MNI.nii"),
    os.path.join(data_dir, "multiverse_outputs_resampled", "FSLVBM_resampled_Z_MNI.nii"),
    os.path.join(data_dir, "multiverse_outputs_resampled", "FSLANAT_resampled_Z_MNI.nii"),
    masker.inverse_transform(SDMA_Stouffer_Zmap),
    masker.inverse_transform(SDMA_GLS_Zmap)
]

map_list_column_2 = [
    CAT12_significant_p,
    FSLVBM_significant_p,
    FSLANAT_significant_p,
    masker.inverse_transform(SDMA_Stouffer_significant_pmap),
    masker.inverse_transform(SDMA_GLS_significant_pmap)
]
map_names = ["CAT12", "FSLVBM", "FSLANAT", "|| SDMA Stouffer ||", "|| SDMA GLS ||"]
perc_sign_list = [CAT12_percentage_significance*100, FSLVBM_percentage_significance*100, FSLANAT_percentage_significance*100, SDMA_Stouffer_percentage_significance*100, SDMA_GLS_percentage_significance*100]

# # make MNI mask to get same shape and affine as ou mask (to allow to better visualisation)
# MNI_mask = load_mni152_brain_mask()
# # Resample the brain mask to match the shape and affine of B.nii
# MNI_mask_resampled = resample_img(MNI_mask, target_affine=multiverse_outputs_mask.affine, target_shape=multiverse_outputs_mask.shape)


# Create a figure for plotting with 5 rows and 2 columns
fig, axes = plt.subplots(5, 2, figsize=(10, 7))
# Loop through each map and plot
for i in range(len(map_list_column_1)):
    # Plot the first column map
    plotting.plot_stat_map(
        map_list_column_1[i],
        annotate=False,
        # bg_img=None,  # Set background to None for a white background
        vmin=0.00000000000001,
        vmax=5,
        cut_coords=(-34, -21, -13, -7, -1, 7, 20),
        colorbar=True,
        display_mode='z',
        cmap='Reds',
        axes=axes[i, 0]  # Specify the axes for the first column
    )
    # Set the title for the first column
    axes[i, 0].set_title(map_names[i])

    # Plot the second column map
    plotting.plot_stat_map(
        map_list_column_2[i],
        annotate=False,
        # bg_img=None,  # Set background to None for a white background
        vmin=0.00000000000001,
        vmax=0.1,
        cut_coords=(-34, -21, -13, -7, -1, 7, 20),
        colorbar=True,
        display_mode='z',
        cmap='Reds_r',
        axes=axes[i, 1]  # Specify the axes for the second column
    )
    # Set the title for the second column
    axes[i, 1].set_title(map_names[i] + " {}%".format(numpy.round(perc_sign_list[i], 2)))

# Save the combined plot as a single image
plt.savefig(os.path.join(figures_dir, "combined_results.png"), bbox_inches='tight', facecolor='white')

# Show the plots
plt.close('all')



