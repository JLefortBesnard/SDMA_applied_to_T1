import glob
import os
import nibabel
from nilearn import masking
from nilearn.input_data import NiftiMasker
from nilearn import image
from nilearn import surface, plotting
from nilearn import surface, plotting, datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import importlib
import utils
from nilearn.datasets import load_mni152_brain_mask
import scipy
from scipy.stats import norm

# reload utils, useful if modifications were made in utils file
importlib.reload(utils)

# folder to store results
results_dir = "results"
figures_dir = "figures"
data_dir = "data"


# path to volume pipeline multiverse outputs
raw_data_path = os.path.join(data_dir, "multiverse_outputs")

#######
# store all Z values raw maps for analysis
#######
raw_z_value_maps = glob.glob(os.path.join(raw_data_path, "*_Z.nii.gz"))
raw_z_value_maps.sort()
raw_p_value_maps = glob.glob(os.path.join(raw_data_path, "*_p.nii.gz"))
raw_p_value_maps.sort()
assert len(raw_z_value_maps) == 3
assert len(raw_p_value_maps) == 3
# check size of maps and visualize maps
for i in range(1):
	print(raw_z_value_maps[i])
	assert nibabel.load(raw_z_value_maps[i]).get_fdata().shape == (91, 109, 91)
	assert nibabel.load(raw_p_value_maps[i]).get_fdata().shape == (91, 109, 91)

#######
# compute mask using the p value maps (if done with the z value maps, the R masking is different (8 voxels less))
#######
masks = []
print("Computing mask...")
for unthreshold_map in raw_z_value_maps:
	mask = masking.compute_background_mask(unthreshold_map)
	masks.append(mask)
multiverse_outputs_mask = masking.intersect_masks(masks, threshold=1, connected=False)
nibabel.save(multiverse_outputs_mask, os.path.join(data_dir, "masking", "multiverse_outputs_mask.nii"))

# # check mask computed using python is equivalent to mask computed using R
# mask_using_R = os.path.join(data_dir, "masking", "Mask_3Proc.nii.gz")
# assert nibabel.load(mask_using_R).get_fdata().sum() == multiverse_outputs_mask.get_fdata().sum()

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=multiverse_outputs_mask)

#######
# masking data with the created mask
#######

def masking_raw_maps(raw_maps, mask):
	# computing mask
	resampled_data_path = os.path.join(data_dir, "mutliverse_outputs_resampled")
	resampled_maps = {} #storing the resampled maps
	for unthreshold_map in raw_maps:
		name = unthreshold_map.split('/')[-1].split('_')[0] + '_resampled_' + unthreshold_map.split('/')[-1].split('_')[2][:-6]
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

resampled_z_maps = masking_raw_maps(raw_z_value_maps, multiverse_outputs_mask)


#######
# masking resampled data (to get K*J matrix)
#######
multiverse_outputs_matrix_z = masker.fit_transform(resampled_z_maps.values())

# compute Z into p to check diff with SDMA outputs and 3 inputs significant values:
multiverse_outputs_matrix_p = scipy.stats.norm.sf(multiverse_outputs_matrix_z)
multiverse_outputs_matrix_sign_p = multiverse_outputs_matrix_p.copy()
multiverse_outputs_matrix_sign_p[multiverse_outputs_matrix_sign_p>0.05] = 0

#######
# plot correlation matrix
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
	plt.show()

plot_corr_matrix(multiverse_outputs_matrix_z, resampled_z_maps.keys(), 'correlation z values', os.path.join(figures_dir, 'correlation_z_values.png'))

#######
# compute and plot SDMA results
#######

output_SDMA_p_from_z = utils.SDMA_Stouffer(multiverse_outputs_matrix_z)[1]
# turn every p > 0.05 equal to 0
output_SDMA_p_significant_from_z = output_SDMA_p_from_z.copy()
output_SDMA_p_significant_from_z[output_SDMA_p_significant_from_z > 0.05] = 0

# save as nii
output_SDMA_p_significant_from_z_nii = masker.inverse_transform(output_SDMA_p_significant_from_z)
CAT12_sign_p_originals = masker.inverse_transform(multiverse_outputs_matrix_sign_p[0])
FSLVBM_sign_p_originals = masker.inverse_transform(multiverse_outputs_matrix_sign_p[1])
FSLANAT_sign_p_originals = masker.inverse_transform(multiverse_outputs_matrix_sign_p[2])

nibabel.save(output_SDMA_p_significant_from_z_nii, os.path.join(results_dir , "SDMA_Stouffer_(z)_p_significant_outputs.nii"))
nibabel.save(CAT12_sign_p_originals, os.path.join(results_dir , "CAT12_sign_p_originals.nii"))
nibabel.save(FSLVBM_sign_p_originals, os.path.join(results_dir , "FSLVBM_sign_p_originals.nii"))
nibabel.save(FSLANAT_sign_p_originals, os.path.join(results_dir , "FSLANAT_sign_p_originals.nii"))




utils.plot_map(CAT12_sign_p_originals, multiverse_outputs_mask, os.path.join(figures_dir , "CAT12_sign_p_originals"))
utils.plot_map(FSLVBM_sign_p_originals, multiverse_outputs_mask, os.path.join(figures_dir , "FSLVBM_sign_p_originals"))
utils.plot_map(FSLANAT_sign_p_originals, multiverse_outputs_mask, os.path.join(figures_dir , "FSLANAT_sign_p_originals"))
utils.plot_map(output_SDMA_p_significant_from_z_nii, multiverse_outputs_mask, os.path.join(figures_dir , "SDMA_Stouffer_p_significant_outputs"))


