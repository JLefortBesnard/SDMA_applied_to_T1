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
for i in range(3):
	assert nibabel.load(raw_z_value_maps[i]).get_fdata().shape == (91, 109, 91)
	assert nibabel.load(raw_p_value_maps[i]).get_fdata().shape == (91, 109, 91)
	plotting.plot_stat_map(raw_p_value_maps[i], bg_img =raw_z_value_maps[i])
	plotting.plot_stat_map(raw_z_value_maps[i], bg_img =raw_z_value_maps[i])
	plt.show()
stop




#######
# compute mask using the p value maps (if done with the z value maps, the R masking is different (8 voxels less))
#######
masks = []
print("Computing mask...")
for unthreshold_map in raw_p_value_maps:
	mask = masking.compute_background_mask(unthreshold_map)
	masks.append(mask)
multiverse_outputs_mask = masking.intersect_masks(masks, threshold=1, connected=False)
nibabel.save(multiverse_outputs_mask, os.path.join(data_dir, "masking", "multiverse_outputs_mask.nii"))

# check mask computed using python is equivalent to mask computed using R
mask_using_R = os.path.join(data_dir, "masking", "Mask_3Proc.nii.gz")
assert nibabel.load(mask_using_R).get_fdata().sum() == multiverse_outputs_mask.get_fdata().sum()

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
resampled_p_maps = masking_raw_maps(raw_p_value_maps, multiverse_outputs_mask)

#######
# masking resampled data (to get K*J matrix)
#######
multiverse_outputs_matrix_z = masker.fit_transform(resampled_z_maps.values())
multiverse_outputs_matrix_z = multiverse_outputs_matrix_z/2 # one tailed to two-tailed
multiverse_outputs_matrix_p = masker.fit_transform(resampled_p_maps.values())


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

plot_corr_matrix(multiverse_outputs_matrix_z, resampled_z_maps.keys(), 'correlation z values', os.path.join(figures_dir, 'correlation_z_values.png'))
plot_corr_matrix(multiverse_outputs_matrix_p, resampled_p_maps.keys(), 'correlation p values', os.path.join(figures_dir, 'correlation_p_values.png'))
multiverse_outputs_matrix_p_to_z = utils.p_value_to_z_matrix(multiverse_outputs_matrix_p , tail='two-tailed')
plot_corr_matrix(multiverse_outputs_matrix_p_to_z, resampled_p_maps.keys(), 'correlation p to Z values', os.path.join(figures_dir, 'correlation_p_to_Z_values.png'))

#######
# compute and plot SDMA results
#######


output_SDMA_p_from_p_to_z = utils.SDMA_Stouffer(multiverse_outputs_matrix_p_to_z)[1]
# turn every p > 0.05 equal to 0
output_SDMA_p_significant_from_p_to_z = output_SDMA_p_from_p_to_z.copy()
output_SDMA_p_significant_from_p_to_z[output_SDMA_p_significant_from_p_to_z > 0.05] = 0

output_SDMA_p_from_z = utils.SDMA_Stouffer(multiverse_outputs_matrix_z)[1]
# turn every p > 0.05 equal to 0
output_SDMA_p_significant_from_z = output_SDMA_p_from_z.copy()
output_SDMA_p_significant_from_z[output_SDMA_p_significant_from_z > 0.05] = 0

# save as nii
output_SDMA_p_significant_from_z_nii = masker.inverse_transform(output_SDMA_p_significant_from_z)
nibabel.save(output_SDMA_p_significant_from_z_nii, os.path.join(results_dir , "SDMA_Stouffer_(z)_p_significant_outputs.nii"))
output_SDMA_p_significant_from_p_to_z_nii = masker.inverse_transform(output_SDMA_p_significant_from_p_to_z)
nibabel.save(output_SDMA_p_significant_from_p_to_z_nii, os.path.join(results_dir , "SDMA_Stouffer_(p_to_z)_p_significant_outputs.nii"))

utils.plot_map(output_SDMA_p_significant_from_p_to_z_nii, multiverse_outputs_mask, os.path.join(figures_dir , "SDMA_Stouffer_(p_to_z)_p_significant_outputs"))
utils.plot_map(output_SDMA_p_significant_from_z_nii, multiverse_outputs_mask, os.path.join(figures_dir , "SDMA_Stouffer_p_significant_outputs"))
