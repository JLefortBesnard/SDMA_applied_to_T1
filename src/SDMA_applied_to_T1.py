import glob
import os
import nibabel
from nilearn import masking
from nilearn.input_data import NiftiMasker
from nilearn import image
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import importlib
import utils
from nilearn.datasets import load_mni152_brain_mask

importlib.reload(utils)

# folder to store results
results_dir = "results"
figures_dir = "figures"
data_dir = "data"


# path to volume pipeline multiverse outputs
raw_data_path = os.path.join(data_dir, "multiverse_outputs")


# compute mask
masks = []
print("Computing mask...")
for unthreshold_map in glob.glob(os.path.join(raw_data_path, "*")):
	mask = masking.compute_background_mask(unthreshold_map)
	resampled_mask = image.resample_to_img(
		mask,
		load_mni152_brain_mask(),
		interpolation='nearest')
	masks.append(resampled_mask)
multiverse_outputs_mask = masking.intersect_masks(masks, threshold=1, connected=False)
nibabel.save(multiverse_outputs_mask, os.path.join(data_dir, "masking", "multiverse_outputs_mask.nii"))

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=multiverse_outputs_mask)

print("Masking raw data...")
# path to volume pipeline multiverse outputs resampled with the multiverse_outputs_mask
resampled_data_path = os.path.join(data_dir, "mutliverse_outputs_resampled")

resampled_maps = {}
for unthreshold_map in glob.glob(os.path.join(raw_data_path, "*")):
	name = unthreshold_map.split('/')[-1][:-15]
	# resample MNI

	resampled_map = image.resample_to_img(
				unthreshold_map,
				multiverse_outputs_mask,
				interpolation='nearest')
	resampled_maps[name] = resampled_map
	assert resampled_map.get_fdata().shape == multiverse_outputs_mask.get_fdata().shape
	resampled_map = None # emptying RAM memory
for key in resampled_maps.keys():
	nibabel.save(resampled_maps[key], os.path.join(data_dir, "multiverse_outputs_resampled", "{}.nii".format(key)))



# masking resampled data (to get K*J matrix)
multiverse_outputs_matrix_p = masker.fit_transform(resampled_maps.values())


def plot_corr_matrix(data, names, title):
	Q = numpy.corrcoef(data)
	# Create a heatmap of the correlation matrix
	plt.figure(figsize=(8, 6))
	sns.heatmap(Q, annot=True, cmap='coolwarm', xticklabels=names, yticklabels=names, cbar=True)

	# Set titles and labels
	plt.title(title)
	plt.xlabel('Conditions')
	plt.ylabel('Conditions')

	# Show the plot
	plt.tight_layout()
	plt.show()

plot_corr_matrix(multiverse_outputs_matrix_p, resampled_maps.keys(), 'correlation p values')

multiverse_outputs_matrix_Z = utils.p_value_to_z_matrix(multiverse_outputs_matrix_p , tail='one-tailed')
plot_corr_matrix(multiverse_outputs_matrix_Z, resampled_maps.keys(), 'correlation Z values')

SDMA_Stouffer_outputs = utils.SDMA_Stouffer(multiverse_outputs_matrix_Z)
SDMA_Stouffer_results = SDMA_Stouffer_outputs[1]

SDMA_Stouffer_results_nii = masker.inverse_transform(SDMA_Stouffer_results)
nibabel.save(SDMA_Stouffer_results_nii, os.path.join(results_dir , "SDMA_Stouffer_results.nii"))

plotting.plot_stat_map(
    SDMA_Stouffer_results_nii,
    annotate=False,
    threshold=0.0001,
    # vmax=8,
    colorbar=True,
    cut_coords=(-24, -10, 4, 18, 32, 52),
    display_mode='z',
    cmap='Reds'
)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "{}.png".format("SDMA_Stouffer_outputs")))
plt.close('all')
print("Done plotting")
