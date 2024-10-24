import glob, os
import numpy
import nibabel
from nilearn import masking
from nilearn.input_data import NiftiMasker
import nibabel
from nilearn import image

# folder to store results
results_dir = "results"
figures_dir = "figures"
data_dir = "data"


# path to volume pipeline multiverse outputs
raw_data_path = os.path.join(data_dir, "multiverse_outputs")


# compute mask
masks = []
print("Computing mask...")
for unthreshold_map in glob.glob(raw_data_path):
	mask = masking.compute_background_mask(unthreshold_map)
	masks.append(mask)
multiverse_outputs_mask = masking.intersect_masks(masks, threshold=1, connected=True)
nibabel.save(multiverse_outputs_mask, os.path.join(data_dir, "masking", "multiverse_outputs_mask.nii"))

# path to participants mask
print("Saving mask...")
multiverse_outputs_mask_path = os.path.join(data_dir, "masking", "multiverse_outputs_mask.nii")
multiverse_outputs_mask = nibabel.load(multiverse_outputs_mask_path)

# save mask for inverse transform
masker = NiftiMasker(
    mask_img=multiverse_outputs_mask)

print("Masking raw data...")
# path to volume pipeline multiverse outputs resampled with the multiverse_outputs_mask
resampled_data_path = os.path.join(data_dir, "mutliverse_outputs_resampled")

for unthreshold_map in glob.glob(raw_data_path):
	# resample MNI
	resampled_map = image.resample_to_img(
				unthreshold_map,
				multiverse_outputs_mask,
				interpolation='nearest')

