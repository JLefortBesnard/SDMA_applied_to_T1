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
from nilearn.image import resample_img
import scipy.stats as stats


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
print("Computing mask... DONE")

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

print("Masking Z maps...")
masked_z_maps = masking_raw_z_maps(raw_z_value_maps, multiverse_outputs_mask)
for key in masked_z_maps.keys(): # save data
	nibabel.save(masked_z_maps[key], os.path.join(data_dir, "multiverse_outputs_resampled", "{}.nii".format(key))),
masked_z_maps_flatten = masker.fit_transform(masked_z_maps.values())
print("Masking Z maps... DONE")
# compute Z into p to check diff with SDMA outputs and 3 inputs significant values:
masked_p_maps_flatten = scipy.stats.norm.sf(masked_z_maps_flatten)
masked_significant_p_maps_flatten = masked_p_maps_flatten.copy()
masked_significant_p_maps_flatten[masked_significant_p_maps_flatten>0.05] = 0
# create significant map with z values where p is significant
masked_significant_z_maps_flatten = masked_z_maps_flatten.copy()
masked_significant_z_maps_flatten[masked_p_maps_flatten>0.05] = 0

CAT12_percentage_significance = len(masked_significant_p_maps_flatten[0][masked_significant_p_maps_flatten[0] != 0]) / len(masked_significant_p_maps_flatten[0])
FSLVBM_percentage_significance = len(masked_significant_p_maps_flatten[1][masked_significant_p_maps_flatten[1] != 0]) / len(masked_significant_p_maps_flatten[1])
FSLANAT_percentage_significance = len(masked_significant_p_maps_flatten[2][masked_significant_p_maps_flatten[2] != 0]) / len(masked_significant_p_maps_flatten[2])

#######
# PLOT CORRELATION MATRICES
#######

print("Plot correlation matrix...")
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

print("Plot correlation matrix... DONE")
print("Compute SDMA...")
#######
# COMPUTE SDMA STOUFFER
#######

SDMA_Stouffer_outputs = utils.SDMA_Stouffer(masked_z_maps_flatten)
# store results for SDMA Stouffer
SDMA_Stouffer_Zmap = SDMA_Stouffer_outputs[0]
SDMA_Stouffer_pmap = SDMA_Stouffer_outputs[1]
SDMA_Stouffer_significant_pmap = SDMA_Stouffer_pmap.copy()
SDMA_Stouffer_significant_pmap[SDMA_Stouffer_significant_pmap>0.05] = 0 # non corrected
# keep t value where p is significant, 0 elsewhere
SDMA_Stouffer_significant_zmap = SDMA_Stouffer_Zmap.copy()
SDMA_Stouffer_significant_zmap[SDMA_Stouffer_pmap>0.05] = 0 # non corrected

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
# keep t value where p is significant, 0 elsewhere
SDMA_GLS_significant_zmap = SDMA_GLS_Zmap.copy()
SDMA_GLS_significant_zmap[SDMA_GLS_pmap>0.05] = 0 # non corrected

SDMA_GLS_percentage_significance = len(SDMA_GLS_significant_pmap[SDMA_GLS_significant_pmap != 0]) / len(SDMA_GLS_significant_pmap)
#save results
nibabel.save(masker.inverse_transform(SDMA_GLS_Zmap), os.path.join(results_dir , "SDMA_GLS_Zmap.nii"))
nibabel.save(masker.inverse_transform(SDMA_GLS_pmap), os.path.join(results_dir , "SDMA_GLS_pmap.nii"))
nibabel.save(masker.inverse_transform(SDMA_GLS_significant_pmap), os.path.join(results_dir , "SDMA_GLS_significant_pmap.nii"))
print("Compute SDMA... DONE")

print("Save data...")
#######
# SAVE ORIGINAL ZVALUE AND PVALUES FOR EACH PIPELINE (to compare with SDMA results)
#######
CAT12_significant_p = masker.inverse_transform(masked_significant_z_maps_flatten[0])
FSLVBM_significant_p = masker.inverse_transform(masked_significant_z_maps_flatten[1])
FSLANAT_significant_p = masker.inverse_transform(masked_significant_z_maps_flatten[2])

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
    masker.inverse_transform(SDMA_Stouffer_significant_zmap),
    masker.inverse_transform(SDMA_GLS_significant_zmap)
]
map_names = ["CAT12", "FSLVBM", "FSLANAT", "|| SDMA Stouffer ||", "|| SDMA GLS ||"]
perc_sign_list = [CAT12_percentage_significance*100, FSLVBM_percentage_significance*100, FSLANAT_percentage_significance*100, SDMA_Stouffer_percentage_significance*100, SDMA_GLS_percentage_significance*100]

# # make MNI mask to get same shape and affine as ou mask (to allow to better visualisation)
# MNI_mask = load_mni152_brain_mask()
# # Resample the brain mask to match the shape and affine of B.nii
# MNI_mask_resampled = resample_img(MNI_mask, target_affine=multiverse_outputs_mask.affine, target_shape=multiverse_outputs_mask.shape)
print("Save data... DONE")


print("Plot combined results...")
# Create a figure for plotting with 5 rows and 2 columns
fig, axes = plt.subplots(5, 2, figsize=(18, 12))
# Loop through each map and plot
for i in range(len(map_list_column_1)):
    # Plot the first column map
    plotting.plot_stat_map(
        map_list_column_1[i],
        annotate=False,
        # bg_img=None,  # Set background to None for a white background
        vmin=-8,
        vmax=8,
        cut_coords=(-34, -21, -13, -7, -1, 7, 20),
        colorbar=True,
        display_mode='z',
        cmap='coolwarm',
        axes=axes[i, 0]  # Specify the axes for the first column
    )
    # Set the title for the first column
    axes[i, 0].set_title(map_names[i])

    # Plot the second column map
    plotting.plot_stat_map(
        map_list_column_2[i],
        annotate=False,
        # bg_img=None,  # Set background to None for a white background
        vmin=-8,
        vmax=8,
        cut_coords=(-34, -21, -13, -7, -1, 7, 20),
        colorbar=True,
        display_mode='z',
        cmap='coolwarm',
        axes=axes[i, 1]  # Specify the axes for the second column
    )
    # Set the title for the second column
    axes[i, 1].set_title(map_names[i] + " {}%".format(numpy.round(perc_sign_list[i], 2)))

# Save the combined plot as a single image
plt.savefig(os.path.join(figures_dir, "combined_results.png"), bbox_inches='tight', facecolor='white')
# Show the plots
plt.close('all')
print("Plot combined results...DONE")


# QQ and PP PLOTs
print("Plot QQ and PP plots...")
pmaps_flatten = numpy.vstack([masked_p_maps_flatten, SDMA_Stouffer_pmap, SDMA_GLS_pmap])
pmaps_flatten_sorted = numpy.sort(pmaps_flatten)
# Generate the theoretical uniform quantiles (0 to 1)
uniform_quantiles = numpy.linspace(0, 1, len(pmaps_flatten[0,:]))
labels = ['CAT12', 'FSLVBM', 'FSLANAT', 'SDMA Stouffer', 'SDMA GLS']
colors = ['#1D5FA1', '#4B8BBE', '#7A9CBB',  # Shades of blue/teal
          '#FF6F61', '#FFB6C1']  # Shades of red/pink
# Generate the theoretical uniform quantiles (0 to 1)
uniform_quantiles = numpy.linspace(0, 1, len(pmaps_flatten[0, :]))
# Labels and colors for the PP plot
labels = ['CAT12', 'FSLVBM', 'FSLANAT', 'SDMA Stouffer', 'SDMA GLS']
colors = ['#A9D0F5', '#4A90E2', '#003366',  # Shades of blue
          '#FF6F61', '#FFB6C1']  # Shades of red/pink

# Create a new figure for the QQ plot
fig, pp_ax = plt.subplots(figsize=(10, 8))
# Plot the QQ plots for all datasets on the same axis
for i in range(5):
    pp_ax.plot(uniform_quantiles, pmaps_flatten_sorted[i, :], label=labels[i], color=colors[i])
# Plot the ideal uniform distribution (diagonal line)
pp_ax.plot(uniform_quantiles, uniform_quantiles, linestyle='--', color='green', label='Ideal Uniform CDF')
# Set QQ plot labels and title
pp_ax.set_xlabel('Theoretical Quantiles (Uniform)')
pp_ax.set_ylabel('Empirical Quantiles (Sorted p-values)')
pp_ax.set_title('QQ Plot')
pp_ax.legend()
pp_ax.grid(True)
# Adjust layout for better spacing
plt.tight_layout()
# Save the QQ plot as a separate image
plt.savefig(os.path.join(figures_dir, "QQ_plot.png"), bbox_inches='tight', facecolor='white')
# Close the plot after saving
plt.close('all')
print("QQ Plot saved...DONE")

# Create a new figure for the PP plot

def distribution_inversed(J):
    distribution_inversed = []
    for i in range(J):
        distribution_inversed.append(i/(J+1))
    return distribution_inversed     

def minusLog10me(values):
    # prevent log10(0)
    return numpy.array([-numpy.log10(i) if i != 0 else 5 for i in values])

J = pmaps_flatten.shape[1]
K = pmaps_flatten.shape[0]
pmaps_flatten_sorted
p_cum = distribution_inversed(J)
x_lim_pplot =  5 #-numpy.log10(1/J)

plt.close('all')
# Create a new figure for the QQ plot
fig, pp_ax = plt.subplots(figsize=(10, 8))

label_pp = ["{}, {}%".format(label, numpy.round(perc_sign_list[ind], 2)) for ind, label in enumerate(labels)]

for col, title in enumerate(label_pp):
    p_obs_p_cum = minusLog10me(pmaps_flatten_sorted[col]) - minusLog10me(p_cum)
    pp_ax.plot(minusLog10me(p_cum), p_obs_p_cum, label=title, color=colors[col])

pp_ax.set_xlabel("-log10 cum p")
pp_ax.set_ylabel("obs p - expt p")
pp_ax.axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
pp_ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
pp_ax.set_xlim(0, x_lim_pplot)
pp_ax.set_ylim(-0.25, 22)
pp_ax.set_title('PP Plots')
pp_ax.legend()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False)     # ticks along the bottom edge are off) 
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "PP_plot_combined.png"))
plt.close('all')
print("PP Plot combined...DONE")

f, axs = plt.subplots(1, len(labels), figsize=(len(labels)*2.5, 8), sharey=True,sharex=True) 
for col, title in enumerate(labels):
    p_obs_p_cum = minusLog10me(pmaps_flatten_sorted[col]) - minusLog10me(p_cum)

    axs[col].title.set_text(title)
    axs[col].title.set_fontsize(15)
    axs[col].set_xlabel("-log10 cum p", fontsize=15)
    axs[col].plot(minusLog10me(p_cum), p_obs_p_cum, color='y')
    if col == 0:
        axs[col].set_ylabel("obs p - expt p", fontsize=15)
    else:
        axs[col].set_ylabel("")
    axs[col].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
    axs[col].axhline(0, color='black', linewidth=0.5, linestyle='--')

    axs[col].set_xlim(0, x_lim_pplot)
    axs[col].set_ylim(-0.25, 22)
    color= 'blue'
    axs[col].text(2.5, 0.7, '{}%'.format(numpy.round(perc_sign_list[col], 2)), color=color, fontsize=9)

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False)     # ticks along the bottom edge are off) 

# plt.savefig("{}/pp_plot_OHBM_ABSTRACT.png".format(results_dir))
plt.suptitle("PP plot", fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "PP_plot.png"))
plt.close('all')
print("PP Plot...DONE")