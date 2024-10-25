import scipy
from scipy.stats import norm
import numpy
from nilearn import plotting
import matplotlib.pyplot as plt
import nibabel
import os

def SDMA_Stouffer(multiverse_outputs_matrix):
    K = multiverse_outputs_matrix.shape[0]
    ones = numpy.ones((K, 1))
    Q = numpy.corrcoef(multiverse_outputs_matrix)
    attenuated_variance = ones.T.dot(Q).dot(ones) / K**2
    # compute meta-analytic statistics
    T_map = numpy.mean(multiverse_outputs_matrix, 0)/numpy.sqrt(attenuated_variance)
    T_map = T_map.reshape(-1)
    # compute p-values for inference
    # p_values = 1 - scipy.stats.norm.cdf(T_map)
    p_values = scipy.stats.norm.sf(T_map)
    p_values = p_values.reshape(-1)
    return T_map, p_values

def p_value_to_z_matrix(p_values, tail='two-tailed'):
    # Ensure the input is a NumPy array
    p_values = numpy.array(p_values)
    if tail == 'two-tailed':
        z_scores = norm.sf(p_values / 2)
    elif tail == 'one-tailed':
        z_scores = norm.sf(p_values)
    else:
        raise ValueError("Tail must be 'one-tailed' or 'two-tailed'")
    return z_scores

def plot_map(map_to_plot, bg_mask, saving_name):
    plotting.plot_stat_map(
        map_to_plot,
        annotate=False, 
        bg_img =bg_mask,
        vmin=0.00000000000001,
        vmax = 1,
        cut_coords=(-34, -21, -13, -7, -1, 7, 20),
        colorbar=True,
        display_mode='z',
        cmap='Reds_r'
    )   
    plt.tight_layout()
    plt.savefig("{}.png".format(saving_name))
    plt.show()
    plt.close('all')
