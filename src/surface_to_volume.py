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
import nibabel as nib

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

 # visualize

surface_img = nib.load(surface_file_paths[0])
surface_data = surface_img.get_fdata()
