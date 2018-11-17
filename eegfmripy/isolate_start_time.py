import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import spatial
import mne as mne
from scipy import stats
import nibabel as nib
from RemoveGradient import get_slice_epochs, get_slice_sample_len

"""
function: find the initial volume timing in the EEG signal, given the 
following parameters: TR, # of slices, multiband factor, # of dummy volumes

if these parameters are not known, you can also input the corresponding FMRI
image, and the parameters will be obtained from nibabel (you do this as a check)
if the number of dummies is not known, it is estimated based on the correlation
structure of the intitial volumes 
"""

dat0 = np.load('/media/sf_shared/graddata/graddata_0.npy')
dat0 = dat0[0:10000000]

fmri = nib.load(
        '/media/sf_shared/CoRe_011/rfMRI/d2/11-BOLD_Sleep_BOLD_Sleep_20150824220820_11.nii')
hdr = fmri.get_header()


TR = hdr.get_zooms()[3] 
n_slices = hdr.get_n_slices()
n_volumes = hdr.get_data_shape()[3]
mb_factor = 1
n_dummies = 3 

n_slice_artifacts = (n_volumes * n_slices) / mb_factor

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_shared/')
raw = mne.io.read_raw_brainvision(
        '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vhdr',
        montage=montage,eog=['ECG','ECG1'])

graddata = raw.get_data(picks=[1])

# the initial onset can only be found by using information from the average gradient
# ie convolve with average gradient and compute offset? 
slice_len = get_slice_sample_len(graddata) # slice length in samples
slice_epochs, slice_inds = get_slice_epochs(graddata,slice_len)






