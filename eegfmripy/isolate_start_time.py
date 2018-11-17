import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import spatial
import mne as mne
from scipy import stats
import nibabel as nib
from RemoveGradient import get_slice_epochs, get_slice_sample_len
import sys
from scipy import signal

sys.path.insert(0, '/media/sf_shared/eegfmripy/eegfmripy/tests')


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

from TestsEEGFMRI import test_example_raw 
raw = test_example_raw()
graddata = raw.get_data(picks=[1])
graddata = np.squeeze(graddata)
# the initial onset can only be found by using information from the average gradient
# ie convolve with average gradient and compute offset? 
slice_len = get_slice_sample_len(graddata) # slice length in samples
slice_epochs, slice_inds = get_slice_epochs(graddata,slice_len)

m_epoch = np.mean(slice_epochs,axis=0)

ground_truth = 21210 

start_search_l = 1000000
conved = np.convolve((graddata[0:start_search_l]),m_epoch,mode='same')
stds = np.zeros(start_search_l)
for i in np.arange(0,start_search_l):
    stds[i] = np.std(graddata[i:i+slice_len])

from scipy.cluster.vq import vq, kmeans, whiten
 
k,d = kmeans(stds, 2)
classes = vq(stds,k)[0]
start_ind = np.max(np.where(classes==1))
 
end_grad = start_ind + n_slice_artifacts*slice_len

buffer = 50000
win_sz = int(slice_len)
std_arr = np.zeros(int(graddata.shape[0] - end_grad + buffer))
icount = 0
for i in np.arange(end_grad-buffer, graddata.shape[0]):
    std_arr[icount] = np.std(graddata[int(i)-win_sz:int(i)])
    icount = icount + 1
    
median_std = np.median(np.std(slice_epochs,axis=1)) 
std_thresh = median_std/4 

local_endpoint = np.min(np.where(std_arr < std_thresh))
global_endpoint = (end_grad - buffer) + local_endpoint

n_grad_samples = global_endpoint - start_ind 
estimated_nvols = n_grad_samples / (slice_len * n_slices)

estimated_ndummies = estimated_nvols - n_volumes
estimated_start = n_grad_samples - (slice_len * n_slices * n_volumes)


