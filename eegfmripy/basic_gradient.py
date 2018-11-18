import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mne as mne
from scipy import stats
import sys as sys
from mne.preprocessing import ICA

"""
basic_gradient.py

epoching and average artifact subtraction of gradient artifact 
for single channel EEG time series

requires the following arguments: 

1) repetition time (TR) in seconds
2) # of slices per imaging volume
3) # of volumes over the entirety of the scan
4) sampling rate of EEG data (default is 5000Hz)

5) ONE of the following:
    a) the sample point index of onset of the 1st slice of the 1st vol
    b) the name of the trigger marking the onset of 1st slice of 1st vol
    c) the number of dummy scans, so that it can find the 1st vol onset time
    
given the above information, the function performs the following:
1) epoching of gradient slice artifacts    
2) average artifact subtraction based on 
    a) sliding window averaging (default) or b) ssd-based averaging
3) zeroing of time points outside the FMRI acquisition window
4) return of gradient-denoised, and optionally downsampled time-series    
    
the command also accepts the path to the corresponding nifti, which 
can be given instead of TR, # slices, and # volumes

examples:
    
when both FMRI nifti is available and gradient triggers are present: 
    
basic_gradient(eeg_data_path, nifti_data_path, trigger_name='R1') 

if the FMRI nifti is not available:

basic_gradient(eeg_data_path, TR=1, n_slices=40, n_volumes=3000,
               trigger_name='R1') 

if the gradient triggers are not present but the first slice time is known:

basic_gradient(eeg_data_path, nifti_data_path, first_slice_sample=21000)

if neither gradient triggers or first slice time is present:
    
basic_gradient(eeg_data_path, nifti_data_path, n_dummies=3)
    
"""

"""

basic_gradient_removal: 
    
    remove gradient artifacts from EEG time series in the
    most basic possible way: volume by volume, using slide window average 
    artifact subtraction. 
    
    can also interpolate amplifier saturation points

"""
def basic_gradient_removal(data, grad_lats, n_volumes, tr_samples):

    print('removing gradient')
    
    vol_epochs = np.zeros((n_volumes,tr_samples))
    vol_inds = np.zeros((n_volumes,tr_samples))
    for i in np.arange(0, n_volumes):
        vol_epochs[i,:] = data[grad_lats[i]:grad_lats[i] + tr_samples]
        vol_inds[i,:] = np.arange(grad_lats[i], grad_lats[i] + tr_samples)
    
    vol_inds = vol_inds.astype(int)
    
    window_sz=30
    window_std=9
    
    window = signal.gaussian(window_sz, std=window_std)
    window = np.reshape(window / np.sum(window),[window.shape[0],1])   
    
    smooth_epochs = np.zeros(vol_epochs.shape)        
    smooth_epochs[:,:] = signal.convolve2d(vol_epochs, 
                 window, boundary='symm', mode='same')   
         
    subbed_epochs = vol_epochs - smooth_epochs
    residuals = np.sum(np.abs(np.diff(subbed_epochs,axis=1)),axis=1)
    
    
    new_ts = np.zeros(data.shape)
    resid_ts = np.zeros(data.shape)
    for i in np.arange(0,subbed_epochs.shape[0]):
        max_i = np.max(vol_epochs[i,:])
        new_ts[vol_inds[i,:]] = subbed_epochs[i,:]
        max_inds = vol_inds[i,np.where(vol_epochs[i,:]==max_i)[0]]
        
        # interpolate the saturation points 
        for j in np.arange(0,max_inds.shape[0]):
            new_ts[max_inds[j]] = new_ts[max_inds[j]-2]/2 + new_ts[max_inds[j]+2]/2
         
        resid_ts[vol_inds[i]] = residuals[i]
            
            
    return new_ts, resid_ts




TR = 1
n_slices = 40
n_volumes = 3000
srate = 5000
trigger_name = 'R1'

sys.path.insert(0, '/media/sf_shared/eegfmripy/eegfmripy/tests')
from tests import test_example_raw

raw, montage = test_example_raw()

graddata = raw.get_data()

import helpers 
event_ids, event_lats = helpers.read_vmrk(
        '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vmrk')

event_lats = np.array(event_lats)
grad_inds = [index for index, value in enumerate(event_ids) if value == 'R1']
grad_inds = np.array(grad_inds)
grad_lats = event_lats[grad_inds]

tr, n_slices, n_volumes = helpers.fmri_info(
'/media/sf_shared/CoRe_011/rfMRI/d2/11-BOLD_Sleep_BOLD_Sleep_20150824220820_11.nii')

tr_samples = int(tr*5000)
residuals = np.zeros(graddata[0,:].shape)

for i in np.arange(0,graddata.shape[0]-1):
    graddata[i,:], resid = basic_gradient_removal(
            graddata[i,:], grad_lats, n_volumes, tr_samples)
    residuals = residuals + resid/(graddata.shape[0]-1)

dec = int(5000/250) 
downsampled = np.zeros((graddata.shape[0]-1, int(graddata.shape[1]/dec)))

dec_residuals = signal.decimate(residuals,dec) 
for i in np.arange(0,graddata.shape[0]-1):
    downsampled[i,:] = signal.decimate(graddata[i,:], dec)
    print(i)

ch_names, ch_types, inds = helpers.prepare_raw_channel_info(
        downsampled, raw, montage)


new_raw = helpers.create_raw_mne(downsampled[inds,:], ch_names, ch_types, montage)
new_raw.filter(1,120)
ica = ICA(n_components=60, method='fastica', random_state=23)
ica.fit(new_raw)
ica.plot_components(picks=np.arange(0,60))
src = ica.get_sources(new_raw).get_data()



