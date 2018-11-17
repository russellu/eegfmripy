import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import spatial
import mne as mne
from scipy import stats


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

TR = 1
n_slices = 40
n_volumes = 3000
srate = 5000
trigger_name = 'R1'











