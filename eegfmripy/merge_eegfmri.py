import mne as mne 
import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np 
import nibabel as nib
from scipy import signal
from nilearn.decomposition import CanICA
import sys as sys
sys.path.insert(0, '/media/sf_shared/eegfmripy/eegfmripy')
import helpers
"""
functions to merge ICA denoised EEG data with BOLD ICA components

"""

eeg_path = '/media/sf_shared/graddata/ica_denoised_raw.fif' 
fmri_path = '/media/sf_shared/CoRe_011/rfMRI/d2/11-BOLD_Sleep_BOLD_Sleep_20150824220820_11.nii'
vmrk_path = '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vmrk'
event_ids, event_lats = helpers.read_vmrk(vmrk_path)

event_lats = np.array(event_lats)
grad_inds = [index for index, value in enumerate(event_ids) if value == 'R1']
grad_inds = np.array(grad_inds)
grad_lats = event_lats[grad_inds]
grad_lats = grad_lats / 20 # resample from 5000Hz to 250Hz
start_ind = int(grad_lats[0])
end_ind = int(grad_lats[-1])

sleep_stages = ['wake','NREM1','NREM2','NREM3']

canica = CanICA(n_components=40, smoothing_fwhm=6.,
                threshold=None, verbose=10, random_state=0)

fmri = nib.load(fmri_path)
# get TR, n_slices, and n_TRs
fmri_info = helpers.fmri_info(fmri_path)

canica.fit(fmri)

comps = canica.transform([fmri])
cimg = canica.components_img_.get_data()

# plot components 
for i in np.arange(0,40):
    plt.subplot(4,10,i+1)
    plt.imshow(np.max(cimg[:,:,:,i],axis=2))


# get the EEG
    
raw = mne.io.read_raw_fif(eeg_path,preload=True)
raw_data = raw.get_data()

# get power spectrum for different sleep stages (EEG)
stage_power = np.zeros((4,63,129))
stage_count = 0 
for stage in sleep_stages:
    print(stage)
    stage_inds = [index for index, value in enumerate(event_ids) if value == stage]
    stage_lats = event_lats[stage_inds]
    lat_epochs = np.zeros((raw_data.shape[0],stage_lats.shape[0],7500))
    for lat in np.arange(0,lat_epochs.shape[1]):
        delete_last = False 
        if stage_lats[lat] + 7500 < raw_data.shape[1]:
            lat_epochs[:,lat,:] = raw_data[:, stage_lats[lat]:stage_lats[lat]+7500]        
        else:
            delete_last = True
     
    if delete_last == True:
        np.delete(lat_epochs, lat_epochs.shape[1] - 1, axis=1)
        
        
    pxx = np.zeros((63,lat_epochs.shape[1],129))
    for chan in np.arange(0,63):
        freqs, pxx[chan,:,:] = signal.welch(lat_epochs[chan,:,:],fs=250)
    
    stage_power[stage_count,:,:] = np.mean(pxx,axis=1)
    stage_count = stage_count + 1
    
    
    
# get power spectrum for different sleep stages (BOLD)


[f,pxx] = signal.welch(lat_epochs[9,:,:],fs=250)












