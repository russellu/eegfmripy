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


canica = CanICA(n_components=40, smoothing_fwhm=6.,
                threshold=None, verbose=10, random_state=0)

fmri = nib.load(fmri_path)
# get TR, n_slices, and n_TRs
fmri_info = helpers.fmri_info(fmri_path)
canica.fit(fmri)
cimg = canica.components_img_.get_data()

# plot components 
for i in np.arange(0,40):
    plt.subplot(4,10,i+1)
    plt.imshow(np.max(cimg[:,:,:,i],axis=2))

# get the EEG
raw = mne.io.read_raw_fif(eeg_path,preload=True)
raw_data = raw.get_data()
    
# get power spectrum for different sleep stages (BOLD)
comps = canica.transform([fmri])[0].transpose()
bold_srate = 1/fmri_info[0]
bold_epochl = int(7500 / (250/bold_srate))

#bold_pxx,bold_f = pxx_bold_component_epoch(comps, bold_srate, 250, bold_epochl, sleep_stages)
#eeg_pxx,eeg_f = pxx_eeg_epochs(raw_data, sleep_stages, 7500)

# concatenate the epochs, then compute the psd 
# 1) get triggers, 2) concatenate data, 3) compute psd 

def get_trigger_inds(trigger_name, event_ids):
    trig_inds = [index for index, value in enumerate(event_ids) if value == trigger_names[trig]]
    return trig_inds

def epoch_triggers(raw_data, lats, pre_samples, post_samples):
    epochs = np.zeros((raw_data.shape[0], lats.shape[0], pre_samples + post_samples))
    for lat in np.arange(0,lats.shape[0]):
        epochs[:,lat,:] = raw_data[:,lats[lat]-pre_samples : lats[lat]+post_samples] 

    return epochs

trigger_names = ['wake','NREM1','NREM2','NREM3']

"""
epoch BOLD and get power for different trigger types 
what you actually want is single trial EEG and BOLD psd

first get all the indices that are contained within the BOLD timeseries
then, get the EEG power spectrum values within those same indices 
"""
bold_pre_samples = 5
bold_post_samples = 15 
eeg_pre_samples = int(bold_pre_samples*fmri_info[0]*eeg_srate)
eeg_post_samples = int(bold_post_samples*fmri_info[0]*eeg_srate)
bold_conversion = eeg_srate / (1/fmri_info[0])

all_bold_epochs = []
for trig in np.arange(0, len(trigger_names)):
    trig_inds = get_trigger_inds(trigger_names[trig], event_ids)
    trig_lats = event_lats[trig_inds]
    bold_lats = ((trig_lats - start_ind) / bold_conversion).astype(int)    
    bads = np.where((bold_lats-bold_pre_samples<0) | (bold_lats+bold_post_samples>=comps.shape[1]))
    
    bold_lats = np.delete(bold_lats,bads,axis=0)
    eeg_lats = np.delete(trig_lats,bads,axis=0)
    
    bold_epochs = epoch_triggers(comps, bold_lats, bold_pre_samples, bold_post_samples)
    eeg_epochs = epoch_triggers(raw_data,eeg_lats,eeg_pre_samples, eeg_post_samples)

    all_bold_epochs.append(bold_epochs)

