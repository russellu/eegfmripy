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


"""
epoch BOLD components based on triggers and compute power spectral density
for each trigger type and average across all epochs

returns an array of [trigger, component, power density] and the
frequency axis of the power density 

"""
def pxx_bold_component_epoch(bold_comps, bold_srate, 
                             eeg_srate, bold_epochl, triggers): 
    
    bold_pxx = np.zeros((4,bold_comps.shape[1],7))
    stage_count = 0 
    for stage in triggers:
        print(stage)
        stage_inds = [index for index, value in enumerate(event_ids) if value == stage]
        stage_lats = event_lats[stage_inds] - start_ind 
        neg_inds = np.where(stage_lats < 0)
        stage_lats = np.delete(stage_lats,neg_inds)
        tr_lats = stage_lats / (eeg_srate/bold_srate)
        tr_lats = tr_lats.astype(int)
        
        lat_epochs = np.zeros((bold_comps.shape[1], tr_lats.shape[0],int(bold_epochl)))
        for lat in np.arange(0,tr_lats.shape[0]):
            delete_last = False
            if tr_lats[lat] + bold_epochl < bold_comps.shape[0]:
                lat_epochs[:,lat,:] = bold_comps[tr_lats[lat]:tr_lats[lat]+bold_epochl,:].transpose()
            else:
                delete_last = True
            
            if delete_last == True:
                lat_epochs = np.delete(lat_epochs,lat_epochs.shape[1]-1,axis=1)
    
        f,pxx = signal.welch(lat_epochs,bold_srate)
        bold_pxx[stage_count,:,:] = np.mean(pxx,axis=1)
        stage_count = stage_count + 1

    return bold_pxx, f

"""
pxx_eeg_component_epochs
epoch EEG signal based on triggers, and compute average power spectral 
density for each trigger

returns array of [trigger, electrode, frequency] and the frequency axis

"""
def pxx_eeg_epochs(raw_data, triggers, epoch_samples ):
    
    stage_power = np.zeros((len(triggers),raw_data.shape[0],129))
    stage_count = 0 
    for stage in triggers:
        print(stage)
        stage_inds = [index for index, value in enumerate(event_ids) if value == stage]
        stage_lats = event_lats[stage_inds]
        lat_epochs = np.zeros((raw_data.shape[0],stage_lats.shape[0],epoch_samples))
        for lat in np.arange(0,lat_epochs.shape[1]):
            delete_last = False 
            if stage_lats[lat] + epoch_samples < raw_data.shape[1]:
                lat_epochs[:,lat,:] = raw_data[:, stage_lats[lat]:stage_lats[lat]+epoch_samples]        
            else:
                delete_last = True
         
        if delete_last == True:
            np.delete(lat_epochs, lat_epochs.shape[1] - 1, axis=1)
                       
        pxx = np.zeros((raw_data.shape[0],lat_epochs.shape[1],129))
        for chan in np.arange(0,raw_data.shape[0]):
            freqs, pxx[chan,:,:] = signal.welch(lat_epochs[chan,:,:],fs=250)
        
        stage_power[stage_count,:,:] = np.mean(pxx,axis=1)
        stage_count = stage_count + 1
    
    return stage_power, freqs


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
comps = canica.transform([fmri])[0]
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
    epochs = np.zeros((raw_data.shape[0], trig_lats.shape[0], pre_samples + post_samples))
    delete_last = False
    for lat in np.arange(0,trig_lats.shape[0]):
        if (trig_lats[lat]-pre_samples > 0) & (
        trig_lats[lat] + post_samples < raw_data.shape[1]):
            epochs[:,lat,:] = raw_data[:,trig_lats[lat]-pre_samples : 
                                         trig_lats[lat]+post_samples] 
        else:
            delete_last = True
            
    if delete_last == True:
        epochs = np.delete(epochs,epochs.shape[1]-1, axis=1)
    
    return epochs

pre_samples = 0 
post_samples = 7500
srate = 250 
trigger_names = ['wake','NREM1','NREM2','NREM3']

all_pxx = np.zeros([4,63,129])
for trig in np.arange(0, len(trigger_names)):
    trig_inds = get_trigger_inds(trigger_names[trig], event_ids)
    trig_lats = event_lats[trig_inds]
    epochs = epoch_triggers(raw_data, trig_lats, pre_samples, post_samples)
    epochs = np.reshape(epochs,[epochs.shape[0],epochs.shape[1]*epochs.shape[2]])
    f, pxx = signal.welch(epochs, srate)
    all_pxx[trig,:,:] = pxx
# have another function for computing the coupling during different epochs


