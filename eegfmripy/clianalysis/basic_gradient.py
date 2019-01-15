import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mne as mne
from scipy import stats
import sys as sys
from mne.preprocessing import ICA
sys.path.insert(0, '/media/sf_shared/eegfmripy/eegfmripy')
import helpers

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
    current_i = 0 
    for i in np.arange(0, np.min([n_volumes,grad_lats.shape[0]])):
        if grad_lats[i] + tr_samples < data.shape[0]:
            vol_epochs[i,:] = data[grad_lats[i]:grad_lats[i] + tr_samples]
            vol_inds[i,:] = np.arange(grad_lats[i], grad_lats[i] + tr_samples)
            current_i = current_i + 1
        else:
            vol_epochs[i,:] = data[grad_lats[current_i-1]:grad_lats[current_i-1] + tr_samples]
            vol_inds[i,:] = np.arange(grad_lats[current_i-1], grad_lats[current_i-1] + tr_samples)
    
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

#CoRe_011, CoRe_023, CoRe_054, CoRe_079, CoRe_082, CoRe_087, CoRe_094, CoRe_100
#CoRe_107, CoRe_155, CoRe_192, CoRe_195, CoRe_220, CoRe_235, CoRe_267, CoRe_268





base_path = '/media/sf_hcp/sleepdata/'
eeg_sub = 'CoRe_296/eeg/'
eeg_path = 'CoRe_296_Day2_Night_01.vhdr'
trig_path = 'CoRe_296_Day2_Night_01.vmrk'

fmri_sub = 'CoRe_296/rfMRI/'
fmri_path = 'd2/sleep_0.nii'

eeg_srate = 5000 
grad_trigger = 'Volume'

tr, n_slices, n_volumes = helpers.fmri_info(base_path + fmri_sub + fmri_path)
event_ids, event_lats, event_labs = helpers.read_vmrk(base_path + eeg_sub + trig_path)

trig_inds = [index for index, value in enumerate(event_labs) if value == 'Volume']

# perform basic check to ensure number of volumes match number of volume trigs
grad_inds, grad_lats = helpers.trig_info(event_labs, event_lats, grad_trigger)
matching = helpers.check_vol_triggers(n_volumes, grad_inds)

# if the basic check fails, remove extra volume trigs at end of scan
if matching==False:
    if n_volumes < grad_inds.shape[0]:
        n_extra = grad_inds.shape[0] - n_volumes
        extra_inds = np.arange(grad_inds.shape[0]-n_extra, grad_inds.shape[0])
        grad_inds = np.delete(grad_inds,extra_inds)
        grad_lats = np.delete(grad_lats, extra_inds)

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_hcp/')
raw = mne.io.read_raw_brainvision(base_path + eeg_sub + eeg_path,
                                  montage=montage,eog=['ECG','ECG1'])

eeg_ch_inds, ecg_ch_inds = helpers.isolate_eeg_channels(raw, montage)

step_sz = int(eeg_ch_inds.shape[0]/3)
picks1 = eeg_ch_inds[np.arange(0,step_sz)]
picks2 = eeg_ch_inds[np.arange(step_sz,step_sz*2)]
picks3 = eeg_ch_inds[np.arange(step_sz*2,eeg_ch_inds.shape[0])]

new_srate = 250 
dec = int(5000/new_srate) 

start_ind = grad_lats[0]
end_ind = grad_lats[-1]

all_picks = [picks1,picks2,picks3]

sz_data = raw.get_data(picks=0)
downsampled = np.zeros((eeg_ch_inds.shape[0], int(sz_data.shape[1]/dec)))
downsampled_resid = np.zeros((len(all_picks),downsampled.shape[1]))

chan_count = 0
for p in np.arange(0,len(all_picks)):
    
    graddata = raw.get_data(picks=all_picks[p])   

    tr_samples = int(tr*5000)
    residuals = np.zeros(graddata[0,:].shape)
    
    for i in np.arange(0,graddata.shape[0]):
        graddata[i,:], resid = basic_gradient_removal(
                graddata[i,:], grad_lats, n_volumes, tr_samples)
        
        residuals = residuals + resid/(graddata.shape[0]-1)
   
    graddata[:,0:start_ind] = 0
    graddata[:,end_ind:-1] = 0 
    
    dec_residuals = signal.decimate(residuals,dec) 
    downsampled_resid[p,:] = dec_residuals 
    for i in np.arange(0,graddata.shape[0]):
        downsampled[chan_count,:] = signal.decimate(graddata[i,:], dec)
        chan_count = chan_count + 1 
        
    del graddata



ch_names, ch_types, inds = helpers.prepare_raw_channel_info(
        downsampled, raw, montage, eeg_ch_inds)

new_raw = helpers.create_raw_mne(downsampled, ch_names, ch_types, montage)

new_raw.save(base_path + eeg_sub + eeg_path.replace('.vhdr','.fif'))
plt.plot(new_raw.get_data()[0,:])


"""
new_raw.filter(1,120)
ica = ICA(n_components=60, method='fastica', random_state=23)
ica.fit(new_raw)
ica.plot_components(picks=np.arange(0,60))
src = ica.get_sources(new_raw).get_data()

res_grad_lats = grad_lats / dec 
start_t = res_grad_lats[0]
end_t = res_grad_lats[-1]

info = mne.create_info(['STIM_GRAD'], new_raw.info['sfreq'], ['stim'])
stim_data = np.zeros((1,src.shape[1]))
stim_data[0,res_grad_lats.astype(int)] = 1 
stim_raw = mne.io.RawArray(stim_data, info)
new_raw.add_channels([stim_raw], force_update_info=True)

new_raw.save('new_raw.fif')
"""