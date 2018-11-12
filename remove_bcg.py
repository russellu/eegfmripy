import mne as mne
import numpy as np
from scipy import signal
from scipy.interpolate import griddata
from scipy.cluster.vq import kmeans, vq
from scipy import signal
import matplotlib.animation as animation
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from scipy import stats

def sort_heart_components(raw):
    raw.filter(1,90)
    ica = ICA(n_components=60, method='fastica')
    ica.fit(raw)
    src = ica.get_sources(raw)
    srcdata = src.get_data()
    skew_l = 700
    n_skews = int(srcdata.shape[1] / skew_l)
    skews = np.zeros([srcdata.shape[0], n_skews])
    icount = 0
    for i in np.arange(0,srcdata.shape[1] - skew_l, skew_l):
        skews[:,icount] = stats.skew(srcdata[:,i:i+skew_l],axis=1) 
        icount = icount + 1
    
    neg_skews = np.where(np.mean(skews, axis=1) < 0) 
    skews[neg_skews,:] = skews[neg_skews,:] * -1 
    srcdata[neg_skews,:] = srcdata[neg_skews,:] * -1 
    sort_skews_desc = np.flip(np.argsort(np.mean(skews, axis=1)))
    
    return srcdata[sort_skews_desc,:]

def get_heartbeat_peaks(peak_signal, peak_widths=[20,25,30,35,40,45]):
    peak_inds = signal.find_peaks_cwt(peak_signal, peak_widths)
    return peak_inds

def remove_bad_peaks(heartdata, peak_inds):  
    epochl = 150 
    halfepochl = np.int(epochl/2)
    
    bcg_epochs = np.zeros([peak_inds.shape[0], epochl])
    icount = 0
    for ind in peak_inds:
        if (ind-halfepochl >= 0) & (ind + halfepochl <= heartdata.shape[0]):
            bcg_epochs[icount,:] = heartdata[ind-halfepochl:ind+halfepochl]
        icount = icount + 1      
            
    corrs = np.corrcoef(bcg_epochs)    
    corrs = np.nan_to_num(corrs)   
    zcorrs = stats.zscore(np.sum(corrs,axis=1))
    badinds = np.where(zcorrs < -4.5)   
    peak_inds = np.delete(peak_inds, badinds)
    
    return peak_inds

def get_heartrate(heartdata, peak_inds):
    averaging_window_l = 15 
    samples_window_l = int(raw.info['sfreq'] * averaging_window_l)
    mean_hr = (raw.info['sfreq'] / np.mean(np.diff(peak_inds))) * 60
    hr_ts = np.zeros(heartdata.shape[0])
    beat_ts = np.zeros(heartdata.shape[0])
    beat_ts[peak_inds] = 1 
    for i in np.arange(samples_window_l,heartdata.shape[0]):
        hr_ts[i] = (np.sum(beat_ts[i-samples_window_l:i]) / averaging_window_l) * 60
    
    hr_ts[0:samples_window_l] = hr_ts[samples_window_l]
    
    return mean_hr, hr_ts

def epoch_channel_heartbeats(hp_raw_data, mean_hr, sfreq):
    
    epochl = int((1/(mean_hr/60)) * sfreq)
    print(epochl)
    if epochl%2 == 1:
        epochl = epochl + 1
        
    halfepochl = int(epochl/2)    
    bcg_epochs = np.zeros([hp_raw_data.shape[0],peak_inds.shape[0], epochl])
    bcg_inds = np.zeros([peak_inds.shape[0],epochl])
    icount = 0
    for ind in peak_inds:
        if (ind-halfepochl >= 0) & (ind + halfepochl <= heartdata.shape[1]):
            bcg_epochs[:,icount,:] = hp_raw_data[:,ind-halfepochl:ind+halfepochl]
            bcg_inds[icount,:] = np.arange(ind-halfepochl,ind+halfepochl) 
        icount = icount + 1            
    bcg_inds = bcg_inds.astype(int)
    
    return bcg_epochs, bcg_inds 

  
def align_heartbeat_peaks(bcg_epochs, bcg_inds):
    xcorrs = np.zeros(bcg_epochs.shape)
    for chan in np.arange(0,bcg_epochs.shape[0]):
        mchan = np.mean(bcg_epochs[chan,:,:],axis=0)
        for i in np.arange(0,bcg_epochs.shape[1]):
            xcorrs[chan,i,:] = signal.correlate(mchan,bcg_epochs[chan,i,:], mode='same')
    
    center_ind = int(bcg_epochs.shape[2]/2)
    mean_xcorrs = np.mean(xcorrs,axis=0)
    max_inds = np.argmax(mean_xcorrs,axis=1)
    center_diffs = max_inds - center_ind 
    bades = np.where(stats.zscore(np.abs(center_diffs)) > 1.5)
    max_shift = 20 
    half_window = int((bcg_epochs.shape[2] - max_shift)/2) 
    shifted_epochs = np.zeros([bcg_epochs.shape[0], bcg_epochs.shape[1], bcg_epochs.shape[2] - max_shift])
    shifted_inds = np.zeros([shifted_epochs.shape[1], shifted_epochs.shape[2]])
    for i in np.arange(0,shifted_epochs.shape[1]):
        if np.abs(center_diffs[i]) <= 10:
            shifted_epochs[:,i,:] = bcg_epochs[:,i,
                center_ind-center_diffs[i]-half_window:center_ind-center_diffs[i]+half_window]
            shifted_inds[i,:] = bcg_inds[i,
                center_ind-center_diffs[i]-half_window:center_ind-center_diffs[i]+half_window]
    
    shifted_inds = shifted_inds.astype(int)
    
    mask = np.ones(shifted_inds.shape[0])
    mask[bades] = False
    goodes = np.where(mask==1)[0]
    shifted_inds = shifted_inds[goodes,:]
    shifted_epochs = shifted_epochs[:,goodes,:]    
    
    return shifted_epochs, shifted_inds


def subtract_heartbeat_artifacts(raw_data, shifted_epochs, shifted_inds, n_avgs=30):
    subbed_raw = raw_data.copy()
    for epoch in np.arange(0,shifted_epochs.shape[1]):
        print(epoch)
        rep_current = np.tile(shifted_epochs[:,[epoch],:],[1,shifted_epochs.shape[1],1])
        rep_diffs = rep_current - shifted_epochs
        mean_diffs = np.mean(np.abs(rep_diffs),axis=0)
        sorted_epochs = np.argsort(np.mean(mean_diffs,axis=1))
        subbed_raw[:,shifted_inds[epoch,:]]  = (raw_data[:,shifted_inds[epoch,:]] 
        - np.mean(shifted_epochs[:,sorted_epochs[1:n_avgs]], axis=1))
        
    return subbed_raw

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_E_DRIVE/')
raw = mne.io.read_raw_eeglab('/media/sf_E_DRIVE/badger_eeg/tegan/gradeeg_retino_rest.set',montage=montage,eog=[31])
raw.load_data()
raw_data = raw.get_data()[0:64,:] 
hp_raw_data = mne.filter.filter_data(raw_data,250,1,124)
heartdata = sort_heart_components(raw)

peak_inds = get_heartbeat_peaks(heartdata[0,:])
peak_inds = remove_bad_peaks(heartdata[0,:], peak_inds)

mean_hr, hr_ts = get_heartrate(heartdata[0,:],peak_inds)
bcg_epochs, bcg_inds = epoch_channel_heartbeats(hp_raw_data, int(mean_hr*0.95), raw.info['sfreq'])
shifted_epochs, shifted_inds = align_heartbeat_peaks(bcg_epochs, bcg_inds)
subbed_raw = subtract_heartbeat_artifacts(raw_data, shifted_epochs, shifted_inds)

    
    

"""
vlines = np.zeros([heartdata.shape[1]])
vlines[peak_inds] = 1
vlines[peak_inds[bades]] = 5
plt.plot(heartdata[0,:]) ; plt.plot(vlines)     

vlines = np.zeros([heartdata.shape[1]])
vlines[peak_inds] = 1
plt.plot(heartdata[0,:]) ; plt.plot(vlines)     
plt.imshow(bcg_epochs[45,:,:])
"""

ch_types = []
ch_names = []
inds = np.arange(0,64)
for i in np.arange(0,64):
    ch_types.append('eeg')
    ch_names.append(raw.ch_names[inds[i]])

info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=250)
newraw = mne.io.RawArray(subbed_raw,info)
newraw.set_montage(montage)
newraw.filter(1,120)
ica2 = ICA(n_components=60, method='fastica', random_state=23)
ica2.fit(newraw)
ica2.plot_components(picks=np.arange(0,60))

plt.plot(raw_data[3,:]); plt.plot(subbed_raw[3,:])