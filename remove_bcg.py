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

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_E_DRIVE/')
raw = mne.io.read_raw_eeglab('/media/sf_E_DRIVE/badger_eeg/alex/gradeeg_retino_gamma_01.set',montage=montage,eog=[31])
raw.load_data()

raw_data = raw.get_data()[0:64,:]
hp_raw = mne.filter.filter_data(raw_data,250,1,124)
# epoch_bcg - get the epoched BCG artifacts

raw.filter(1,90)

ica = ICA(n_components=35, method='fastica')
ica.fit(raw)
src = ica.get_sources(raw)
srcdata = src.get_data()

peak_inds = signal.find_peaks_cwt(srcdata[0,:],[20,25,30])
vlines = np.zeros([srcdata.shape[1]])
vlines[peak_inds] = 1

plt.plot(srcdata[0,:]) ; plt.plot(vlines)

epochl = 250 
halfepochl = np.int(epochl/2)

bcg_epochs = np.zeros([peak_inds.shape[0], epochl])
icount = 0
for ind in peak_inds:
    if (ind-halfepochl >= 0) & (ind + halfepochl <= srcdata.shape[1]):
        bcg_epochs[icount,:] = srcdata[0,ind-halfepochl:ind+halfepochl]
        icount = icount + 1
                
corrs = np.corrcoef(bcg_epochs)    
zcorrs = stats.zscore(np.sum(corrs,axis=1))
badinds = np.where(zcorrs < -2)[0]
peak_inds = np.delete(peak_inds, badinds)

bcg_epochs = np.zeros([hp_raw.shape[0],peak_inds.shape[0], epochl])
bcg_inds = np.zeros([peak_inds.shape[0],epochl])
icount = 0
for ind in peak_inds:
    if (ind-halfepochl >= 0) & (ind + halfepochl <= srcdata.shape[1]):
        bcg_epochs[:,icount,:] = hp_raw[:,ind-halfepochl:ind+halfepochl]
        bcg_inds[icount,:] = np.arange(ind-halfepochl,ind+halfepochl) 
        icount = icount + 1
        
bcg_inds = bcg_inds.astype(int)
        
mean_ts = np.zeros(bcg_epochs.shape)
for chan in np.arange(0,mean_ts.shape[0]):
    print(chan)
    ts = np.squeeze(bcg_epochs[chan,:,:])
    n_avgs = 29
    for i in np.arange(0,ts.shape[0]):
        ts_i = ts[i,:]
        sum_abs_diffmat = np.sum(np.abs(ts - np.tile(ts_i, (ts.shape[0],1))),axis=1)
        sort_epochs = np.argsort(sum_abs_diffmat)
        mean_ts_i = np.mean(ts[sort_epochs[1:n_avgs],:],axis=0)
        mean_ts[chan,i,:] = mean_ts_i 
                
subbed_raw = raw_data.copy()
for i in np.arange(0, bcg_inds.shape[0]):
    subbed_raw[:,bcg_inds[i,:]] = raw_data[:,bcg_inds[i,:]] - mean_ts[:,i,:]
    
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

# epoch based on the peak indices


#centroids = kmeans(bcg_epochs[:,halfepochl-50:halfepochl+50], 3)[0]
#inds = vq(bcg_epochs[:,halfepochl-50:halfepochl+50], centroids)













