import mne as mne 
import numpy as np
import matplotlib.pyplot as plt 
from mne.preprocessing import ICA
import pandas as pd 
from scipy import stats


def create_raw_mne(data, ch_names, ch_types, montage):
    info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=250)
    newraw = mne.io.RawArray(data,info)
    newraw.set_montage(montage)
    
    return newraw

data_path = '/media/sf_shared/graddata/dec_chans.npy'
raw_eeg_path = '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vhdr'
montage_path = '/media/sf_shared/standard-10-5-cap385.elp'
montage = mne.channels.read_montage(montage_path)
raw = mne.io.read_raw_brainvision(
        raw_eeg_path,montage=montage,eog=['ECG','ECG1'])

dec = np.load(data_path)
#rawdat = raw.get_data()
#raw = raw.resample(250)

inds = np.arange(0,dec.shape[0])
bads = [raw.ch_names.index('ECG'), raw.ch_names.index('ECG1')]
inds = np.delete(inds, bads)

newinds = np.zeros(inds.shape)
positions = np.zeros([inds.shape[0],2])
for i in np.arange(0,inds.shape[0]):    
    newinds[i] = np.int(montage.ch_names.index(raw.ch_names[inds[i]]))
    positions[i,:] = montage.get_pos2d()[newinds[i].astype(int),:]

ch_types = [] 
ch_names = []
for i in np.arange(0,63):
    ch_types.append('eeg')
    ch_names.append(raw.ch_names[inds[i]])
    


newraw = create_raw_mne(dec[inds,:], ch_names, ch_types, montage)
newraw.filter(1,120)
# remove some bad epochs
newraw_data = newraw.get_data()
raw_std_z = stats.zscore(np.std(np.abs(np.diff(newraw_data,axis=1)),axis=0))
smooth_std = np.convolve(raw_std_z, np.ones(10)/100,mode='same')
bads = np.where(smooth_std > 0.1)

newraw_data[:,bads] = 0 

raw_interp = create_raw_mne(newraw_data, ch_names, ch_types, montage)
ica2 = ICA(n_components=60, method='fastica', random_state=23)
ica2.fit(raw_interp)
ica2.plot_components(picks=np.arange(0,60))
src = ica2.get_sources(newraw).get_data()



