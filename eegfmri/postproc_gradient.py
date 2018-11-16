import mne as mne 
import numpy as np
import matplotlib.pyplot as plt 

data_path = '/media/sf_shared/graddata/dec_chans.npy'
raw_eeg_path = '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vhdr'
montage_path = '/media/sf_shared/standard-10-5-cap385.elp'
montage = mne.channels.read_montage(montage_path)
raw = mne.io.read_raw_brainvision(
        raw_eeg_path,montage=montage,eog=['ECG','ECG1'])

#dec = np.load(data_path)
rawdat = raw.get_data()
#raw = raw.resample(250)

