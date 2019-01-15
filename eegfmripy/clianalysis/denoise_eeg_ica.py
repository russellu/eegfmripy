import mne as mne 
import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np 
from mne.preprocessing import ICA


"""
functions to denoise an EEG dataset using ICA

at the moment, relies on hand-picked components, but in the future will be
able to classify components automatically using neural networks based on the 
components scalp map and power spectrum 

"""

raw_path = '/media/sf_shared/graddata/bcg_denoised_raw.fif' 
raw = mne.io.read_raw_fif(raw_path)
raw.load_data()

raw_filt = raw
raw_filt.filter(1,124.5)

ica2 = ICA(n_components=60, method='fastica', random_state=23)
ica2.fit(raw_filt)
ica2.plot_components(picks=np.arange(0,60))
src = ica2.get_sources(raw).get_data()

# list of good components 
includes = [5,6,8,10,11,12,13,15,16,17,19,20,22,23,27,28,29,32,36,38]

# subtract out bad components
ica_raw = ica2.apply(raw, include=includes)

ica_path = '/media/sf_shared/graddata/ica_denoised_raw.fif' 
ica_raw.save(ica_path)



