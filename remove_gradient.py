import mne as mne
import numpy as np
from scipy import signal
from scipy.interpolate import griddata
import matplotlib.animation as animation
from mne.preprocessing import ICA

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_E_DRIVE/')
mne.io.read_raw_brainvision('/media/sf_E_DRIVE/badger_eeg/russell/retino_gamma_01.vhdr',montage=montage,eog=[31])

