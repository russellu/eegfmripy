import mne as mne

"""
test code and example data for eegfmripy. 

"""

def example_raw():
    montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_shared/')
    raw = mne.io.read_raw_brainvision(
        '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vhdr',
        montage=montage,eog=['ECG','ECG1'])
    
    return raw