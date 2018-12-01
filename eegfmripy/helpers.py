import nibabel as nib 
import mne as mne
import numpy as np

"""
read .vmrk file, get all unique events, and return the following lists

list1: event ids
list2: event latencies
"""

def read_vmrk(path):

    with open(path) as f:
        content = f.readlines()
    
    event_ids = []
    event_lats = []
    for line in content:
        if line[0:2] == 'Mk':
            a = line.split(',')
            event_ids.append(a[1])
            event_lats.append(int(a[2]))
            
    return event_ids, event_lats


"""
fmri_info: get information relating to repetition time (TR), number of slices
per image (n_slices), and number of volumes per scan (n_volumes)

"""
def fmri_info(path):
    fmri = nib.load(path)
    hdr = fmri.get_header()
    
    TR = hdr.get_zooms()[3] 
    n_slices = hdr.get_n_slices()
    n_volumes = hdr.get_data_shape()[3]
    
    return TR, n_slices, n_volumes


"""
create_raw_mne: create a new mne 'raw' data structure from data, channel
names, channel types, and montage

"""

def create_raw_mne(data, ch_names, ch_types, montage):
    info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=250)
    newraw = mne.io.RawArray(data,info)
    newraw.set_montage(montage)
    
    return newraw


"""


"""

def prepare_raw_channel_info(downsampled, raw, montage, bads):
    
    eeg_inds = np.arange(0,downsampled.shape[0])
    
    eeg_inds = np.delete(eeg_inds, bads)
    
    newinds = np.zeros(eeg_inds.shape)
    positions = np.zeros([eeg_inds.shape[0],2])
    for i in np.arange(0,eeg_inds.shape[0]):    
        newinds[i] = np.int(montage.ch_names.index(raw.ch_names[eeg_inds[i]]))
        positions[i,:] = montage.get_pos2d()[newinds[i].astype(int),:]
    
    ch_types = [] 
    ch_names = []
    for i in np.arange(0,63):
        ch_types.append('eeg')
        ch_names.append(raw.ch_names[eeg_inds[i]])
        
    return ch_names, ch_types, eeg_inds


def montage_path():
    return '/media/sf_shared/standard-10-5-cap385.elp'

def test_data_path():
    return '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vhdr'


