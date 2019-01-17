import nibabel as nib 
import mne as mne
import numpy as np
from scipy.stats import gamma

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
    event_labs = []
    for line in content:
        if line[0:2] == 'Mk':
            a = line.split(',')
            event_ids.append(a[1])
            event_lats.append(int(a[2]))
            
            lab = a[0]
            eq_split = lab.split('=')
            event_labs.append(eq_split[1])
            
            
    return event_ids, event_lats, event_labs


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
prepare_raw_channel_info
"""
def prepare_raw_channel_info(downsampled, raw, montage,eeg_inds):
            
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
    return '/media/sf_hcp/standard-10-5-cap385.elp'

def test_data_path():
    return '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vhdr'

"""
trig_info
"""
def trig_info(event_ids, event_lats, trig_name):
    event_lats = np.array(event_lats)
    trig_inds = [index for index, value in enumerate(event_ids) if value == trig_name]
    trig_inds = np.array(trig_inds)
    print(trig_inds.shape)
    trig_lats = event_lats[trig_inds]
    
    return trig_inds, trig_lats

"""
check_vol_triggers
"""
def check_vol_triggers(n_volumes, grad_inds):
    if n_volumes == grad_inds.shape[0]:
        print('#volumes matches #volume trigs')
              
        return True 
    else:
        print('WARNING: #volumes does not match #volume trigs')
        print('#volumes = ' + str(n_volumes))
        print('#volume trigs = ' + str(grad_inds.shape[0]))
        print('will use # volumes to trim EEG data')
              
        return False
  

    
"""
isolate_eeg_channels
get the EEG and ECG channels indices from the montage in two separate arrays
"""
def isolate_eeg_channels(raw, montage):

    heart_chans = ['ECG','ECG1']
    
    eeg_chan_inds = []
    heart_chan_inds = [] 
    
    for i in np.arange(0, len(raw.ch_names)):
        if raw.ch_names[i] in montage.ch_names:
            eeg_chan_inds.append(i)
        if raw.ch_names[i] in heart_chans:
            heart_chan_inds.append(i)
    
    eeg_chan_inds = np.asarray(eeg_chan_inds)
    heart_chan_inds = np.asarray(heart_chan_inds)             
    
    return eeg_chan_inds, heart_chan_inds 
              

"""
get_hrf
"""
def get_hrf(times):
    peak_values = gamma.pdf(times, 6)
    undershoot_values = gamma.pdf(times, 12)
    values = peak_values - 0.35 * undershoot_values
    return values / np.max(values) * 0.6













