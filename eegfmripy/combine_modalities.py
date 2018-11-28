import mne as mne 
import nibabel as nib 
import remove_bcg
import matplotlib.pyplot as plt 
import numpy as np 

raw = mne.io.read_raw_fif('/media/sf_shared/graddata/new_raw.fif')
raw.load_data()
heartdata = remove_bcg.sort_heart_components(raw)


# chunk the data and get heartbeat peaks in individual chunks

chunk_size = 50000
all_peak_inds = np.zeros(0)
for i in np.arange(0,heartdata.shape[1], chunk_size):
    print(i)
    if i+chunk_size < heartdata.shape[1]:
        peak_inds = remove_bcg.get_heartbeat_peaks(heartdata[0,i:i+chunk_size]) + i
        all_peak_inds = np.concatenate(all_peak_inds, peak_inds)
    else:
        peak_inds = remove_bcg.get_heartbeak_peaks(heartdata[0,i:]) + i
        all_peak_inds = np.concatenate(all_peak_inds, peak_inds)

peak_arr = np.zeros([0,100000])
peak_arr[peak_inds] = 1 




peak_inds = remove_bcg.remove_bad_peaks(heartdata[0,:], peak_inds)

mean_hr, hr_ts = remove_bcg.get_heartrate(heartdata[0,:],peak_inds)
bcg_epochs, bcg_inds = remove_bcg.epoch_channel_heartbeats(
        hp_raw_data, int(mean_hr*0.95), raw.info['sfreq'])

