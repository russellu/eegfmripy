import mne as mne 
import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np 
import sys as sys

import remove_bcg
from ..utils import helpers
from ..cli import AnalysisParser

def run(args=None, config=None):
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

    raw = mne.io.read_raw_fif('/media/sf_shared/graddata/new_raw.fif')
    raw.load_data()
    heartdata = remove_bcg.sort_heart_components(raw)

    """
    run_bcg_denoise.py

    run the bcg denoising pipeline - calls functions from remove_bcg.py

    isolates the heartbeat component by running ICA and finding 
    top median skew component

    finds peaks by first 'chunking' the data (peak detection is slow on long ts)
    and using wavelet peak detection to locate peaks of specified width

    removes spurious peaks by correlating each peak with all other peaks, and
    removing the least commonly occuring peaks (assumes true peaks are most common)

    re-aligns peaks across all channels using cross-correlation within ~100ms 
    acceptable peak range shift (empirical values across scalp)

    finally, subtracts peaks from similar peaks using ssd similarity metric across
    the entirety of the peak epoch vector 

    """

    # chunk the data and get heartbeat peaks in individual chunks

    chunk_size = 50000
    all_peak_inds = np.zeros(0)
    for i in np.arange(0,heartdata.shape[1], chunk_size):
        print(i)
        if i+chunk_size < heartdata.shape[1]:
            peak_inds = remove_bcg.get_heartbeat_peaks(heartdata[0,i:i+chunk_size]) + i
            new_peak_inds = np.zeros(all_peak_inds.shape[0] + peak_inds.shape[0])
            new_peak_inds[0:all_peak_inds.shape[0]] = all_peak_inds[0:]
            new_peak_inds[all_peak_inds.shape[0]:] = peak_inds[0:]
            all_peak_inds = new_peak_inds 
        else:
            peak_inds = remove_bcg.get_heartbeat_peaks(heartdata[0,i:]) + i
            new_peak_inds = np.zeros(all_peak_inds.shape[0] + peak_inds.shape[0])
            new_peak_inds[0:all_peak_inds.shape[0]] = all_peak_inds[0:]
            new_peak_inds[all_peak_inds.shape[0]:] = peak_inds[0:]
            all_peak_inds = new_peak_inds 

    all_peak_inds = all_peak_inds.astype(int)
    peak_arr = np.zeros(heartdata.shape[1])
    peak_arr[all_peak_inds] = 1 

    peak_inds = remove_bcg.remove_bad_peaks(heartdata[0,:], all_peak_inds)

    mean_hr, hr_ts = remove_bcg.get_heartrate(raw,heartdata[0,:],peak_inds)

    bcg_epochs, bcg_inds = remove_bcg.epoch_channel_heartbeats(
            raw.get_data(), int(mean_hr*0.95), peak_inds, raw.info['sfreq'])

    shifted_epochs, shifted_inds = remove_bcg.align_heartbeat_peaks(
            bcg_epochs, bcg_inds)

    subbed_raw = remove_bcg.subtract_heartbeat_artifacts(
           raw.get_data(), shifted_epochs, shifted_inds)

    bads = [raw.ch_names.index('STIM_GRAD')]
    ch_names, ch_types, eeg_inds = helpers.prepare_raw_channel_info(
            subbed_raw, raw, mne.channels.read_montage(helpers.montage_path()), bads)

    new_raw = helpers.create_raw_mne(subbed_raw[0:63,:], ch_names, ch_types,
              mne.channels.read_montage(helpers.montage_path()))

    save_path = '/media/sf_shared/graddata/bcg_denoised_raw.fif'

    new_raw.save(save_path)


