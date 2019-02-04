import mne as mne 
import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np 
import sys as sys
import logging
import os
import time

from ..clianalysis import remove_bcg
from ..utils import helpers
from ..cli import AnalysisParser

log = logging.getLogger("eegfmripy")


def run(args=None, config=None):
    if not config:
        parser = AnalysisParser('config')
        args = parser.parse_analysis_args(args)
        config = args.config

    montage_path = config['montage_path']
    gradrem_path = config['raw_gradrem_fif']
    output = config['output']

    debug_plot = False
    if 'debug-plot' in config:
        debug_plot = config['debug-plot']

    raw = mne.io.read_raw_fif(gradrem_path)
    raw.load_data()
    tmpdata = raw.get_data()

    for t in range(tmpdata.shape[0]):
        raw[t,:1790] = 0

    raw = raw.copy().resample(250, npad='auto', verbose='error')
    print("Srate: %s" % raw.info['sfreq'])
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
    log.info("Gathering peak inds..")
    for i in np.arange(0,heartdata.shape[1], chunk_size):
        log.info("On chunk starting from %s" % str(i))
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
    print(len(peak_inds))

    plt.figure()
    plt.plot(heartdata[0,:])
    for i in peak_inds:
        plt.axvline(x=i, color='black')
    plt.show()

    if debug_plot:
        for t in range(heartdata.shape[0]):
            plt.figure()
            plt.plot(heartdata[t,:])
            for i in peak_inds:
                plt.axvline(x=i, color='black')
            plt.show()

    mean_hr, hr_ts = remove_bcg.get_heartrate(raw,heartdata[0,:],peak_inds)
    print("HR:")
    print(mean_hr)

    bcg_epochs, bcg_inds = remove_bcg.epoch_channel_heartbeats(
            raw.get_data(), int(mean_hr*0.95), peak_inds, raw.info['sfreq'])

    plt.figure()
    print(bcg_epochs.shape)
    plt.imshow(np.squeeze(bcg_epochs[3,:,:]))
    plt.clim(-0.0001, 0.0001)
    plt.show()

    shifted_epochs, shifted_inds = remove_bcg.align_heartbeat_peaks(
            bcg_epochs, bcg_inds)

    subbed_raw = remove_bcg.subtract_heartbeat_artifacts(
           raw.get_data(), shifted_epochs, shifted_inds)

    new_raw = helpers.create_raw_mne(subbed_raw, raw.ch_names, ['eeg' for _ in range(len(raw.ch_names))],
              mne.channels.read_montage(montage_path))

    fname = os.path.join(
        output,
        'bcgrem_gradrem_' + str(int(time.time())) + gradrem_path.split('/')[-1].split('.')[0] + '.fif'
    )

    new_raw.save(fname)


