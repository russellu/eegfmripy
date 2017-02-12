import mne as mne 
import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np 
import sys as sys
import logging
import os
import time
import code

from mne.time_frequency import tfr_morlet
from mne.preprocessing import ICA

from ..clianalysis import remove_bcg as bcg_utils
from ..utils import helpers
from ..cli import AnalysisParser

log = logging.getLogger("eegfmripy")


def remove_bcg(
        raw,
        montage_path,
        output='bcgremoved' + str(int(time.time())) + '.fif',
        debug_plot=False,
        debug_plot_verbose=False,
        bcg_peak_widths=list(range(25,45)),
        display_ica_components=False
    ):

    orig_srate = raw.info['sfreq']
    raw = raw.resample(256, verbose='error')
    print("Srate: %s" % raw.info['sfreq'])

    events = mne.find_events(raw)
    print(events)
    raw = raw.resample(raw.info['sfreq'], verbose='error')

    heartdata = bcg_utils.sort_heart_components(raw)

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
            peak_inds = bcg_utils.get_heartbeat_peaks(heartdata[0,i:i+chunk_size], peak_widths=bcg_peak_widths) + i
            new_peak_inds = np.zeros(all_peak_inds.shape[0] + peak_inds.shape[0])
            new_peak_inds[0:all_peak_inds.shape[0]] = all_peak_inds[0:]
            new_peak_inds[all_peak_inds.shape[0]:] = peak_inds[0:]
            all_peak_inds = new_peak_inds 
        else:
            peak_inds = bcg_utils.get_heartbeat_peaks(heartdata[0,i:], peak_widths=bcg_peak_widths) + i
            new_peak_inds = np.zeros(all_peak_inds.shape[0] + peak_inds.shape[0])
            new_peak_inds[0:all_peak_inds.shape[0]] = all_peak_inds[0:]
            new_peak_inds[all_peak_inds.shape[0]:] = peak_inds[0:]
            all_peak_inds = new_peak_inds 

    all_peak_inds = all_peak_inds.astype(int)
    peak_arr = np.zeros(heartdata.shape[1])
    peak_arr[all_peak_inds] = 1

    peak_inds = bcg_utils.remove_bad_peaks(heartdata[0,:], all_peak_inds)
    print(len(peak_inds))

    plt.figure()
    plt.plot(heartdata[0,:])
    for i in peak_inds:
        plt.axvline(x=i, color='black')
    plt.show()

    if debug_plot_verbose:
        for t in range(heartdata.shape[0]):
            plt.figure()
            plt.plot(heartdata[t,:])
            for i in peak_inds:
                plt.axvline(x=i, color='black')
        plt.show()

    mean_hr, hr_ts = bcg_utils.get_heartrate(raw,heartdata[0,:],peak_inds)
    print("HR:")
    print(mean_hr)


    bcg_epochs, bcg_inds = bcg_utils.epoch_channel_heartbeats(
            raw.get_data(), int(mean_hr*0.95), peak_inds, raw.info['sfreq'])

    log.info(
        "In this figure, a recurring pattern (the BCG artifact) should be visible.\n" +
        "If not, either gradient artifact removal failed, or the BCG peak widths must\n" +
        "be changed."
    )
    plt.figure()
    print(bcg_epochs.shape)
    plt.imshow(np.squeeze(bcg_epochs[3,:,:]))
    plt.clim(-0.0001, 0.0001)
    plt.show()


    # Setup for reading the raw data
    events = mne.find_events(raw)

    shifted_epochs, shifted_inds = bcg_utils.align_heartbeat_peaks(
            bcg_epochs, bcg_inds)

    subbed_raw = bcg_utils.subtract_heartbeat_artifacts(
           raw.get_data(), shifted_epochs, shifted_inds)

    new_raw = helpers.create_raw_mne(subbed_raw, raw.ch_names, ['eeg' for _ in range(len(raw.ch_names))],
              mne.channels.read_montage(montage_path))

    fname = output

    # Setup for reading the raw data
    events = mne.find_events(raw)

    raw[0:63,:] = subbed_raw[0:63,:]

    raw.resample(512, npad='auto', verbose='error')
    raw.save(fname)

    if display_ica_components:
        n_components = 60
        log.info("Displaying ICA components post-clean...")
        raw.filter(1,250)
        ica = ICA(n_components=n_components, method='fastica', random_state=23)

        ica.fit(raw,decim=4)
        ica.plot_components(picks=np.arange(0,60), cmap='jet')

        # Setup for reading the raw data
        events = mne.find_events(raw)

        # Construct Epochs
        icaraw = ica.get_sources(raw)
        event_id, tmin, tmax = 11, -1., 3.
        baseline = (None, 0)
        epochs = mne.Epochs(icaraw, events, event_id, tmin, tmax,
                            baseline=baseline,
                            preload=True)

        freqs = np.logspace(*np.log10([1, 100]), num=100)
        n_cycles = freqs / 2.  # different number of cycle per frequency
        power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, picks=list(range(n_components)))
        power.plot(list(range(n_components)), baseline=(-0.5, 0), mode='percent',yscale='linear', cmap='jet', vmin=-5, vmax=5)

    return raw


def run(args=None, config=None):
    if not config:
        parser = AnalysisParser('config')
        args = parser.parse_analysis_args(args)
        config = args.config

    montage_path = config['montage_path']
    gradrem_path = config['raw_gradrem_fif']
    output = config['output']
    display_ica_components = config['display_ica_components']

    debug_plot = False
    if 'debug-plot' in config:
        debug_plot = config['debug-plot']

    debug_plot_verbose = False
    if 'debug-plot-verbose' in config:
        debug_plot_verbose = config['debug-plot-verbose']

    bcg_peak_widths = list(range(25,45))
    if 'bcg_peak_widths' in config:
        bcg_peak_widths = config['bcg_peak_widths']

    raw = mne.io.read_raw_fif(gradrem_path)
    raw.load_data()
    tmpdata = raw.get_data()
    events = mne.find_events(raw)
    print(events)

    for t in range(tmpdata.shape[0]):
        raw[t,:1790] = 0

    remove_bcg(
        raw,
        montage_path,
        output=os.path.join(
            output,
            'bcgngradrem_' + str(int(time.time())) + gradrem_path.split('/')[-1].split('.')[0] + '.fif'
        ),
        debug_plot=debug_plot,
        debug_plot_verbose=debug_plot_verbose,
        bcg_peak_widths=bcg_peak_widths,
        display_ica_components=display_ica_components
    )

    log.info("Interactive mode started...")
    code.interact(local=locals())

