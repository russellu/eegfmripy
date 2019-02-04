import mne as mne
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time
import os 
import logging
import copy

from ..cli import AnalysisParser
from ..utils.general_utils import write_same_line, finish_same_line
from ..utils.helpers import save_eeg
from ..utils import helpers

log = logging.getLogger('eegfmripy')


def get_slicegap(single_channel, wsize=25000, maxcorr_pad=50):
    log.info("getting slice gap...")
    mcorrs = np.zeros([wsize,np.int(single_channel.shape[0]/wsize)])
    icount = 0
    for i in np.arange(0, single_channel.shape[0] - wsize - 1, wsize):
        mcorrs[:,icount] = signal.correlate(single_channel[i:i+wsize],single_channel[i:i+wsize], mode='same')
        icount = icount + 1
        
    mcorrs = np.mean(mcorrs,axis=1)
    return np.argmax(mcorrs[np.int(wsize/2)+maxcorr_pad:]) + maxcorr_pad

def get_slicepochs(single_channel, slicegap):
    nepochs = np.int(single_channel.shape[0] / slicegap)
    slice_epochs = np.zeros([nepochs, slicegap])
    slice_inds = np.zeros([nepochs,slicegap])
    icount = 0
    for i in np.arange(0, single_channel.shape[0]-slicegap, slicegap):
        slice_epochs[icount,:] = single_channel[i:i+slicegap]
        slice_inds[icount,:] = np.arange(i,i+slicegap) 
        icount = icount + 1 
            
    slice_inds = slice_inds.astype(int)
    return slice_epochs, slice_inds

def find_bad_slices(slice_epochs, corrthresh=0.9):
    log.info("finding bad slices (slices without gradient artifacts)")
    corrmat = np.corrcoef(slice_epochs)
    print("Finding correlation mats")
    print(corrmat)
    mean_corrmat = np.nanmean(corrmat,axis=1)
    print(mean_corrmat)
    
    good_epoch_inds = np.where(mean_corrmat > corrthresh)[0]
    bad_epoch_inds = np.where(mean_corrmat <= corrthresh)[0]
    corrmat_thresh = mean_corrmat > corrthresh
    print(corrmat_thresh)

    return good_epoch_inds, bad_epoch_inds, corrmat_thresh

def replace_bad_slices(slice_epochs, good_epoch_inds, bad_epoch_inds):
    for i in bad_epoch_inds:
        dists_i = np.abs(i - good_epoch_inds)
        min_index = good_epoch_inds[np.argmin(dists_i)]
        slice_epochs[i,:] = slice_epochs[min_index,:]

    return slice_epochs

def subtract_gradient(slice_epochs, slice_inds, corrmat_thresh, data_l, window_sz=60, window_std=5):
    window = signal.gaussian(window_sz, std=window_std)
    window = np.reshape(window / np.sum(window),[window.shape[0],1])   
    smooth_epochs = np.zeros(slice_epochs.shape)        
    smooth_epochs[:,:] = signal.convolve2d(slice_epochs, window, boundary='wrap', mode='same')        
    subbed_epochs = slice_epochs - smooth_epochs
    
    subbed_ts = np.zeros(data_l)
    for i in np.arange(0,slice_epochs.shape[0]):
        subbed_ts[slice_inds[i,:]] = subbed_epochs[i,:] if corrmat_thresh[i] else 0

    return subbed_ts

def isolate_frequencies_old(data,midfreq,fs):
    spec1 = np.fft.fft(data) 
    spec2 = np.zeros(spec1.shape,dtype=np.complex_)
    spec2[0:] = spec1[0:] ; 
    
    length = data.shape[0]
    binsz = fs/length 
    lowCutIndex = int(midfreq/binsz)
    spec1[1:lowCutIndex] = 0
    spec2[lowCutIndex:] = 0

    highpass = np.real(np.fft.ifft(spec1))
    lowpass = np.real(np.fft.ifft(spec2))
    
    return highpass,lowpass 

def isolate_frequencies(data, midfreq, srate):
    # Get frequency domain signal
    freq_bins =  np.fft.fftfreq(data.shape[0], d=1/srate)
    freq_signal = np.fft.fft(data, n=len(data))

    # Filter signal
    low_freq_signal = freq_signal.copy()
    high_freq_signal = freq_signal.copy()
    zero_freq = copy.deepcopy(freq_signal[freq_bins == 0])
    high_freq_signal[(abs(freq_bins) < midfreq)] = 0
    low_freq_signal[(abs(freq_bins) > midfreq)] = 0
    low_freq_signal[freq_bins == 0] = zero_freq

    highpass = np.fft.ifft(high_freq_signal, n=len(data)).real
    lowpass = np.fft.ifft(low_freq_signal, n=len(data)).real

    return highpass, lowpass

# define function for finding start of TS given number of dummies (for aligning with FMRI)

def run(args=None, config=None):
    if not config:
        parser = AnalysisParser('config')
        args = parser.parse_analysis_args(args)
        config = args.config

    montage_path = config['montage_path']
    raw_vhdr = config['raw_vhdr']
    alignment_trigger_name = config['alignment_trigger_name']
    slice_gradient_similarity_correlation = config[
        'slice_gradient_similarity_correlation'
    ]
    output = config['output']

    debug_plot = False
    if 'debug-plot' in config:
        debug_plot = config['debug-plot']

    root, fname = os.path.split(montage_path)

    montage = mne.channels.read_montage(montage_path)
    raw = mne.io.read_raw_brainvision(
        raw_vhdr,
        montage=montage,
        eog=['ECG','ECG1']
    )

    tmpgraddata = raw.get_data()
    events = mne.find_events(raw)
    print(events)

    plt.figure()
    plt.plot(tmpgraddata[3,:])
    plt.title("Channel 3 before subtraction")

    # Get alignment trigger
    alignment_latency = 0
    for latency, _, name in events:
        if name == alignment_trigger_name:
            alignment_latency = latency

    # Shift graddata to alignment
    graddata = np.zeros((tmpgraddata.shape[0], tmpgraddata.shape[1]-alignment_latency))
    for t in range(tmpgraddata.shape[0]):
        graddata[t, :] = tmpgraddata[t, alignment_latency:]

    slice_gap = get_slicegap(graddata[3,:])
    slice_epochs, slice_inds = get_slicepochs(graddata[0,:], slice_gap)

    good_epoch_inds, bad_epoch_inds, corrmat_thresh = find_bad_slices(
        slice_epochs,
        corrthresh=slice_gradient_similarity_correlation
    )

    error_msg = \
        'ERROR: Cannot find any gradients that are sufficiently similar to each other.' +\
        'Try decreasing the value of `slice_gradient_similarity_correlation`.'
    assert any(corrmat_thresh), error_msg


    log.info("Epoching slices...total runs: {}".format(graddata.shape[0]))
    for i in np.arange(0,graddata.shape[0]):
        write_same_line(str(i+1) + "/{}".format(graddata.shape[0]))
        highpass, lowpass = isolate_frequencies(graddata[i,:], 2, raw.info['sfreq'])

        if debug_plot:
            plt.figure()
            fft = np.abs(np.fft.fft(highpass[:]))
            plt.plot(fft)
            plt.title("FFT - High frequencies before gradient subtraction")

        slice_epochs, slice_inds = get_slicepochs(highpass, slice_gap)

        if debug_plot:
            plt.figure()
            log.info(len(slice_epochs))
            plt.imshow(slice_epochs[500:1500])
            plt.title("Gradient epochs (portion of total)")

        slice_epochs = replace_bad_slices(slice_epochs, good_epoch_inds, bad_epoch_inds)
        graddata[i,:] = subtract_gradient(slice_epochs, slice_inds, 
                corrmat_thresh, graddata.shape[1]) + lowpass

        if debug_plot:
            plt.figure()
            fft = np.abs(np.fft.fft(graddata[i,50000:graddata.shape[1]-50000]))
            plt.plot(fft)
            plt.title("FFT - After gradient subtraction")

        if debug_plot:
            plt.show()

    finish_same_line()
    '''
    TODO: Output information about how to interpret the results.
    '''

    fname = os.path.join(
        output,
        'gradientremoved_' + str(int(time.time())) + raw_vhdr.split('/')[-1].split('.')[0] + '.fif'
    )

    newgraddata = np.zeros(tmpgraddata.shape)
    for t in range(tmpgraddata.shape[0]):
        newgraddata[t, :alignment_latency] = tmpgraddata[t,:alignment_latency]
        newgraddata[t, alignment_latency:] = graddata[t,:]

    new_raw = helpers.create_raw_mne(newgraddata, raw.ch_names, ['eeg' for _ in range(len(raw.ch_names))],
              mne.channels.read_montage(montage_path), sfreq=raw.info['sfreq']
    )
    new_raw.save(fname)

    plt.figure()
    fft = np.abs(np.fft.fft(newgraddata[3,50000:newgraddata.shape[1]-50000]))
    plt.plot(fft)

    plt.figure()
    plt.plot(graddata[3,:])
    plt.title("Channel 3 after subtraction")

    log.info("Close figures to end analysis.")
    plt.show()

