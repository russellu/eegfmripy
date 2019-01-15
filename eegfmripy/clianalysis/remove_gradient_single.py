import mne as mne
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os 

from ..cli import AnalysisParser


def get_slicegap(single_channel, wsize=25000, maxcorr_pad=50):
    print("getting slice gap...")
    mcorrs = np.zeros([wsize,np.int(single_channel.shape[0]/wsize)])
    icount = 0
    for i in np.arange(0, single_channel.shape[0] - wsize - 1, wsize):
        mcorrs[:,icount] = signal.correlate(single_channel[i:i+wsize],single_channel[i:i+wsize], mode='same')
        icount = icount + 1
        
    mcorrs = np.mean(mcorrs,axis=1)
    return np.argmax(mcorrs[np.int(wsize/2)+maxcorr_pad:]) + maxcorr_pad

def get_slicepochs(single_channel, slicegap):
    print("epoching slices...")
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
    print("finding bad slices (slices without gradient artifacts)")
    corrmat = np.corrcoef(slice_epochs)
    mean_corrmat = np.mean(corrmat,axis=1)
    
    good_epoch_inds = np.where(mean_corrmat > corrthresh)[0]
    bad_epoch_inds = np.where(mean_corrmat <= corrthresh)[0]
    
    return good_epoch_inds, bad_epoch_inds, mean_corrmat > corrthresh

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
        if corrmat_thresh[i] == 1:
            subbed_ts[slice_inds[i,:]] = subbed_epochs[i,:]
        else:
            subbed_ts[slice_inds[i,:]] = 0
            
    return subbed_ts

def isolate_frequencies(data,midfreq,fs):
    spec1 = np.fft.fft(data) 
    spec2 = np.zeros(spec1.shape,dtype=np.complex_)
    spec2[0:] = spec1[0:] ; 
    
    length = data.shape[0]
    binsz = fs/length 
    lowCutIndex = int(midfreq/binsz)
    spec1[1:lowCutIndex] = 0 
    spec1[length-lowCutIndex:] = 0 
    spec2[lowCutIndex+1 : length-lowCutIndex-1] = 0 

    highpass = np.real(np.fft.ifft(spec1))
    lowpass = np.real(np.fft.ifft(spec2))
    
    return highpass,lowpass 

# define function for finding start of TS given number of dummies (for aligning with FMRI)

def run(args=None, config=None):
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

    montage_path = config['montage_path']
    raw_vhdr = config['raw_vhdr']

    root, fname = os.path.split(montage_path)

    montage = mne.channels.read_montage(fname, path=root)
    raw = mne.io.read_raw_brainvision(
        raw_vhdr,
        montage=montage,
        eog=['ECG','ECG1']
    )

    graddata = raw.get_data()

    slice_gap = get_slicegap(graddata[3,:])
    slice_epochs, slice_inds = get_slicepochs(graddata[0,:], slice_gap)
    good_epoch_inds, bad_epoch_inds, corrmat_thresh = find_bad_slices(slice_epochs, corrthresh=0.9)

    for i in np.arange(0,graddata.shape[0]):
        highpass, lowpass = isolate_frequencies(graddata[i,:], 2, 5000)
        slice_epochs, slice_inds = get_slicepochs(highpass, slice_gap)
        slice_epochs = replace_bad_slices(slice_epochs, good_epoch_inds, bad_epoch_inds)
        graddata[i,:] = subtract_gradient(slice_epochs, slice_inds, 
                corrmat_thresh, graddata.shape[1]) + lowpass

    '''
    TODO: Output information about how to interpret the results.
    '''

    fft = np.abs(np.fft.fft(graddata[3,50000:graddata.shape[1]-50000]))
    plt.plot(fft)
