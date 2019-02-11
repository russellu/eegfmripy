import mne as mne
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ..cli import AnalysisParser

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

def subtract_gradient(slice_epochs, slice_inds, corrmat_thresh, data_l, window_sz=30, window_std=4):
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

def run(args=None, config=None):
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

    # define function for finding start of TS given number of dummies (for aligning with FMRI)

    montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_shared/')
    raw = mne.io.read_raw_brainvision(
            '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vhdr',
            montage=montage,eog=['ECG','ECG1'])

    graddata = raw.get_data()[0:64,:]




    wsize=15000
    single_channel = graddata[0,:] 
    mcorrs = np.zeros([wsize,np.int(single_channel.shape[0]/wsize)])
    icount = 0
    for i in np.arange(0, single_channel.shape[0] - wsize - 1, wsize):
        mcorrs[:,icount] = signal.correlate(single_channel[i:i+wsize],single_channel[i:i+wsize], mode='same')
        icount = icount + 1
        
    mcorrs = np.mean(mcorrs,axis=1)
    slice_gap = np.argmax(mcorrs[np.int(wsize/2)+50:]) + 50

    slice_epochs, slice_inds = get_slicepochs(graddata[0,:], slice_gap)

    # if scanner clock is not synchronized, get the offset
    wsize=2000
    timepoint_epochs = slice_epochs[:,215]
    lagcorrs = np.zeros([wsize,np.int(timepoint_epochs.shape[0]/wsize)])
    icount = 0
    for i in np.arange(0, timepoint_epochs.shape[0] - wsize - 1, wsize):
        lagcorrs[:,icount] = signal.correlate(timepoint_epochs[i:i+wsize],
                timepoint_epochs[i:i+wsize], mode='same')
        icount = icount + 1

    # manually find the clock offset
        
    badchans = []
    chans = np.zeros((graddata.shape[0]))
    chans[badchans] = 1
    goodchans = np.where(chans==0)[0]

    clock_offset = 40 
    for e in np.arange(0,1):
        highpass, lowpass = isolate_frequencies(graddata[e,:], 2, 5000)
        slice_epochs, slice_inds = get_slicepochs(highpass, slice_gap)
        new_chan = np.zeros(graddata.shape[1])
        # get the clock offset here
        for i in np.arange(0, clock_offset):
            good_epoch_inds, bad_epoch_inds, corrmat_thresh = find_bad_slices(
                    slice_epochs[i::clock_offset,:], corrthresh=0.9)
            
            if good_epoch_inds.shape[0] != 0: 
            
                short_slice_epochs = replace_bad_slices(
                        slice_epochs[i::clock_offset,:], good_epoch_inds, bad_epoch_inds)    
                
                mean_epochs = np.mean(short_slice_epochs,axis=0)
                all_peaks = np.unique(np.argmax(short_slice_epochs, axis=1))
                            
                subbed = subtract_gradient(
                        short_slice_epochs, slice_inds[i::clock_offset,:], 
                        corrmat_thresh, graddata.shape[1])
                
                # interpolate subbed at the max peak
                current_inds = slice_inds[i::clock_offset,:]
                for ind in np.arange(0,current_inds.shape[0]):
                    subbed[current_inds[ind,all_peaks]] = np.tile((
                        subbed[current_inds[ind,np.max(all_peaks)+1]] 
                        + subbed[current_inds[ind,np.min(all_peaks)-1]]) / 2,
                    (np.shape(all_peaks)))
                    
                
                new_chan = new_chan + subbed
            
        graddata[e,:] = new_chan + lowpass

            
    #f = np.abs(np.fft.fft(new_chan[500000:new_chan.shape[0]-500000]))      
    #plt.plot(f[1:10000000])
            
    #slice_gap = get_slicegap(graddata[3,:])
    """
    for i in np.arange(0,graddata.shape[0]):
        highpass, lowpass = isolate_frequencies(graddata[i,:], 2, 5000)
        slice_epochs, slice_inds = get_slicepochs(highpass, slice_gap)
        slice_epochs = replace_bad_slices(slice_epochs, good_epoch_inds, bad_epoch_inds)
        graddata[i,:] = subtract_gradient(slice_epochs, slice_inds, 
                corrmat_thresh, graddata.shape[1]) + lowpass

    fft = np.abs(np.fft.fft(graddata[3,50000:graddata.shape[1]-50000]))
    plt.plot(fft)
    """









