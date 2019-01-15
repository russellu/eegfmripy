import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import spatial
import mne as mne
from scipy import stats

from ..cli import AnalysisParser


def epoch_gradient_slices(single_channel, slicegap):
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

def epoch_timeseries(timeseries, window_length):
    epochs = np.zeros([int(timeseries.shape[0]/window_length), window_length])
    icount = 0
    for i in np.arange(0, timeseries.shape[0]-window_length, window_length):
        epochs[icount,:] = timeseries[i:i+window_length]
        icount = icount + 1 
        
    return epochs

def fit_func(x, a):
    return a*x

def get_offset(slice_epochs):
    peak_grad = np.argmax(np.diff(np.mean(slice_epochs,axis=0)))
    shift_ts = slice_epochs[:,peak_grad]
    shift_epoch = epoch_timeseries(shift_ts, slice_gap*5)
    shift_xcorrs = np.zeros(shift_epoch.shape)
    for i in np.arange(0,shift_epoch.shape[0]):
        shift_xcorrs[i,:] = signal.correlate(shift_epoch[i,:],shift_epoch[i,:],mode='same')
    
    mean_xcorrs = np.mean(shift_xcorrs,axis=0)   
    maxinds = np.where(np.r_[True, mean_xcorrs[1:] > mean_xcorrs[:-1]] 
            & np.r_[mean_xcorrs[:-1] > mean_xcorrs[1:], True])
       
    offset = int(np.median(np.diff(maxinds)))
    
    return offset 

def remove_gap(new_ts,slice_inds):
    for i in np.arange(1,slice_inds.shape[0]):
        diff = new_ts[slice_inds[i,0]] - new_ts[slice_inds[i-1,slice_inds.shape[1]-1]]
        new_ts[slice_inds[i,:]] = new_ts[slice_inds[i,:]] - diff
        
    return new_ts

def average_artifact_subtraction(slice_epochs, slice_inds, offset):
    new_ts = np.zeros(single_channel.shape)
    for i in np.arange(0,offset):
        slices_i = slice_epochs[i::offset,:]
        inds_i = slice_inds[i::offset,:]
        dists = spatial.distance.pdist(slices_i,'euclidean')
        dists = spatial.distance.squareform(dists)
        
        subbed = np.zeros(slices_i.shape)
        for j in np.arange(0,subbed.shape[0]):
            sorted_inds = np.argsort(dists[j,:])
            avg_artifact = np.mean(slices_i[sorted_inds[1:40],:],axis=0)
            subbed[j,:] = slices_i[j,:] - avg_artifact
            new_ts[inds_i[j,:]] = subbed[j,:]  
        print(i/40)
        
    return new_ts

def interpolate_saturated_points(slice_epochs, slice_inds, new_ts, npoints=4):
    interps = np.flip(np.argsort(np.mean(slice_epochs,axis=0)))
    for i in np.arange(0,slice_inds.shape[0]):
            badinds = slice_inds[i,interps[0:npoints]]
            min_i = new_ts[badinds[np.argmin(badinds)]-1]
            max_i = new_ts[badinds[np.argmax(badinds)]+1]
            new_ts[badinds] = np.tile((min_i/2) +(max_i/2),npoints)
            
    return new_ts

def get_slice_timing(single_channel, wsize=15000):
    mcorrs = np.zeros([wsize,np.int(single_channel.shape[0]/wsize)])
    icount = 0
    for i in np.arange(0, single_channel.shape[0] - wsize - 1, wsize):
        mcorrs[:,icount] = signal.correlate(
                single_channel[i:i+wsize],single_channel[i:i+wsize], mode='same')
        icount = icount + 1
    
    mcorrs = np.mean(mcorrs,axis=1)
    slice_gap = np.argmax(mcorrs[np.int(wsize/2)+50:]) + 50
    
    return slice_gap


def run(args=None, config=None):
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

    #dat0 = np.load('/media/sf_shared/graddata/graddata_0.npy')
    #dat0 = dat0[0:10000000]

    from ..tests.TestsEEGFMRI import example_raw
    raw = example_raw()

    graddata = raw.get_data()[0:64,:]

    for gd in np.arange(0,graddata.shape[0]):
        dat0 = graddata[gd,:]
        hp, lp = isolate_frequencies(dat0,2,5000)
        single_channel = hp
        slice_gap = get_slice_timing(single_channel)
        slice_epochs, slice_inds = epoch_gradient_slices(hp, slice_gap)
        offset = get_offset(slice_epochs)
        
        new_ts = average_artifact_subtraction(slice_epochs, slice_inds, offset)
        new_ts = remove_gap(new_ts, slice_inds)
        new_ts = isolate_frequencies(new_ts,2,5000)[0]
        new_ts = interpolate_saturated_points(slice_epochs, slice_inds, new_ts)
        new_ts = new_ts + lp
        graddata[gd,:] = new_ts 
        
        print(gd)


    chan1 = signal.decimate(graddata[0,:],20)
    dec_chans = np.zeros([graddata.shape[0],chan1.shape[0]])
    for i in np.arange(1,graddata.shape[0]):    
        dec_chans[i,:] = signal.decimate(graddata[i,:],20)
        print(i)

    np.save('dec_chans',dec_chans)






