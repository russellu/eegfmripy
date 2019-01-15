import mne as mne
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import collections

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

def epoch_timeseries(timeseries, window_length):
    epochs = np.zeros([int(timeseries.shape[0]/window_length), window_length])
    icount = 0
    for i in np.arange(0, timeseries.shape[0]-window_length, window_length):
        epochs[icount,:] = timeseries[i:i+window_length]
        icount = icount + 1 
        
    return epochs

def fit_func(x, a):
    return a*x


def run(args=None, config=None):
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

    dat0 = np.load('/media/sf_shared/graddata/graddata_0.npy')
    dat0 = dat0[1:12000000]

    hp, lp = isolate_frequencies(dat0,2,5000)

    wsize=15000
    single_channel = hp 
    mcorrs = np.zeros([wsize,np.int(single_channel.shape[0]/wsize)])
    icount = 0
    for i in np.arange(0, single_channel.shape[0] - wsize - 1, wsize):
        mcorrs[:,icount] = signal.correlate(single_channel[i:i+wsize],single_channel[i:i+wsize], mode='same')
        icount = icount + 1
        
    mcorrs = np.mean(mcorrs,axis=1)
    slice_gap = np.argmax(mcorrs[np.int(wsize/2)+50:]) + 50
    slice_epochs, slice_inds = get_slicepochs(hp, slice_gap)

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
    new_ts = np.zeros(dat0.shape)
    avg_epochs = np.zeros(slice_epochs.shape)
    for ofst in np.arange(0,offset):   
        epochs_i = slice_epochs[ofst::40,:]       
        max_dist = 800
        if max_dist >= epochs_i.shape[0]:
            max_dist = epochs_i.shape[0] 
            
        sort_diffs = np.zeros([epochs_i.shape[0], max_dist])

        for i in np.arange(0,epochs_i.shape[0]):      
            closest_inds = np.where(np.abs(i - np.arange(0,epochs_i.shape[0])) < max_dist)[0]
            
            if closest_inds.shape[0] > max_dist:
                closest_inds = closest_inds[0:max_dist]
            
            smallest_ind = np.min(closest_inds)
            largest_ind = np.max(closest_inds)
            
            diffs_i = (np.tile(epochs_i[[i],:],(closest_inds.shape[0],1)) 
                    - epochs_i[smallest_ind:largest_ind+1,:])
            
            sumdiffs = np.sum(np.abs(diffs_i),axis=1)
            sort_diffs[i,:] = np.argsort(sumdiffs) + smallest_ind
              
        current_slice_inds = slice_inds[ofst::40,:]
        inds_i = np.arange(ofst,slice_epochs.shape[0],40)
        sort_diffs = sort_diffs.astype(int) 
        sub_epochs = np.zeros(epochs_i.shape)
        
        for i in np.arange(0, sort_diffs.shape[0]):
            template_i = np.mean(epochs_i[sort_diffs[i,1:50],:],axis=0)
            params = curve_fit(fit_func, template_i, epochs_i[i,:])
            fit_artifact = template_i*params[0][0] #+ params[0][1]
            fit_artifact = template_i
            sub_epochs[i,:] = epochs_i[i,:] - fit_artifact
            new_ts[current_slice_inds[i,:]] = sub_epochs[i,:]
            avg_epochs[inds_i[i],:] = template_i 
            
        print(ofst)

    temp_avgs = np.zeros(avg_epochs.shape)
    temp_avgs[0,:] = avg_epochs[0,:]
    diffs = np.zeros(avg_epochs.shape[0])
    subbed_epochs = np.zeros(avg_epochs.shape)
    new_ts2 = np.zeros(dat0.shape)    
    for i in np.arange(0,avg_epochs.shape[0]-1):
        new_ts2[slice_inds[i,:]] = slice_epochs[i,:] - avg_epochs[i,:]
        subbed_epochs[i,:] = slice_epochs[i,:] - avg_epochs[i,:]
        diffs[i] = new_ts2[slice_inds[i,0]] - new_ts2[slice_inds[i-1,slice_inds.shape[1]-1]]
        new_ts2[slice_inds[i,:]] = new_ts2[slice_inds[i,:]] - diffs[i]

    #plt.plot(new_ts2[1068850:1069000]); plt.plot(new_ts2[2873800:2874000]); 

    hp2,lp2 = isolate_frequencies(new_ts2,2,5000)
    plt.plot(hp2 + lp); plt.plot(new_ts + (np.mean(hp2) - np.mean(new_ts)))

    max_epochs = np.argmax(slice_epochs, axis=1)
    counts = collections.Counter(max_epochs)


    """
    vline = np.zeros(new_ts.shape)
    vline[slice_inds[:,0]] = 1
     
    new_ts2 = np.zeros(dat0.shape)    
    for i in np.arange(0,slice_inds.shape[0]):
        new_ts2[slice_inds[i,:]] = slice_epochs[i,:] - avg_epochs[i,:] - diffs[i]
        
    plt.plot(new_ts2[1068850:1069000]); plt.plot(new_ts2[2873800:2874000]); 
    """
        
    
    
