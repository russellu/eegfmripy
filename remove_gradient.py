import mne as mne
import numpy as np
from scipy import signal
from scipy.interpolate import griddata
from scipy.cluster.vq import kmeans, vq
from scipy import signal
import matplotlib.animation as animation
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

def get_slicegap(graddata, wsize=25000, maxcorr_pad=50):
    print("getting slice gap...")
    mcorrs = np.zeros([wsize,np.int(graddata.shape[1]/wsize)])
    icount = 0
    for i in np.arange(0, graddata.shape[1] - wsize - 1, wsize):
        mcorrs[:,icount] = signal.correlate(graddata[2,i:i+wsize],graddata[2,i:i+wsize], mode='same')
        icount = icount + 1

    mcorrs = np.mean(mcorrs,axis=1)
    return np.argmax(mcorrs[np.int(wsize/2)+maxcorr_pad:]) + maxcorr_pad

def get_slicepochs(graddata, slicegap):
    print("epoching slices...")
    nepochs = np.int(graddata.shape[1] / slicegap)
    slice_epochs = np.zeros([graddata.shape[0], nepochs, slicegap])
    slice_inds = np.zeros([nepochs,slicegap])
    icount = 0
    for i in np.arange(0, graddata.shape[1]-slicegap, slicegap):
        slice_epochs[:,icount,:] = graddata[:,i:i+slicegap]
        slice_inds[icount,:] = np.arange(i,i+slicegap) 
        icount = icount + 1 
            
    slice_inds = slice_inds.astype(int)
    return slice_epochs, slice_inds

def remove_nongradient_epochs(slice_epochs, channel=3, corrthresh=0.9):
    print("replacing non-gradient epochs with closest gradient artifact")
    channel = 3 
    corrthresh = 0.9 
    corrmat = np.corrcoef(slice_epochs[channel,:])
    mean_corrmat = np.mean(corrmat,axis=1)
    
    grad_epochs = np.where(mean_corrmat > corrthresh)[0]
    non_grad_epochs = np.where(mean_corrmat <= corrthresh)[0]
    for i in non_grad_epochs:
        dists_i = np.abs(i - grad_epochs)
        min_index = grad_epochs[np.argmin(dists_i)]
        slice_epochs[:,i,:] = slice_epochs[:,min_index,:]

    return slice_epochs, mean_corrmat > corrthresh

def subtract_gradient(slice_epochs, slice_inds, corrmat_thresh, window_sz=60, window_std=5):
    print('smoothing and subtracting gradients...')
    window = signal.gaussian(window_sz, std=window_std)
    window = np.reshape(window / np.sum(window),[window.shape[0],1])   
    smooth_epochs = np.zeros(slice_epochs.shape)
    
    for i in np.arange(0,slice_epochs.shape[0]):
        chan_i = np.squeeze(slice_epochs[i,:,:])
        smooth_epochs[i,:,:] = signal.convolve2d(chan_i[:,:], window, boundary='wrap', mode='same')
        print(i)
    
    subbed_epochs = slice_epochs - smooth_epochs
    subbed_ts = np.zeros(graddata.shape)
    for i in np.arange(0,slice_epochs.shape[1]):
        if corrmat_thresh[i] == 1:
            subbed_ts[:,slice_inds[i,:]] = subbed_epochs[:,i,:]
        else:
            subbed_ts[:,slice_inds[i,:]] = 0
            
    return subbed_ts


# define function for finding start of TS given number of dummies (for aligning with FMRI)

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_E_DRIVE/')
raw = mne.io.read_raw_brainvision('/media/sf_E_DRIVE/badger_eeg/russell/retino_allstim_02.vhdr',montage=montage,eog=[31])

graddata = raw.get_data()[0:64,:]

slicegap = get_slicegap(graddata)
slice_epochs, slice_inds = get_slicepochs(graddata, slicegap)
slice_epochs, corrmat_thresh = remove_nongradient_epochs(slice_epochs)
subbed_ts = subtract_gradient(slice_epochs, slice_inds, corrmat_thresh)

fft = np.abs(np.fft.fft(subbed_ts[3,50000:subbed_ts.shape[1]-50000]))
plt.plot(fft)











