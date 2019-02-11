import mne as mne 
import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np 
import nibabel as nib
from scipy import signal
from nilearn.decomposition import CanICA
import sys as sys
from ..utils import helpers
from scipy import stats
from scipy.stats import gamma
from skimage import transform

"""
functions to merge ICA denoised EEG data with BOLD ICA components

"""
from scipy.signal import butter, lfilter

from ..cli import AnalysisParser

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_hrf(times):
    peak_values = gamma.pdf(times, 6)
    undershoot_values = gamma.pdf(times, 12)
    values = peak_values - 0.35 * undershoot_values
    return values / np.max(values) * 0.6

def filter_and_downsample(raw_data, comps, freqs, start_ind, end_ind):
    chan_freqs = np.zeros((raw_data.shape[0], freqs.shape[0], comps.shape[1]))
    for i in np.arange(0,freqs.shape[0]):
        filt = butter_bandpass_filter(raw_data, freqs[i,0],freqs[i,1], 250)
        filt = np.abs(filt)
        gauss_filt = np.zeros(filt.shape)
        for j in np.arange(0,raw_data.shape[0]):
            smooth_filt = signal.convolve(filt[j,:],gauss,mode='same')
            gauss_filt[j,:] = smooth_filt[0:raw_data.shape[1]]
        
        gauss_filt = transform.resize(
                gauss_filt[:,start_ind:end_ind],[filt.shape[0],comps.shape[1]])
        
        chan_freqs[:,i,:] = gauss_filt
        print(i)
    
    return chan_freqs

def convolve_chanfreqs(chan_freqs, hrf):
    conved = np.zeros(chan_freqs.shape)
    for i in np.arange(0,chan_freqs.shape[0]):
        for j in np.arange(0,chan_freqs.shape[1]):
            conved_ij = signal.convolve(chan_freqs[i,j,:], hrf, mode='full')
            conved[i,j,:] = conved_ij[0:chan_freqs.shape[2]]
            
    return conved

def run(args=None, config=None):
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

    eeg_path = '/media/sf_shared/graddata/ica_denoised_raw.fif' 
    fmri_path = '/media/sf_shared/CoRe_011/rfMRI/d2/11-BOLD_Sleep_BOLD_Sleep_20150824220820_11.nii'
    vmrk_path = '/media/sf_shared/CoRe_011/eeg/CoRe_011_Day2_Night_01.vmrk'
    event_ids, event_lats = helpers.read_vmrk(vmrk_path)

    event_lats = np.array(event_lats)
    grad_inds = [index for index, value in enumerate(event_ids) if value == 'R1']
    grad_inds = np.array(grad_inds)
    grad_lats = event_lats[grad_inds]
    grad_lats = grad_lats / 20 # resample from 5000Hz to 250Hz
    start_ind = int(grad_lats[0])
    end_ind = int(grad_lats[-1])

    canica = CanICA(n_components=40, smoothing_fwhm=6.,
                    threshold=None, verbose=10, random_state=0)

    fmri = nib.load(fmri_path)
    # get TR, n_slices, and n_TRs
    fmri_info = helpers.fmri_info(fmri_path)
    canica.fit(fmri)
    cimg = canica.components_img_.get_data()
    TR = fmri_info[0]
    tr_times = np.arange(0, 30, TR)
    hrf = get_hrf(tr_times)



    # plot components 
    for i in np.arange(0,40):
        plt.subplot(4,10,i+1)
        plt.imshow(np.max(cimg[:,:,:,i],axis=2))

    # get the EEG
    raw = mne.io.read_raw_fif(eeg_path,preload=True)
    raw_data = raw.get_data()
        
    # get power spectrum for different sleep stages (BOLD)
    comps = canica.transform([fmri])[0].transpose()
    bold_srate = 1/fmri_info[0]
    bold_epochl = int(7500 / (250/bold_srate))

    #bold_pxx,bold_f = pxx_bold_component_epoch(comps, bold_srate, 250, bold_epochl, sleep_stages)
    #eeg_pxx,eeg_f = pxx_eeg_epochs(raw_data, sleep_stages, 7500)

    # concatenate the epochs, then compute the psd 
    # 1) get triggers, 2) concatenate data, 3) compute psd 

    def get_trigger_inds(trigger_name, event_ids):
        trig_inds = [index for index, value in enumerate(event_ids) if value == trigger_names[trig]]
        return trig_inds

    def epoch_triggers(raw_data, lats, pre_samples, post_samples):
        epochs = np.zeros((raw_data.shape[0], lats.shape[0], pre_samples + post_samples))
        for lat in np.arange(0,lats.shape[0]):
            epochs[:,lat,:] = raw_data[:,lats[lat]-pre_samples : lats[lat]+post_samples] 

        return epochs

    trigger_names = ['wake','NREM1','NREM2','NREM3']

    """
    epoch BOLD and get power for different trigger types 
    what you actually want is single trial EEG and BOLD psd

    first get all the indices that are contained within the BOLD timeseries
    then, get the EEG power spectrum values within those same indices 
    """
    eeg_srate = 250 
    bold_pre_samples = 15
    bold_post_samples = 25 
    eeg_pre_samples = int(bold_pre_samples*fmri_info[0]*eeg_srate)
    eeg_post_samples = int(bold_post_samples*fmri_info[0]*eeg_srate)
    bold_conversion = eeg_srate / (1/fmri_info[0])

    all_bold_epochs = []
    all_eeg_epochs = []
    for trig in np.arange(0, len(trigger_names)):
        trig_inds = get_trigger_inds(trigger_names[trig], event_ids)
        trig_lats = event_lats[trig_inds]
        bold_lats = ((trig_lats - start_ind) / bold_conversion).astype(int)    
        bads = np.where((bold_lats-bold_pre_samples<0) | (bold_lats+bold_post_samples>=comps.shape[1]))
        
        bold_lats = np.delete(bold_lats,bads,axis=0)
        eeg_lats = np.delete(trig_lats,bads,axis=0)
        
        bold_epochs = epoch_triggers(comps, bold_lats, bold_pre_samples, bold_post_samples)
        eeg_epochs = epoch_triggers(raw_data,eeg_lats,eeg_pre_samples, eeg_post_samples)

        all_bold_epochs.append(bold_epochs)
        all_eeg_epochs.append(eeg_epochs)

    # comput power
    for i in np.arange(0,len(all_eeg_epochs)):
        eeg_epochs = all_eeg_epochs[i]
        bold_epochs = all_bold_epochs[i]  
        bold_f, bold_pxx = signal.welch(bold_epochs)
        eeg_f, eeg_pxx = signal.welch(eeg_epochs)
       
    gauss = signal.gaussian(eeg_srate, 20)
    gauss = gauss/np.sum(gauss)

    freqs = np.zeros((5,2))
    freqs[0,0] = 1; freqs[0,1] = 3
    freqs[1,0] = 4; freqs[1,1] = 7
    freqs[2,0] = 8; freqs[2,1] = 15
    freqs[3,0] = 17; freqs[3,1] = 30
    freqs[4,0] = 30; freqs[4,1] = 80

    chan_freqs = filter_and_downsample(raw_data, comps, freqs, start_ind, end_ind)
    conved = convolve_chanfreqs(np.log(chan_freqs), hrf)

    # epoch all the hrf-convolved filtered EEG power 
    all_conved_epochs = []
    for trig in np.arange(0, len(trigger_names)):
        trig_inds = get_trigger_inds(trigger_names[trig], event_ids)
        trig_lats = event_lats[trig_inds]
        bold_lats = ((trig_lats - start_ind) / bold_conversion).astype(int)    
        bads = np.where((bold_lats-bold_pre_samples<0) | (bold_lats+bold_post_samples>=comps.shape[1]))    
        bold_lats = np.delete(bold_lats,bads,axis=0)
        
        conved_epochs = np.zeros(
                (conved.shape[0],conved.shape[1],bold_lats.shape[0],bold_pre_samples+bold_post_samples))
        for i in np.arange(0, conved.shape[1]):
            conved_epochs[:,i,:] = epoch_triggers(
                    conved[:,i,:], bold_lats, bold_pre_samples, bold_post_samples)

        all_conved_epochs.append(conved_epochs)


    sig1 = chan_freqs[3,2,:]
    sig2 = comps[0,:]
    sig2 = butter_bandpass_filter(sig2,0.005,0.1,1/fmri_info[0])
    nlags = 50


    def xcorr(sig1, sig2, nlags): 
        vec_l = sig1.shape[0] - nlags
        xcorrs = np.zeros(nlags)
        vec1 = sig1[int(sig1.shape[0]/2 - vec_l/2) : int(sig1.shape[0]/2+ vec_l/2)]
        start_p = 0
        for i in np.arange(0,nlags):
            vec2 = sig2[(start_p+i):(start_p+vec_l+i)]    
            xcorrs[i] = np.corrcoef(vec1,vec2)[0,1]
        
        return xcorrs
            


    all_xcorrs = []
    for i in np.arange(0,len(all_conved_epochs)):
        
        xc_i = np.zeros((1, all_conved_epochs[i].shape[1],
            all_conved_epochs[i].shape[2], all_bold_epochs[i].shape[0],20))
        
        for j in np.arange(0,1):
            print(j)
            for k in np.arange(0,all_conved_epochs[i].shape[1]):
                for el in np.arange(0,all_conved_epochs[i].shape[2]):
                        for m in np.arange(0,all_bold_epochs[i].shape[0]):
                            xc_i[j,k,el,m,:] = xcorr(
                                    all_conved_epochs[i][5,k,el,:],
                                    all_bold_epochs[i][m,el,:],20)
                            
        all_xcorrs.append(xc_i)

    plt.plot(np.mean(all_xcorrs[1][0,1,:,0,:],axis=0))
    plt.plot(np.mean(all_xcorrs[2][0,1,:,0,:],axis=0))
    plt.plot(np.mean(all_xcorrs[3][0,1,:,0,:],axis=0))

    # correlate power across different epochs 
    """
    r = np.zeros((bold_pxx.shape[0], bold_pxx.shape[2],eeg_pxx.shape[2]))
    for i in np.arange(0,bold_pxx.shape[0]):
        for j in np.arange(0,bold_pxx.shape[2]):
            for k in np.arange(0,eeg_pxx.shape[2]):
                r[i,j,k],p = stats.pearsonr(bold_pxx[i,:,j],eeg_pxx[0,:,k])
    """        
