import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mne as mne
from scipy import stats
import sys as sys
from mne.preprocessing import ICA
import glob
import remove_bcg

from ..utils import helpers
from ..cli import AnalysisParser


def run(args=None, config=None):
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

    subs = ['CoRe_011', 'CoRe_023', 'CoRe_054', 'CoRe_079', 'CoRe_082', 'CoRe_087', 'CoRe_094', 'CoRe_100',
            'CoRe_107', 'CoRe_155', 'CoRe_192', 'CoRe_195', 'CoRe_220', 'CoRe_235', 'CoRe_267', 'CoRe_268']

    base_path = '/media/sf_hcp/sleepdata/'


    for sb_i in np.arange(0,len(subs)):
        sb = subs[sb_i]
        day1_eeg = glob.glob(base_path + sb + '/proc/*Day*1*_N*fif')
        day1_vmrk = glob.glob(base_path + sb + '/proc/*Day*1*_N*vmrk')
        day2_eeg = glob.glob(base_path + sb + '/proc/*Day*2*_N*fif')
        day2_vmrk = glob.glob(base_path + sb + '/proc/*Day*2*_N*vmrk')
      
        raw = mne.io.read_raw_fif(day1_eeg[0])
        raw.load_data()
        print(day1_eeg)
        print(day1_vmrk)
        
        #subbed_raw, heart_ts = run_bcg(raw)
        
        #subbed_raw.save(str.replace(day1_eeg[0],'.fif','_bcg.fif'))

        heartdata = remove_bcg.sort_heart_components(raw)
        
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
        
        #subbed_raw = remove_bcg.subtract_heartbeat_artifacts(
        #       raw.get_data(), shifted_epochs, shifted_inds)
        
        
        n_searches = 350
        n_avgs=30
        subbed_raw = raw.get_data().copy()
        raw_data = raw.get_data().copy()
        for epoch in np.arange(0,shifted_epochs.shape[1]):
            print(epoch)
            #rep_current = np.tile(shifted_epochs[:,[epoch],:],[1,shifted_epochs.shape[1],1])
            
            ind_vec = np.arange(0,shifted_epochs.shape[1])
            diff_vec = np.abs(ind_vec - epoch)
            sort_vec = np.argsort(diff_vec)
            
            
            rep_current = np.tile(shifted_epochs[:,[epoch],:],[1,n_searches,1])
            
            rep_diffs = rep_current - shifted_epochs[:,sort_vec[0:n_searches],:]
            
            mean_diffs = np.mean(np.abs(rep_diffs),axis=0)
            
            sorted_epochs = np.argsort(np.mean(mean_diffs,axis=1))
            
            subbed_raw[:,shifted_inds[epoch,:]]  = (raw_data[:,shifted_inds[epoch,:]] 
            - np.mean(shifted_epochs[:,sort_vec[sorted_epochs[1:n_avgs]]], axis=1))
          
       
        ch_names, ch_types, eeg_inds = helpers.prepare_raw_channel_info(
                subbed_raw, raw, mne.channels.read_montage(helpers.montage_path()), np.arange(0,63))
        
        new_raw = helpers.create_raw_mne(subbed_raw[:,:], ch_names, ch_types,
                  mne.channels.read_montage(helpers.montage_path()))
     
        new_raw.save(str.replace(day1_eeg[0],'.fif','_bcg.fif'),overwrite=True)
        
        
        """
        raw = mne.io.read_raw_fif(day2_eeg[0])
        raw.load_data()
        print(day2_eeg)
        print(day2_vmrk)

        save_path = '/media/sf_shared/graddata/bcg_denoised_raw.fif'
        
        new_raw.save(save_path)
        """
     

        """
        new_raw = raw.copy()
        
        raw.filter(1,120)
        ica = ICA(n_components=60, method='fastica', random_state=23)
        ica.fit(raw,decim=4)
        ica.plot_components(picks=np.arange(0,60))

        src = ica.get_sources(raw).get_data()
        """
    
    
    
    
    
    
    