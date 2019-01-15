import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mne as mne
from scipy import stats
import sys as sys
from mne.preprocessing import ICA
sys.path.insert(0, '/media/sf_shared/eegfmripy/eegfmripy')
import helpers
import glob
import remove_bcg


subs = ['CoRe_011', 'CoRe_023', 'CoRe_054', 'CoRe_079', 'CoRe_082', 'CoRe_087', 'CoRe_094', 'CoRe_100',
        'CoRe_107', 'CoRe_155', 'CoRe_192', 'CoRe_195', 'CoRe_220', 'CoRe_235', 'CoRe_267', 'CoRe_268']

base_path = '/media/sf_hcp/sleepdata/'


for sb_i in np.arange(0,len(subs)):
    

    sb_i=16
    sb = subs[sb_i]
    day1_eeg = glob.glob(base_path + sb + '/proc/*Day*1*_N*bcg.fif')
    day1_vmrk = glob.glob(base_path + sb + '/proc/*Day*1*_N*vmrk')
    print(day1_eeg)
    
      
    raw = mne.io.read_raw_fif(day1_eeg[0])
    raw.load_data()
    
    orig_raw = raw.copy()
    orig_raw_2 = raw.copy() 
    
    raw.filter(1,90)
    ica = ICA(n_components=45, method='fastica', random_state=23)
    ica.fit(raw,decim=4)
    ica.plot_components(picks=np.arange(0,45))

    #src = ica.get_sources(raw).get_data()
    
    # select the hand-picked components, change indice in 'goods'
    
    goods = [4,5,8,9,10,11,12,13,17,19,20,25]
    

    raw = ica.apply(orig_raw, include=goods)
    plt.plot(orig_raw_2.get_data()[50,:])   
    plt.plot(raw.get_data()[50,:])
    


    raw.save(str.replace(day1_eeg[0],'.fif','_ica.fif'),overwrite=True)
    
    del raw, orig_raw, orig_raw_2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
