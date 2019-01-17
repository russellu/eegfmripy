import nibabel as nib 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mne as mne
from scipy import stats
import sys as sys
from mne.preprocessing import ICA
import glob
from nilearn.decomposition import CanICA

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
        
        sb_i=1
        sb = subs[sb_i]
        day1_fmri = glob.glob(base_path + sb + '/proc/*nii')
        day1_vmrk = glob.glob(base_path + sb + '/proc/*Day*1*_N*vmrk')
        print(day1_fmri)
        
        fmri = nib.load(day1_fmri[0])
        
        canica = CanICA(n_components=40, smoothing_fwhm=6.,
                    threshold=None, verbose=10, random_state=0)
        
        
        fmri_info = helpers.fmri_info(day1_fmri[0])
        canica.fit(fmri)
        cimg = canica.components_img_.get_data()
        TR = fmri_info[0]
        tr_times = np.arange(0, 30, TR)
        hrf = helpers.get_hrf(tr_times)
        #        %matplotlib auto 

        for i in np.arange(0,40):
            plt.subplot(4,10,i+1)
            plt.imshow(np.max(cimg[:,:,:,i],axis=2))
            plt.title(str(i))
            
            
        # in order: DMN, auditory, visual, lingual, parietal, striatal, thalamic
        comps = [35,28,30,39,8,9,32]
        allcomp_ts = canica.transform([fmri])[0].transpose()
        comps_ts = allcomp_ts[comps,:]
        
        network_labs = ['DMN','auditory','visual','lingual',
                       'parietal','striatal','thalamic']
       
        for i in np.arange(0,len(comps)):
            plt.subplot(2,5,i+1)
            plt.imshow(np.max(cimg[:,:,:,comps[i]],axis=2))
            plt.title(network_labs[i])
            
        np.save(str.replace(day1_fmri[0],'.nii','_comps'), comps_ts)
      