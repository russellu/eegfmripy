import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import spatial
import mne as mne
from scipy import stats
import nibabel as nib
from RemoveGradient import get_slice_epochs, get_slice_sample_len

from ..cli import AnalysisParser


def run(args=None, config=None):
	"""
	function: find the initial volume timing in the EEG signal, given the 
	following parameters: TR, # of slices, multiband factor, # of dummy volumes

	if these parameters are not known, you can also input the corresponding FMRI
	image, and the parameters will be obtained from nibabel (you do this as a check)
	if the number of dummies is not known, it is estimated based on the correlation
	structure of the intitial volumes 
	"""
	
    parser = AnalysisParser('config')
    args = parser.parse_analysis_args(args)
    config = args.config

	dat0 = np.load('/media/sf_shared/graddata/graddata_0.npy')
	dat0 = dat0[0:10000000]

	fmri = nib.load(
	        '/media/sf_shared/CoRe_011/rfMRI/d2/11-BOLD_Sleep_BOLD_Sleep_20150824220820_11.nii')
	hdr = fmri.get_header()


	TR = hdr.get_zooms()[3] 
	n_slices = hdr.get_n_slices()
	n_volumes = hdr.get_data_shape()[3]
	mb_factor = 1
	n_dummies = 3 

	n_slice_artifacts = (n_volumes * n_slices) / mb_factor

	from TestsEEGFMRI import example_raw 
	raw = example_raw()
	graddata = raw.get_data(picks=[1])
	graddata = np.squeeze(graddata)
	# the initial onset can only be found by using information from the average gradient
	# ie convolve with average gradient and compute offset? 
	slice_len = get_slice_sample_len(graddata) # slice length in samples
	slice_epochs, slice_inds = get_slice_epochs(graddata,slice_len)

	m_epoch = np.mean(slice_epochs,axis=0)

	conved = np.convolve((graddata[0:1000000]),m_epoch,mode='same')

	sort_desc = np.flip(np.argsort(conved))
	min_ind = np.min(sort_desc[0:10000]) - slice_len
	# use the projected number of slices and the start index to find the total length
	end_grad = min_ind + n_slice_artifacts*slice_len
	# find the number of slices after end_grad
	# use the slide window z-score approach - find the first outlier (out of 30+)




