import os.path as op

import numpy as np
from mayavi import mlab

import mne
from mne.datasets import sample

from ..cli import AnalysisParser


def run(args=None, config=None):
	print(__doc__)

	data_path = sample.data_path()
	subjects_dir = op.join(data_path, 'subjects')
	raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
	trans_fname = op.join(data_path, 'MEG', 'sample',
	                      'sample_audvis_raw-trans.fif')
	raw = mne.io.read_raw_fif(raw_fname)
	trans = mne.read_trans(trans_fname)
	src = mne.read_source_spaces(op.join(subjects_dir, 'sample', 'bem',
	                                     'sample-oct-6-src.fif'))