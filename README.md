# eegfmripy
python code for denoising eeg data from eeg-fmri experiments, and fusing eeg-fmri data

a fully automatic pipeline from raw EEG data acquired inside the scanner, to source
space EEG-BOLD correlations in each voxel.

the package performs the following in sequential order:

1) gradient artifact removal
2) ballistocardigram artifact removal
3) residual noise removal using ICA
4) EEG motion detection and censoring
5) source localization of cleaned scalp space signal
6) co-registration of source space with FMRI voxel space
7) frequency specific EEG-FMRI correlations in each voxel 

the package also outputs the following regressors:
  1) time-varying heart rate based on frequency of EEG ballistocardiogram artifacts
  2) respiratory volume based on amplitude of EEG ballistocardiogram artifacts
  3) millisecond precise head motion amplitude based on broadband high frequency EEG power
  
  
NB: due to overhead, the maximum size EEG dataset for a computer with 16 Gb RAM is 1.5Gb, this will typically only be a problem during the gradient denoising step

Dependencies:

the dependencies of eegfmripy are the same as for mne-python (https://github.com/mne-tools/mne-python/blob/master/README.rst)


