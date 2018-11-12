# eegfmripy
python code for denoising eeg data from eeg-fmri experiments, and fusing eeg-fmri data

a fully automatic pipeline from raw EEG data acquired inside the scanner, to source
space EEG-BOLD correlations in each voxel.

the package performs the following in sequential order:

1) gradient artifact removal
2) ballistocardigram artifact removal
3) residual noise removal using ICA
4) source localization of cleaned scalp space signal
5) co-registration of source space with FMRI voxel space
6) frequency specific EEG-FMRI correlations in each voxel 

the package also estimates the following:
  1) time-varying heart rate trace based on frequency of EEG ballistocardiogram artifacts
  2) respiratory volume trace based on amplitude of EEG ballistocardiogram artifacts
  3) millisecond precise head motion amplitude trace based on broadband high frequency EEG power
  



