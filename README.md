# eegfmripy
Python code for denoising eeg data from eeg-fmri experiments, and fusing eeg-fmri data.

A fully automatic pipeline for cleaning raw EEG data acquired inside the scanner.

The package performs the following in sequential order:

1. gradient artifact removal
1. ballistocardigram artifact removal

The following features may be added in the future:
residual noise removal using ICA
1. EEG motion detection and censoring
1. Source localization of cleaned scalp space signal
1. Co-registration of source space with FMRI voxel space
1. Frequency specific EEG-FMRI correlations in each voxel 
1. Time-varying heart rate based on frequency of EEG ballistocardiogram artifacts
1. Respiratory volume based on amplitude of EEG ballistocardiogram artifacts
1. Millisecond precise head motion amplitude based on broadband high frequency EEG power
  
NB: due to overhead, the maximum size EEG dataset for a computer with 16 Gb RAM is 1.5Gb, this will typically only be a problem during the gradient denoising step

Dependencies can be installed with
```
cd eegfmripy
python setup.py install
```

These dependencies are the same as for mne-python (https://github.com/mne-tools/mne-python/blob/master/README.rst).

After installation, you can run `eegfmripy --help` for more info. The pipelines available can be found in the [clianalysis folder](https://github.com/russellu/eegfmripy/tree/dev/eegfmripy/clianalysis). Use the configuration YAML's in the configs folder for options that can be supplied to each script.

Example for the full clean - gradient artifact removal, followed by BCG artifact removal:
```
eegfmripy clean -c configs/config_clean.yml
```

