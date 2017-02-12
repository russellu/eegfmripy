import os
import time
import mne
import logging

from ..cli import AnalysisParser
from ..clianalysis.run_bcg_denoise import remove_bcg
from ..clianalysis.remove_gradient_single import remove_gradient

log = logging.getLogger("eegfmripy")


def run(args=None, config=None):
    if not config:
        parser = AnalysisParser('config')
        args = parser.parse_analysis_args(args)
        config = args.config

    montage_path = config['montage_path']
    data_path = config['data_path']
    output = config['output']
    display_ica_components = config['display_ica_components']
    alignment_trigger_name = config['alignment_trigger_name']
    slice_gradient_similarity_correlation = config[
        'slice_gradient_similarity_correlation'
    ]
    output = config['output']

    gradrem_path = None
    if 'gradrem_path' in config:
        gradrem_path = config['gradrem_path']

    debug_plot = False
    if 'debug-plot' in config:
        debug_plot = config['debug-plot']

    debug_plot_verbose = False
    if 'debug-plot-verbose' in config:
        debug_plot_verbose = config['debug-plot-verbose']

    bcg_peak_widths = list(range(25,45))
    if 'bcg_peak_widths' in config:
        bcg_peak_widths = eval(config['bcg_peak_widths'])
    print(bcg_peak_widths)

    log.info("Reading data...")

    def open_data(data_path):
        if not data_path:
            raise Exception("Required data path 'raw_vhdr' not supplied.")
        if data_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(data_path)
        elif data_path.endswith('.vhdr'):
            montage = mne.channels.read_montage(montage_path)
            raw = mne.io.read_raw_brainvision(
                data_path,
                montage=montage,
                eog=['ECG','ECG1']
            )
        elif data_path.endswith('.set'):
            montage = mne.channels.read_montage(montage_path)
            raw = mne.io.read_raw_eeglab(
                data_path,
                montage=montage,
                eog=['ECG', 'ECG1']
            )
        else:
            raise Exception("Unsupported EEG file format given. Supported types: .fif, .vhdr")
        return raw

    if not gradrem_path:
        log.info("Opening uncleaned data...")
        raw = open_data(data_path)

        raw.load_data()
        tmpdata = raw.get_data()
        events = mne.find_events(raw)
        print(events)

        log.info("Running gradient removal...")
        raw = remove_gradient(
            raw,
            alignment_trigger_name=alignment_trigger_name,
            slice_gradient_similarity_correlation=slice_gradient_similarity_correlation,
            output=os.path.join(
                output,
                'gradientremoved_' + str(int(time.time())) + data_path.split('/')[-1].split('.')[0] + '.fif'
            ),
            debug_plot=debug_plot,
            debug_plot_verbose=debug_plot_verbose
        )
    else:
        log.info("Opening gradient removed data...")
        raw = open_data(gradrem_path)
        raw.load_data()

    events = mne.find_events(raw)
    print(events)

    alignment_latency = 0
    for latency, _, name in events:
        if name == alignment_trigger_name:
            alignment_latency = latency

    for t in range(raw.get_data().shape[0]):
        raw[t,:alignment_latency] = 0

    log.info("Running BCG removal...")
    raw = remove_bcg(
        raw,
        montage_path,
        output=os.path.join(
            output,
            'bcgngradremoved_' + str(int(time.time())) + data_path.split('/')[-1].split('.')[0] + '.fif'
        ),
        debug_plot=debug_plot,
        debug_plot_verbose=debug_plot_verbose,
        bcg_peak_widths=bcg_peak_widths,
        display_ica_components=display_ica_components
    )
