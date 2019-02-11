import mne

from ..cli import AnalysisParser
from ..run_bcg_denoise import remove_bcg
from ..remove_gradient_single import remove_gradient


def run(args=None, config=None):
    if not config:
        parser = AnalysisParser('config')
        args = parser.parse_analysis_args(args)
        config = args.config

    montage_path = config['montage_path']
    gradrem_path = config['raw_vhdr']
    output = config['output']
    display_ica_components = config['display_ica_components']
    alignment_trigger_name = config['alignment_trigger_name']
    slice_gradient_similarity_correlation = config[
        'slice_gradient_similarity_correlation'
    ]
    output = config['output']

    debug_plot = False
    if 'debug-plot' in config:
        debug_plot = config['debug-plot']

    bcg_peak_widths = list(range(25,45))
    if 'bcg_peak_widths' in config:
        bcg_peak_widths = config['bcg_peak_widths']

    raw = mne.io.read_raw_fif(gradrem_path)
    raw.load_data()
    tmpdata = raw.get_data()
    events = mne.find_events(raw)
    print(events)

    raw = remove_gradient(
        raw,
        alignment_trigger_name=alignment_trigger_name,
        slice_gradient_similarity_correlation=slice_gradient_similarity_correlation,
        output=output,
        debug_plot=debug_plot
    )

    for t in range(raw.get_data().shape[0]):
        raw[t,:1790] = 0

    raw = remove_bcg(
        raw,
        montage_path,
        output=output,
        debug_plot=debug_plot,
        bcg_peak_widths=bcg_peak_widths,
        display_ica_components=display_ica_components
    )
