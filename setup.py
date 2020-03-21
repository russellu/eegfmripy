import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

PACKAGE_VERSION = '0.1.0'
DESC = "Package of processing tools for simultaneous EEG-fMRI recordings."
with open(os.path.join(here, 'README.md')) as fh:
    README = fh.read()

DEPS = [
    'mne',
    'numpy',
    'matplotlib',
    'ruamel.yaml',
    'scipy'
]

setup(
    name='eegfmripy',
    version=PACKAGE_VERSION,
    description=DESC,
    long_description=README,
    keywords='eeg,fmri,eeg-fmri',
    author='Russell Butler, Gregory Mierzwinksi',
    author_email='',
    url='https://github.com/russellu/eegfmripy',
    license='BSD',
    packages=[
        'eegfmripy',
        'eegfmripy.clianalysis',
        'eegfmripy.utils'
    ],
    include_package_data=True,
    install_requires=DEPS,
    entry_points="""
    # -*- Entry points: -*-
    [console_scripts]
    eegfmripy = eegfmripy.cli:cli
    """,
)