import logging

from ..cli import AnalysisParser

log = logging.getLogger('eegfmripy')

def run(args=None, config=None):
	# 'config' is one argument group;
	# see `..cli` for other groups.
	parser = AnalysisParser('config')
	args = parser.parse_analysis_args(args)

	log.info("Hello world!")

	if 'config' in args:
		log.info('Found a config!')
		log.info('If it is correct, you will see True below.')
		log.info(True if 'hello' in args['config'] else False)
	else:
		log.info('Did not find a YAML config')