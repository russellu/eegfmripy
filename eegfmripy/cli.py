from __future__ import print_function, absolute_import

import importlib
import logging
import os
import sys
import ruamel.yaml as yaml
from argparse import ArgumentParser

here = os.path.abspath(os.path.dirname(__file__))

log = logging.getLogger('eegfmripy')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

ANALYSISTYPES_DIR = os.path.join(here, 'clianalysis')

ARGUMENT_GROUPS = {
	'config': [
		[['-c', '--config'],
		 {'required': True,
		  'dest': 'config',
		  'help': "Configuration YAML for analysis types.",
		  }],
	],
}
"""
These are commonly used arguments which can be re-used. They are shared to
provide a consistent CLI across the tools.
"""


class AnalysisParser(ArgumentParser):
	'''
		Used to parse ARGUMENT_GROUPS from within `analysistypes` files.
	'''
	arguments = []

	def __init__(self, *groups, **kwargs):
		ArgumentParser.__init__(self, **kwargs)

		for cli, kwargs in self.arguments:
			self.add_argument(*cli, **kwargs)

		for name in groups:
			group = self.add_argument_group("{} arguments".format(name))
			arguments = ARGUMENT_GROUPS[name]
			for cli, kwargs in arguments:
				group.add_argument(*cli, **kwargs)

	def parse_analysis_args(self, args):
		args = self.parse_args(args)

		# Convert YAML to python dict
		norm_config_path = os.path.abspath(args.config)
		if not os.path.exists(norm_config_path):
			log.info("Cannot find configuration YAML at: {}".format(norm_config_path))
			return None
		else:
			with open(norm_config_path, 'r') as f:
				args.config = yaml.safe_load(f)
		return args


def run_analysis(analysis, args):
	modname = '.clianalysis.{}'.format(analysis)
	mod = importlib.import_module(modname, package='eegfmripy')
	log.debug("Result:")
	mod.run(args=args)


def cli(args=sys.argv[1:]):
	parser = ArgumentParser()
	parser.add_argument('analysis-level', nargs='*', help="Analysis types to run.")
	parser.add_argument('-l', '--list', action='store_true', default=False,
						help="List available analysis types.")
	parser.add_argument('-v', '--verbose', action='store_true', default=False,
						help="Print debugging information.")
	args, remainder = parser.parse_known_args(args)

	if args.verbose:
		log.setLevel(logging.DEBUG)
	else:
		log.setLevel(logging.INFO)

	all_analysistypes = [
		os.path.splitext(p)[0] for p in os.listdir(ANALYSISTYPES_DIR)
		if p.endswith('.py') if p != '__init__.py'
	]

	if args.list:
		log.info('\n'.join(sorted(all_analysistypes)))
		return

	for analysis in args.analysistypes:
		if analysis not in all_analysistypes:
			log.error("analysis '{}' not found!".format(analysis))
			continue
		run_analysis(analysis, remainder)


if __name__ == '__main__':
	sys.exit(cli())
