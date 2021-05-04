import argparse
from typing import Callable
from src.config import Config
from src.experiments.register import ExperimentRegistry
from src.tools.resample_to_flacs import Resampler
from src.tools.random_copy import RandomCopy

experiment_registry = ExperimentRegistry()


def generate_experiments_help_message():
    experiments = ', '.join(experiment_registry.get_experiment_names())
    return f"Runnable experiments: {experiments}"


def configure_experiment_parser(parser):
    experiment_parser = parser.add_subparsers()
    experiment_parser.required = True
    experiment_parser.dest = 'experiment'
    for experiment in experiment_registry.get_experiment_names():
        experiment_help = experiment_registry.get_experiment_help(experiment)
        specific_parser = experiment_parser.add_parser(experiment, description=experiment_help)

        def run_experiment(args):
            experiment_registry.run_experiment(args.experiment, None)
        specific_parser.set_defaults(func=run_experiment)


def configure_resampler_parser(parser):
    resampler = Resampler()
    resampler.configure_argument_parser(parser)

    def run(args):
        resampler.run(args)
    parser.set_defaults(func=run)


def configure_copier_parser(parser):
    copier = RandomCopy()
    copier.configure_argument_parser(parser)

    def run(args):
        copier.run(args)
    parser.set_defaults(func=run)


if __name__ == "__main__":
    Config()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='type of run')

    # experiment parser
    experiment_parser = subparsers.add_parser("experiment", help="Run an experiment")
    configure_experiment_parser(experiment_parser)

    # resampler parser
    resampler_parser = subparsers.add_parser("resample", help="Run resampler")
    configure_resampler_parser(resampler_parser)

    # copier parser
    copier_parser = subparsers.add_parser("copy", help="Run random copying tool")
    configure_copier_parser(copier_parser)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_usage()
