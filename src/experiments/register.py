from typing import Dict
import logging
from .cec_vs_linear_experiment import CECvsLinearExperiment
from .hyperparameter_searcher_experiment import HyperParameterSearcherExperiment
from .oneshot_learning_experiment import CecOneshotTrainingExperiment
from .tester import Tester
from .disk_ds_learner import DiskDsLearner
from .window_to_spectro import WindowToSpectro
from .sqlite_ds_learner import SQLiteDsLearner

logger = logging.getLogger(__name__)

class ExperimentRegistry:
    def __init__(self):
        # Add experiments here so that the runner can pick them up:
        self.experiments = {
            'cec-lin-cifar': CECvsLinearExperiment,
            'hyper-searcher': HyperParameterSearcherExperiment,
            'oneshot': CecOneshotTrainingExperiment,
            'tester': Tester,
            'disk-ds-learner': DiskDsLearner,
            'spectro-precompute': WindowToSpectro,
            'sqlite-ds-learner': SQLiteDsLearner
        }

    def run_experiment(self, experiment_name: str, parameter_overrides: Dict[str, str]):
        if experiment_name not in self.experiments:
            logger.error(f"Experiment `{experiment_name}` not found in the registry,\n"
            f"Did you mean one of: {', '.join(self.get_experiment_names())}""")
            return
        experiment = self.experiments[experiment_name](parameter_overrides)
        experiment.run()
    
    def get_experiment_names(self):
        return [x for x in self.experiments]