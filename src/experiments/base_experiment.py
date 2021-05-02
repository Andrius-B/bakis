from typing import Dict
from abc import ABC, abstractmethod

from src.runners.run_parameters import RunParameters


class BaseExperiment(ABC):

    def __init__(self, parameter_overrides: Dict[str, str]):
        self.parameter_overrides = parameter_overrides

    def get_run_params(self):
        # the order here is important:
        #   * we take the default run parameters
        #   * override to the experiment specific defaults (if any)
        #   * override to the run specific (if provided)
        params = RunParameters('cifar-10')
        params.apply_overrides(self.get_experiment_default_parameters())
        params.apply_overrides(self.parameter_overrides)
        return params

    @abstractmethod
    def get_experiment_default_parameters(self) -> Dict[str, str]:
        return None

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def help_str():
        return "BaseExperiment class should never be instantiated!"
