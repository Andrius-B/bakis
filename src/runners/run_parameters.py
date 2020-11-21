from typing import Dict
import logging
from src.datasets.dataset_provider import DatasetProvider
from .run_parameter_keys import RunParameterKey, R

logger = logging.getLogger(__name__)

class RunParameters:
    """
    All the parameters that I can think of for a particular train/test run:
    ******************************************

    dataset_name - like cifar-10 or cifar-100

    training_validation_mode - one of 'batch', 'epoch', 'finished', 'never', 'sets'
        the validation mode during training

    test_with_one_sample - 'True' or 'False', tests the data with a single sample
        rather verboseley

    optimizer - one of 'adam', 'sgd', the type of optimizer to create

    lr - learing rate, should be small float, 1e-3 by default

    weight_decay - should be a small float, 0 by default

    loss - 'crossentropy', 'mse' - type of criterion to use

    epochs - number of epochs to run

    measurements - comma separated list of 'accuracy', 'loss', 'top-5-acc', 'top-10-acc'
        default 'loss'
    """
    def __init__(self, dataset_name):
        self.all_params = {
            R.DATASET_NAME: dataset_name
        }
        self.defaulted_params = {}

    def set(self, key: str, value: str):
        if not isinstance(key, RunParameterKey):
            raise Exception(f'Run parameter key is not an instance of RunParameterKey: {key}')
        self.all_params[key] = value

    def get(self, key) -> str:
        if not isinstance(key, RunParameterKey):
            raise Exception(f'Run parameter key is not an instance of RunParameterKey: {key}')
        if key in self.all_params:
            return self.all_params[key] 
        raise Exception(f"No such param found and no default provided for: {key}")

    def getd(self, key, default) -> str:
        if not isinstance(key, RunParameterKey):
            raise Exception(f'Run parameter key is not an instance of RunParameterKey: {key}')
        if key in self.all_params:
            return self.all_params[key] 
        self.defaulted_params[key] = default
        return default
    
    def get_dataset_name(self):
        return self.all_params[R.DATASET_NAME]

    def validate_params(self):
        logger.info("Validating parameters..")
        if(self.all_params[R.TRAINING_VALIDATION_MODE] not in ['batch', 'epoch', 'finished', 'never']):
            raise Exception(f'invalid training validation mode: {self.all_params[R.TRAINING_VALIDATION_MODE]}')
        assert self.all_params[R.LR] != None
        for k in self.all_params:
            if not isinstance(k, RunParameterKey):
                 raise Exception(f'Run parameter key is not an instance of RunParameterKey: {k} -> {self.all_params[k]}')
    
    def apply_overrides(self, overrides: Dict[RunParameterKey, str]):
        if overrides == None:
            logger.warn("Tried to apply None parameter overrides!")
            return
        for param in overrides:
            if not isinstance(param, RunParameterKey):
                raise Exception(f'Run parameter key is not an instance of RunParameterKey: {param}')
            if param in self.all_params:
                logger.info(f"""Overriding run parameter {param}:
                \t{self.all_params[param]} -> {overrides[param]}""")
                self.all_params[param] = overrides[param]
            else:
                logger.warning(f"Applying an unkown override {param}: {overrides[param]}")
                self.all_params[param] = overrides[param]