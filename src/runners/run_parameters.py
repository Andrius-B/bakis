from typing import Dict
import logging
from .dataset_provider import DatasetProvider

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
            'dataset_name': dataset_name
        }
        self.defaulted_params = {}

    def set(self, key: str, value: str):
        self.all_params[key] = value

    def get(self, key) -> str:
        if key in self.all_params:
            return self.all_params[key] 
        raise Exception(f"No such param found and no default provided for: {key}")

    def getd(self, key, default) -> str:
        if key in self.all_params:
            return self.all_params[key] 
        self.defaulted_params[key] = default
        return default
    
    def get_dataset_name(self):
        return self.all_params['dataset_name']

    def validate_params(self):
        if(self.all_params['training_validation_mode'] not in ['batch', 'epoch', 'finished', 'never']):
            raise Exception(f'invalid training validation mode: {self.all_params["training_validation_mode"]}')
        assert self.all_params['lr'] != None
    
    def apply_overrides(self, overrides: Dict[str, str]):
        if overrides == None:
            logger.warn("Tried to apply None parameter overrides!")
            return
        for param in overrides:
            if param in self.all_params:
                logger.info(f"""Overriding run parameter {param}:
                \t{self.all_params[param]} -> {overrides[param]}""")
                self.all_params[param] = overrides[param]
            else:
                logger.warning(f"Applying an unkown override {param}: {overrides[param]}")
                self.all_params[param] = overrides[param]