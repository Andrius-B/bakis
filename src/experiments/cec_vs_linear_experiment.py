from src.runners.abstract_runner import AbstractRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.models.res_net_akamaster import *

class CECvsLinearExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'cifar-100',
            R.EPOCHS: '50',
            R.NUM_CLASSES: '100',
            R.MEASUREMENTS: 'loss, accuracy',
        }

    def run(self):
        run_params = super().get_run_params()
        num_classes = int(run_params.getd('num_classes', '100'))
        for i in range(15):
            net = resnet20(ceclustering=True, num_classes=num_classes)
            runner = AbstractRunner(net, run_params, tensorboard_prefix='cec')
            runner.train()

        run_params = super().get_run_params()
        num_classes = int(run_params.getd('num_classes', '100'))
        for i in range(15):
            net = resnet20(ceclustering=False, num_classes=num_classes)
            runner = AbstractRunner(net, run_params, tensorboard_prefix='lin')
            runner.train()

    def help_str(self):
        return """This experiment runs a comparison: CEClustering vs Linear
        on resnet20 with only the classification part of the model altered.
        By default using 10 trials of 30 epochs for each model"""