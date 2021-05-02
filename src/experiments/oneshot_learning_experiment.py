from src.runners.abstract_runner import AbstractRunner
from src.experiments.base_experiment import BaseExperiment
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.models.res_net_akamaster import *


class CecOneshotTrainingExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'cifar-100',
            R.EPOCHS: '50',
            R.NUM_CLASSES: '100',
            R.MEASUREMENTS: 'loss, accuracy',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: '1e-3'
        }

    def run(self):
        run_params = super().get_run_params()
        num_classes = int(run_params.getd(R.NUM_CLASSES, '100'))
        for i in range(15):
            net = resnet56(ceclustering=True, num_classes=num_classes, init_radius=0.2, ce_n_dim=15)
            runner = AbstractRunner(net, run_params, tensorboard_prefix='cec-oneshot')
            runner.train()

    @staticmethod
    def help_str():
        return """This experiment train for multiple epochs on 99 classes of cifar100 and then introduces the final 
        class with only showing the images once"""
