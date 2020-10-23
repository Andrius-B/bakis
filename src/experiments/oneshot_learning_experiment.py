from src.runners.abstract_runner import AbstractRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.models.res_net_akamaster import *

class CecOneshotTrainingExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            'dataset_name': 'cifar-100',
            'epochs': '50',
            'num_classes': '100',
            'measurements': 'loss, accuracy',
        }

    def run(self):
        run_params = super().get_run_params()
        num_classes = int(run_params.getd('num_classes', '100'))
        for i in range(15):
            net = resnet56(ceclustering=True, num_classes=num_classes, init_radius=0.2, ce_n_dim=15)
            runner = AbstractRunner(net, run_params, tensorboard_prefix='cec-oneshot')
            runner.train()

    def help_str(self):
        return """This experiment train for multiple epochs on 99 classes of cifar100 and then introduces the final 
        class with only showing the images once"""