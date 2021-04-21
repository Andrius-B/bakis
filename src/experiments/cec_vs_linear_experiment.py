from src.runners.abstract_runner import AbstractRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.models.res_net_akamaster import *
from torchsummary import summary

class CECvsLinearExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'cifar-100',
            R.EPOCHS: '50',
            R.NUM_CLASSES: '100',
            R.MEASUREMENTS: 'loss, accuracy',
            R.LR: str(1e-3),
            R.TRAINING_VALIDATION_MODE: 'epoch',
        }

    def run(self):
        run_params = super().get_run_params()
        num_classes = int(run_params.getd(R.NUM_CLASSES, '100'))
        # for i in range(15):
        net = resnet32(ceclustering=False, num_classes=num_classes, massclustering=True)
        net.to("cuda")
        summary(net, (3, 32, 32))
        runner = AbstractRunner(net, run_params, tensorboard_prefix='cifar_mass')
        runner.train()

        run_params = super().get_run_params()
        num_classes = int(run_params.getd(R.NUM_CLASSES, '100'))
        # for i in range(15):
        net = resnet32(ceclustering=False, num_classes=num_classes)
        net.to("cuda")
        summary(net, (3, 32, 32))
        runner = AbstractRunner(net, run_params, tensorboard_prefix='cifar_lin')
        runner.train()

    def help_str(self):
        return """This experiment runs a comparison: CEClustering vs Linear
        on resnet20 with only the classification part of the model altered.
        By default using 10 trials of 30 epochs for each model"""