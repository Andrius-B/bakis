from src.runners.abstract_runner import AbstractRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.util.searcher_run_params_provider import SearcherRunParamsProviderBuilder
from src.models.res_net_akamaster import *
import time

class HyperParameterSearcherExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            'dataset_name': 'cifar-100',
            'epochs': '50',
            'num_classes': '100',
            'measurements': 'loss, accuracy',
            'ce_n_dim': '5',
            'ce_init_radius': '0.4'
        }

    def run(self):
        param_provider = (SearcherRunParamsProviderBuilder(super().get_run_params)
        .add_param('ce_n_dim', [1, 2, 3, 5, 6, 10, 15, 20])
        .add_param('ce_init_radius', [x/10 for x in range(1, 8)])
        .build())

        total_iterations = len(param_provider)
        start_time = time.time()

        for i, run_params in enumerate(param_provider):
            percent_complete = i/float(total_iterations)
            elapsed_time = time.time() - start_time
            if percent_complete > 0:
                remaining = elapsed_time/percent_complete - elapsed_time
                print(f"Elapsed time: {self.get_time_string(elapsed_time)}, remaining time (est.): {self.get_time_string(remaining)}")
            print(f"Run parameters: {run_params}")
            num_classes = int(run_params.get('num_classes'))
            n_dim = int(run_params.get('ce_n_dim'))
            init_radius = float(run_params.get('ce_init_radius'))
            net = resnet56(
                ceclustering=True,
                num_classes=num_classes,
                ce_n_dim=n_dim,
                init_radius=init_radius
                )
            runner = AbstractRunner(net, run_params, tensorboard_prefix='cec-search')
            runner.train()

    def help_str(self):
        return """Hyper parameter searcher experiment -- this uses a super naive grid
        search strategy, where all the provided parameters are tested"""

    def get_time_string(self, time):
        hours, rem = divmod(time, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)