import logging
from src.runners.sqlite_audio_runner import SQliteAudioRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.models.mel_2048_conv2d import net
from src.runners.run_parameter_keys import R
from src.datasets.dataset_provider import DatasetProvider

class SQLiteDsLearner(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            # R.DATASET_NAME: 'disk-ds(/home/andrius/git/bakis/data/spotifyTop10000)',
            R.DATASET_NAME: 'sqlite-ds(/home/andrius/git/bakis/test.db)',
            R.BATCH_SIZE_TRAIN: '10',
            R.EPOCHS: '1',
            R.BATCH_SIZE_VALIDATION: '5'
        }

    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        runner = SQliteAudioRunner(net, run_params, tensorboard_prefix='diskds')
        log.info("Runner initialized, starting train")
        runner.train()
        

    def help_str(self):
        return """Tries to teach a simple cec resnet for classes read from disk"""