import logging
from torchsummary import summary
from src.runners.audio_runner import AudioRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.models.res_net_akamaster_audio import *
from src.runners.run_parameter_keys import R
from src.datasets.dataset_provider import DatasetProvider

class DiskDsLearner(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/home/andrius/git/bakis/data/spotifyTop10000)',
            # R.DATASET_NAME: 'disk-ds(/home/andrius/git/searchify/resampled_music)',
            R.DISKDS_NUM_FILES: '100',
            R.NUM_CLASSES: '100',
            R.BATCH_SIZE_TRAIN: '50',
            R.EPOCHS: '10',
            R.BATCH_SIZE_VALIDATION: '50',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: '1e-3'
        }

    def run(self):
        
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        num_classes = int(run_params.getd(R.NUM_CLASSES, '10'))
        net = resnet20(ceclustering=False, num_classes=num_classes)
        net.to("cuda")
        summary(net, (1, 129, 129))

        runner = AudioRunner(net, run_params, tensorboard_prefix='diskds')
        log.info("Runner initialized, starting train")
        runner.train()
        

    def help_str(self):
        return """Tries to teach a simple cec resnet for classes read from disk"""