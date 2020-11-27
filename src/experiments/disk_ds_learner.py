import logging
from torchsummary import summary
from src.runners.audio_runner import AudioRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.models.res_net_akamaster_audio import *
from src.runners.run_parameter_keys import R
from src.datasets.diskds.disk_dataset import DiskDataset
from src.datasets.dataset_provider import DatasetProvider
from src.datasets.diskds.ceclustering_model_loader import CEClusteringModelLoader

class DiskDsLearner(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/home/andrius/git/bakis/data/spotifyTop10000)',
            # R.DATASET_NAME: 'disk-ds(/home/andrius/git/searchify/resampled_music)',
            R.DISKDS_NUM_FILES: '10',
            R.NUM_CLASSES: '10',
            R.BATCH_SIZE_TRAIN: '50',
            R.EPOCHS: '150',
            R.BATCH_SIZE_VALIDATION: '50',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: '1e-3'
        }

    def run(self):
        ce_clustering_loader = CEClusteringModelLoader()
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        num_classes = int(run_params.getd(R.NUM_CLASSES, '10'))
        net = resnet20(ceclustering=True, num_classes=num_classes, ce_n_dim=5)
        train_l, train_bs, valid_l, valid_bs = DatasetProvider().get_datasets(run_params)
        # here the train dataset has the file list we are interested in..
        file_list = train_l.dataset.get_file_list()
        ce_clustering_loader.save(net.classification[-2], "temp.dat", file_list)
        
        # net.to("cuda")
        # summary(net, (1, 129, 129))

        # runner = AudioRunner(net, run_params, tensorboard_prefix='diskds')
        # log.info("Runner initialized, starting train")
        # runner.train()
        

    def help_str(self):
        return """Tries to teach a simple cec resnet for classes read from disk"""