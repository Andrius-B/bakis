import logging
import os
import torch
from torchsummary import summary
from src.runners.audio_runner import AudioRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.models.res_net_akamaster_audio import *
from src.models.working_model_loader import *
from src.runners.run_parameter_keys import R
from src.datasets.diskds.ceclustering_model_loader import CEClusteringModelLoader

class DiskDsLearner(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/spotifyTop10000)',
            # R.DATASET_NAME: 'disk-ds(/home/andrius/git/searchify/resampled_music)',
            R.DISKDS_NUM_FILES: '5000',
            R.BATCH_SIZE_TRAIN: '75',
            R.EPOCHS: '15',
            R.BATCH_SIZE_VALIDATION: '150',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: '1e-4',
            R.DISKDS_WINDOW_HOP_TRAIN: str((2**18)),
            R.DISKDS_WINDOW_HOP_VALIDATION: str((2**20)),
            R.MEASUREMENTS: 'loss,accuracy',
            R.DISKDS_WINDOW_LENGTH: str((2**17)),
            R.DISKDS_TRAIN_FEATURES: 'data,onehot',
            R.DISKDS_VALID_FEATURES: 'data,onehot',
            R.DISKDS_USE_SOX_RANDOM_PRE_SAMPLING_TRAIN: 'True',
        }

    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        model, _ = load_working_model(run_params, 'zoo/5000v8resample')
        model.to("cpu")
        summary(model, (1, 64, 128), device="cpu")

        runner = AudioRunner(model, run_params, tensorboard_prefix='diskds')
        log.info("Runner initialized, starting train")
        runner.train()
        # torch.save(model, net_save_path)
        # ce_clustering_loader.save(model.classification[-1], cec_save_path, file_list)
        save_working_model(model, run_params, 'zoo/5000v8resample')
        

    def help_str(self):
        return """Tries to teach a simple cec resnet for classes read from disk"""