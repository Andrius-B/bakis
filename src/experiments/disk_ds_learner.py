import logging
import os
import torch
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
            R.DISKDS_NUM_FILES: '5000',
            R.NUM_CLASSES: '5000',
            R.BATCH_SIZE_TRAIN: '150',
            R.EPOCHS: '100',
            R.BATCH_SIZE_VALIDATION: '40',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: '1e-3',
            R.MEASUREMENTS: 'loss,accuracy',
            R.DISKDS_TRAIN_FEATURES: 'data,onehot',
            R.DISKDS_VALID_FEATURES: 'data,onehot'
        }

    def run(self):
        ce_clustering_loader = CEClusteringModelLoader()
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        num_classes = int(run_params.getd(R.NUM_CLASSES, '-1'))
        model = resnet56(ceclustering=True, num_classes=num_classes)
        net_save_path = "temp.pth"
        cec_save_path = "temp.csv"
        if(os.path.isfile(net_save_path)):
            logging.info(f"Using saved model from: {net_save_path}")
            model = torch.load(net_save_path)
        # this loader generation consumes quite a bit of memory because it regenerates
        # all the idx's for the train set, maybe this would make sense to make DiskDsProvider
        # stateful and able to list all files (by calling a static method on disk_storage..)
        train_l, train_bs, valid_l, valid_bs = DatasetProvider().get_datasets(run_params)
        # here the train dataset has the file list we are interested in..
        file_list = train_l.dataset.get_file_list()
        ceclustring = model.classification[-1]
        ceclustring = ce_clustering_loader.load(ceclustring, cec_save_path, file_list)
        model.classification[-1] = ceclustring
        model.to("cuda")
        summary(model, (1, 64, 64))

        runner = AudioRunner(model, run_params, tensorboard_prefix='diskds')
        log.info("Runner initialized, starting train")
        runner.train()
        torch.save(model, net_save_path)
        ce_clustering_loader.save(ceclustring, cec_save_path, file_list)
        

    def help_str(self):
        return """Tries to teach a simple cec resnet for classes read from disk"""