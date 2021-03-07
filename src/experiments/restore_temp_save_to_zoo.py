import logging
import os
from src.models.working_model_loader import *
from src.config import Config
from src.experiments.base_experiment import BaseExperiment
import torch

log = logging.getLogger(__name__)

class RestoreTempSaveToZooExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            #IMPORTANT: configure the correct dataset for the model here (as file data is not in the temporary pth file)
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/spotifyTop10000)',
            R.DISKDS_NUM_FILES: '9000',
        }

    def run(self):
        run_params = super().get_run_params()
        config = Config()
        model = resnet56(
            ceclustering = True,
            num_classes = int(run_params.get(R.DISKDS_NUM_FILES))
        )
        model.load_state_dict(torch.load("net.pth"))
        save_working_model(model, run_params, 'zoo/9000v1')


    def help_str(self):
        return """A quick experiment designed to load a temporary save file that the runner outputs and save it as a model in the zoo"""