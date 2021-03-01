from src.datasets.dataset_provider import DatasetProvider
from src.datasets.diskds.ceclustering_model_loader import CEClusteringModelLoader
from src.models.res_net_akamaster_audio import resnet56
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
import os
import torch
import logging

net_save_path = "temp.pth"
cec_save_path = "temp.csv"

ce_clustering_loader = CEClusteringModelLoader()

def load_working_model(
    run_params: RunParameters,
    model_path: str = None,
    reload_classes_from_dataset=True
):
    """A utility to load a working model"""

    model = resnet56(
        ceclustering = True,
        num_classes = int(run_params.get(R.DISKDS_NUM_FILES))
        )
    final_net_path = net_save_path
    final_cec_path = cec_save_path
    if(model_path != None):
        final_net_path = f"{model_path}.pth"
        final_cec_path = f"{model_path}.csv"
    if(os.path.isfile(final_net_path)):
        logging.info(f"Using saved model from: {final_net_path}")
        model = torch.load(final_net_path)
    logging.info(f"Loading model from: {final_net_path} and {final_cec_path}")
    file_list = []
    if reload_classes_from_dataset:
        # this loader generation consumes quite a bit of memory because it regenerates
        # all the idx's for the train set, maybe this would make sense to make DiskDsProvider
        # stateful and able to list all files (by calling a static method on disk_storage..)
        train_l, train_bs, valid_l, valid_bs = DatasetProvider().get_datasets(run_params)
        # here the train dataset has the file list we are interested in..
        file_list = train_l.dataset.get_file_list()
        ceclustring = model.classification[-1]
        ceclustring, file_list = ce_clustering_loader.load(ceclustring, final_cec_path, file_list)
        model.classification[-1] = ceclustring
    else:
        ceclustring = model.classification[-1]
        ceclustring, file_list = ce_clustering_loader.load(ceclustring, final_cec_path, [])
        model.classification[-1] = ceclustring
    return model, file_list

def save_working_model(model, run_params: RunParameters, model_path: str = None):
    final_net_path = net_save_path
    final_cec_path = cec_save_path
    if(model_path != None):
        final_net_path = f"{model_path}.pth"
        final_cec_path = f"{model_path}.csv"
    logging.info(f"Saving model to: {final_net_path} and {final_cec_path}")
    torch.save(model, final_net_path)
    file_list = DatasetProvider().get_datasets(run_params)[0].dataset.get_file_list()
    ce_clustering_loader.save(model.classification[-1], final_cec_path, file_list)