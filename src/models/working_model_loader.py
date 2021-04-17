from src.datasets.dataset_provider import DatasetProvider
from src.datasets.diskds.diskds_provider import DiskDsProvider
from src.datasets.diskds.ceclustering_model_loader import CEClusteringModelLoader
from src.datasets.diskds.mass_clustering_model_loader import MassClusteringModelLoader
from src.models.res_net_akamaster_audio import resnet56
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
import collections
from src.models.ceclustering import CEClustering
from src.models.mass_clustering import MassClustering
import os
import torch
import logging

log = logging.getLogger(__name__)

net_save_path = "temp.pth"
cec_save_path = "temp.csv"

def __create_clustering_model_loader(run_params: RunParameters):
    clustering_model_type = run_params.getd(R.CLUSTERING_MODEL, "cec")
    if clustering_model_type == "cec":
        log.info("Using CEClustering module")
        return CEClusteringModelLoader()
    elif clustering_model_type == "mass":
        log.info("Using MassClustering module")
        return MassClusteringModelLoader()
    else:
        raise Exception(f"Unknown clustering model type: '{clustering_model_type}'")

def load_working_model(
    run_params: RunParameters,
    model_path: str,
    reload_classes_from_dataset=True
):
    """A utility to load a working model"""
    clustering_model_type = run_params.getd(R.CLUSTERING_MODEL, "cec")
    model_new = resnet56(
        ceclustering = (clustering_model_type == "cec"),
        num_classes = int(run_params.get(R.DISKDS_NUM_FILES)),
        massclusterong = (clustering_model_type == "mass")
        )
    log.info(f"Created a new model for loading data into with classification: {model_new.classification}")
    model = model_new
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
    clustering_model_loader = __create_clustering_model_loader(run_params)
    clustering = model.classification[-1]
    logging.info(f"Before loader, classification model = {clustering}")
    if reload_classes_from_dataset:
        clustering, file_list = clustering_model_loader.load(clustering, final_cec_path, file_list)
        model.classification[-1] = clustering
    else:
        clustering, file_list = clustering_model_loader.load(clustering, final_cec_path, [])
        model.classification[-1] = clustering
    log.info(f"Loaded clustering module = {clustering} with loader: {clustering_model_loader}")
    return model, file_list

def save_working_model(
        model,
        run_params: RunParameters,
        model_path: str,
    ):
    final_net_path = net_save_path
    final_cec_path = cec_save_path
    if(model_path != None):
        final_net_path = f"{model_path}.pth"
        final_cec_path = f"{model_path}.csv"
    logging.info(f"Saving model to: {final_net_path} and {final_cec_path}")
    clustering_model_loader = __create_clustering_model_loader(run_params)
    torch.save(model, final_net_path)
    file_list = DatasetProvider().get_datasets(run_params)[0].dataset.get_file_list()
    clustering_model_loader.save(model.classification[-1], final_cec_path, file_list)