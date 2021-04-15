import os
from io import BytesIO
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from src.models.mass_clustering import MassClustering
import sqlite3
from typing import List
from logging import getLogger
from torch import save as save_model
from tqdm import tqdm
logger = getLogger(__name__)

class MassClusteringModelLoader:
    CENTROID_POSITION = 'centroid_positions'
    CLUSTER_MASS = 'cluster_mass'
    GRAVITATIONAL_CONSTANT = 'gravitational_constant'
    FILEPATH = 'centroid_filepath'    
    def __init__(self):
        pass

    def load(self, model: MassClustering, filepath: str, file_list: List[str]) -> MassClustering:
        logger.info(f"Loading MassClustering from :{filepath}")
        if not os.path.isfile(filepath):
            logger.warn("File to load MassClustering module not found, using the one provided as default!")
            return model, file_list
        
        df = pd.read_csv(filepath)
        if len(file_list) <= 0:
            file_list = list(df[MassClusteringModelLoader.FILEPATH])
        if model.cluster_mass.shape[0] != len(file_list):
            logger.warn(
"""Seems that the provided default model has incorrect dimensions for\
the provided file list. It will be resized to match the required size."""
            )
            with torch.no_grad():
                model.cluster_mass.data = torch.rand((len(file_list),))
                model.centroids.data = torch.rand((len(file_list), model.centroids.shape[1]))
        for i,f in tqdm(enumerate(file_list), total=len(file_list), leave=False):
            if not (df[MassClusteringModelLoader.FILEPATH] == f).any():
                logger.warn(f"No centroids found for file {f}, using the one provided f, which might be pretty wrong..")
                with torch.no_grad():
                    model.cluster_mass[i] = torch.tensor(0.4)
                continue
            row = df.loc[df[MassClusteringModelLoader.FILEPATH] == f]
            cluster_mass = torch.tensor(float(row[MassClusteringModelLoader.CLUSTER_MASS]))
            centroid_pos = torch.tensor(np.array(eval(list(row[MassClusteringModelLoader.CENTROID_POSITION])[0])))
            # logger.info(f"Centroid size: {cluster_mass} pos: {centroid_pos} file: {f}")
            with torch.no_grad():
                model.cluster_mass[i] = cluster_mass
                model.centroids[i] = centroid_pos
        # print(df)
        model.centroids = nn.Parameter(model.centroids)
        model.cluster_mass = nn.Parameter(model.cluster_mass)
        return model, file_list
        
    def save(self, model: MassClustering, filepath: str, file_list: List[str]):
        model = model.cpu()
        logger.info(f"Saving model:{str(model)}")
        if model.centroids.shape[0] != len(file_list):
            
            logger.error("""
            File list provided has a different amount of files than the model has centroids..
            Can not figure out the 1:1 relationship between centroid and file.""")
            logger.error(f"Centroids shape: {model.centroids.shape}, file_list length: {len(file_list)}")
        else:
            logger.info("Centroid amount and file count match!")
            logger.info(f"Centroids shape: {model.centroids.shape}")
            logger.info(f"Cluster sizes: {model.cluster_mass.shape}")
            logger.info(f"Gravity constant: {model.g_constant}")
            logger.info(f"Model in features: {model.in_features} out features: {model.out_features}")
            df_source = {
                MassClusteringModelLoader.CENTROID_POSITION: [],
                MassClusteringModelLoader.CLUSTER_MASS: [],
                MassClusteringModelLoader.FILEPATH: [],
            }
            for i, f in enumerate(file_list):
                centroid_pos = model.centroids[i].detach().numpy().tolist()
                df_source[MassClusteringModelLoader.CENTROID_POSITION].append(centroid_pos)
                cluster_mass = model.cluster_mass[i].item()
                df_source[MassClusteringModelLoader.CLUSTER_MASS].append(cluster_mass)
                df_source[MassClusteringModelLoader.FILEPATH].append(f)
            df = pd.DataFrame(data=df_source)
            logger.info("Saving MassClustering csv file:")
            logger.info("\n"+str(df))
            df.to_csv(filepath, index=False)
