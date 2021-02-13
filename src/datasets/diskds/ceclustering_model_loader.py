import os
from io import BytesIO
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from src.models.ceclustering import CEClustering
import sqlite3
from typing import List
from logging import getLogger
from torch import save as save_model
logger = getLogger(__name__)

class CEClusteringModelLoader:
    CENTROID_POSITION = 'centroid_positions'
    CLUSTER_SIZE = 'cluster_sizes'
    FILEPATH = 'centroid_filepath'    
    def __init__(self):
        pass

    def load(self, model: CEClustering, filepath: str, file_list: List[str]) -> CEClustering:
        logger.info(f"Loading CEClustring from :{filepath}")
        if not os.path.isfile(filepath):
            logger.warn("File to load CEClustering module not found, using the one provided as default!")
            return model
        else:
            df = pd.read_csv(filepath)
            if model.cluster_sizes.shape[0] != len(file_list):
                logger.warn(
"""Seems that the provided default model has incorrect dimensions for\
the provided file list. It will be resized to match the required size."""
                )
                with torch.no_grad():
                    model.cluster_sizes.data = torch.rand((len(file_list),))
                    model.centroids.data = torch.rand((len(file_list), model.centroids.shape[1]))
            for i,f in enumerate(file_list):
                if not (df[CEClusteringModelLoader.FILEPATH] == f).any():
                    logger.warn(f"No centroids found for file {f}, using the one provided f, which might be pretty wrong..")
                    model.cluster_sizes[i] = torch.tensor(0.4)
                    continue
                row = df.loc[df[CEClusteringModelLoader.FILEPATH] == f]
                cluster_size = torch.tensor(float(row[CEClusteringModelLoader.CLUSTER_SIZE]))
                centroid_pos = torch.tensor(np.array(eval(list(row[CEClusteringModelLoader.CENTROID_POSITION])[0])))
                # logger.info(f"Centroid size: {cluster_size} pos: {centroid_pos} file: {f}")
                model.cluster_sizes[i] = cluster_size
                model.centroids[i] = centroid_pos
            # print(df)
            model.centroids = nn.Parameter(model.centroids)
            model.cluster_sizes = nn.Parameter(model.cluster_sizes)
            return model

    def save(self, model: CEClustering, filepath: str, file_list: List[str]):
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
            logger.info(f"Cluster sizes: {model.cluster_sizes.shape}")
            logger.info(f"Model in features: {model.in_features} out features: {model.out_features}")
            df_source = {
                CEClusteringModelLoader.CENTROID_POSITION: [],
                CEClusteringModelLoader.CLUSTER_SIZE: [],
                CEClusteringModelLoader.FILEPATH: [],
            }
            for i, f in enumerate(file_list):
                centroid_pos = model.centroids[i].detach().numpy().tolist()
                df_source[CEClusteringModelLoader.CENTROID_POSITION].append(centroid_pos)
                cluster_size = model.cluster_sizes[i].item()
                df_source[CEClusteringModelLoader.CLUSTER_SIZE].append(cluster_size)
                df_source[CEClusteringModelLoader.FILEPATH].append(f)
            df = pd.DataFrame(data=df_source)
            logger.info("Saving CEClustering csv file:")
            logger.info("\n"+str(df))
            df.to_csv(filepath, index=False)
