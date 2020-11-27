import os
from io import BytesIO
import torch.nn as nn
from src.models.ceclustering import CEClustering
import sqlite3
from typing import List
from logging import getLogger
from torch import save as save_model
logger = getLogger(__name__)

class CEClusteringModelLoader:
    
    def __init__(self):
        pass

    def load(self, filepath: str):
        pass

    def save(self, model: CEClustering, filepath: str, file_list: List[str]):
        logger.info(f"Saving model:{str(model)}")
        if model.centroids.shape[0] != len(file_list):
            
            logger.error("""
            File list provided has a different amount of files than the model has centroids..
            Can not figure out the 1:1 relationship between centroid and file.""")
            logger.error(f"Centroids shape: {model.centroids.shape}, file_list length: {len(file_list)}")
            for f in file_list:
                print(f)
        else:
            logger.info("Centroid amount and file count match!")

