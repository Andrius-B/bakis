import logging
import os
import torch
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.models.res_net_akamaster_audio import *
from src.models.working_model_loader import *
from src.config import Config
from src.runners.run_parameter_keys import R
from src.datasets.diskds.memory_file_storage import MemoryFileDiskStorage
from src.datasets.diskds.ceclustering_model_loader import CEClusteringModelLoader
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
from torch.utils.data import DataLoader
import json

log = logging.getLogger(__name__)
class SearchExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/spotifyTop10000)',
            # R.DATASET_NAME: 'disk-ds(/home/andrius/git/searchify/resampled_music)',
            R.DISKDS_NUM_FILES: '5000',
            R.DISKDS_WINDOW_LENGTH: str((2**17)),
            R.DISKDS_WINDOW_HOP_TRAIN: str((2**14)),
        }

    def load_model(self, net_save_path: str, cec_save_path: str, run_params: RunParameters, ce_clustering_loader: CEClusteringModelLoader):

        return model

    def run(self):
        run_params = self.get_run_params()
        config = Config()
        config.run_device = torch.device("cpu")
        spectrogram_generator = SpectrogramGenerator(config)
        model, file_list = load_working_model(run_params, "zoo/5000v8resample", True)
        cluster_positions = model.classification[-1].centroids.clone().detach().to("cpu")
        cluster_sizes = model.classification[-1].cluster_sizes.clone().detach().to("cpu")
        log.info(f"Cluster positions: {cluster_positions.shape}")
        # model.classification = torch.nn.Identity()
        model.to(config.run_device)
        model.save_distance_output = True
        log.info("Loading complete, server ready")
        example_file = "/media/andrius/FastBoi/micer/recordings/- 0New Order - Blue Monday(original).flac"
        _, file_extension = os.path.splitext(example_file)
        # a map of cluster -> (sum_dist, num_dist)
        sample_distances_to_clusters = {}
        total_samples = 0

        topk = 2

        with MemoryFileDiskStorage(example_file, format=file_extension[1:], run_params=run_params, features=["data"]) as memory_storage:
            loader = DataLoader(memory_storage, shuffle=False, batch_size=256, num_workers=0)
            for item_data in loader:
                with torch.no_grad():
                    samples = item_data["samples"]
                    samples = samples.to(config.run_device)
                    spectrogram = spectrogram_generator.generate_spectrogram(
                        samples, narrow_to=128,
                        timestretch=False, random_highpass=False,
                        random_bandcut=False, normalize_mag=True)
                    outputs = model(spectrogram)
                    _, topk_indices = outputs.topk(topk)
                    log.info(f"Top-{topk} classes shape: {topk_indices.shape}")
                    
                    batch_sample_distances = model.distance_output.to("cpu")
                    for i, batch_item_topk_indices in enumerate(topk_indices):
                        log.info(f"Batch item topk indicies: {batch_item_topk_indices}")
                        sample_norm_space_position = batch_sample_distances[i]
                        # log.info(f"Position of {i}th sample in norm space: {sample_norm_space_position.shape}")
                        for track_idx in batch_item_topk_indices.numpy():
                            track_norm_space_position = cluster_positions[track_idx]
                            track_cluster_size = float(cluster_sizes[track_idx])
                            # log.info(f"Position of {track_idx} cluster in norm space: {track_norm_space_position.shape}")
                            sample_distance_to_class_cluster = float(torch.dist(sample_norm_space_position, track_norm_space_position, 2))
                            sample_distance_to_class_cluster = sample_distance_to_class_cluster * (1/track_cluster_size)
                            if track_idx not in sample_distances_to_clusters:
                                sample_distances_to_clusters[track_idx] = (sample_distance_to_class_cluster, 1)
                            else:
                                previuos_distances = sample_distances_to_clusters[track_idx]
                                sample_distances_to_clusters[track_idx] = (previuos_distances[0] + sample_distance_to_class_cluster, previuos_distances[1] + 1)
        sample_avg_distances_to_clusters = {}
        for track_idx in sample_distances_to_clusters:
            distances = sample_distances_to_clusters[track_idx]
            sample_avg_distances_to_clusters[track_idx] = (distances[0]/distances[1], distances[1])
        sorted_sample_avg_distances_to_clusters = dict(sorted(sample_avg_distances_to_clusters.items(), key=lambda item: item[1][0], reverse=True))
        log.info(f"Sample average distances to class clusters: {sorted_sample_avg_distances_to_clusters}")
                    # for i, idx_t in enumerate(idxs):
                    #     idx = int(idx_t)
                    #     if idx not in model_output: 
                    #         model_output[idx] = {
                    #             "filename": file_list[idx],
                    #             "samplesOf": int(counts[i])
                    #         }
                    #     else:
                    #         model_output[idx]["samplesOf"] += int(counts[i])
                    # sorted_output = dict(sorted(model_output.items(), key=lambda item: item[1]["samplesOf"], reverse=True))
                    # log.info(f"Sorted sections: {json.dumps(sorted_output)}")
        

    def help_str(self):
        return """Tries to teach a simple cec resnet for classes read from disk"""