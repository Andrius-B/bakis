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
            R.DISKDS_NUM_FILES: '9000',
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
        model, file_list = load_working_model(run_params, "zoo/9000v3finetune", True)
        cluster_positions = model.classification[-1].centroids.clone().detach().to("cpu")
        cluster_sizes = model.classification[-1].cluster_sizes.clone().detach().to("cpu")
        cec_max_dist = model.classification[-1].max_dist
        log.info(f"Cluster positions: {cluster_positions.shape}")
        # model.classification = torch.nn.Identity()
        model.to(config.run_device)
        model.save_distance_output = True
        log.info("Loading complete, server ready")
        example_file = "/media/andrius/FastBoi/micer/recordings/Hardfloor - Acperience 1(original).flac"
        _, file_extension = os.path.splitext(example_file)
        # a map of cluster -> (sum_dist, num_dist)
        total_samples = 0
        topk = 5
        topk_classes_to_evaluate = torch.Tensor()
        sample_output_centroid_positions = torch.Tensor()
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
                    topk_indices = topk_indices.unique()
                    topk_classes_to_evaluate = torch.cat([topk_classes_to_evaluate, topk_indices]).unique()

                    batch_sample_distances = model.distance_output.to("cpu")
                    sample_output_centroid_positions = torch.cat([sample_output_centroid_positions, batch_sample_distances])
        log.info(f"Topk class indexes to evaluate in search: {topk_classes_to_evaluate.shape}")
        log.info(f"Output centroid positions shape: {sample_output_centroid_positions.shape}")
        sample_distances_to_clusters = {}
        for track_idx in topk_classes_to_evaluate.numpy():
            track_idx = int(track_idx)
            track_norm_space_position = cluster_positions[track_idx]
            track_cluster_size = float(cluster_sizes[track_idx])

            track_cluster_position_repeated = track_norm_space_position.repeat(sample_output_centroid_positions.shape[-2], 1)
            # log.info(f"Repeated track cluster position: {track_cluster_position_repeated.shape} should be: {sample_output_centroid_positions.shape}")
            output_distances = torch.norm(track_cluster_position_repeated - sample_output_centroid_positions, p=2, dim=-1)
            output_distances = torch.div(output_distances, cec_max_dist)
            # output_distances = torch.mul(output_distances, torch.div(1, track_cluster_size))
            # log.info(f"Output distances: {output_distances.shape} -- \n {output_distances}")
            mean_output_distance_for_track = output_distances.mean()
            sample_distances_to_clusters[track_idx] = mean_output_distance_for_track.item()
        sorted_sample_avg_distances_to_clusters = dict(sorted(sample_distances_to_clusters.items(), key=lambda item: item[1], reverse=False))
        log.info("Found the following recommendataions:")
        for track_idx in sorted_sample_avg_distances_to_clusters:
            track_name = os.path.basename(file_list[track_idx])
            track_name, _ = os.path.splitext(track_name)
            log.info(f"\t{track_name}: {sorted_sample_avg_distances_to_clusters[track_idx]} cluster_size={cluster_sizes[track_idx].item()}")
        

    def help_str(self):
        return """Tries to teach a simple cec resnet for classes read from disk"""