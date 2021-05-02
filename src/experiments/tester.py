import logging
import os
from src.experiments.base_experiment import BaseExperiment
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.datasets.diskds.disk_storage import DiskStorage
from src.datasets.diskds.sqlite_storage import SQLiteStorage, SQLiteStorageObject
from src.config import Config
from src.util.mutagen_utils import read_mp3_metadata, read_flac_metadata
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
from multiprocessing import Pool
import torch
import torchaudio


class Tester(BaseExperiment):

    def process_file(fp):
        samples, sample_rate = torchaudio.backend.sox_io_backend.load(filepath, normalize=False)
        samples = samples.to(config.run_device)
        samples = samples.view(1, 1, -1)
        spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize=False)
        SUM += torch.sum(spectrogram)
        N += torch.numel(spectrogram)
        min_v = torch.min(spectrogram)
        if min_v < MIN:
            MIN = min_v
        max_v = torch.max(spectrogram)
        if max_v > MAX:
            MAX = max_v
        mean = SUM/N
        return (MIN, MAX, N, SUM)

    def get_experiment_default_parameters(self):
        return {}

    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        config = Config()
        config.run_device = torch.device("cpu")
        spectrogram_generator = SpectrogramGenerator(config)
        paths = list(DiskStorage("/home/andrius/git/bakis/data/resampled_2019").get_audio_file_paths())
        paths.extend(list(DiskStorage("/home/andrius/git/bakis/data/spotifyTop10000").get_audio_file_paths()))
        paths = paths[:10]
        N = 0
        SUM = 0
        MIN = 9999999
        MAX = -MIN
        with Pool(5) as p:
            result = p.map(self.process_file, paths)
            print(result)

        print(f"Mean={SUM}/{N}={mean}")
        print(f"Min={MIN}; Max={MAX}")

    @staticmethod
    def help_str():
        return """This experiment load a dataset from disk and iterates though it once"""
