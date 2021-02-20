import logging
import os
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.datasets.diskds.disk_storage import DiskStorage, UniformReadWindowGenerationStrategy
from src.datasets.diskds.disk_dataset import DiskDataset
from torch.utils.data import DataLoader
from src.datasets.diskds.sqlite_storage import SQLiteStorage, SQLiteStorageObject
from src.config import Config
from src.util.mutagen_utils import read_mp3_metadata,read_flac_metadata
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
from multiprocessing import Pool
from random import shuffle
from tqdm import tqdm
import math
import torch
import torchaudio
class MeanStdCalculator(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {}
    def process_file_step1(self, filepath):
        try:
            config = Config()
            config.run_device = torch.device("cpu")
            spectrogram_generator = SpectrogramGenerator(config)
            samples, sample_rate = torchaudio.backend.sox_io_backend.load(filepath, normalize=False)
            samples = samples.to(torch.device("cpu"))
            samples = samples.view(1, 1, -1)
            spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize=False)
            sum = torch.sum(spectrogram)
            n = torch.numel(spectrogram)
            min = torch.min(spectrogram)
            max = torch.max(spectrogram)
        except Exception as e:
            print(f"Failed reading file: {filepath}... check it out bro")
            print(e)
            return (int(99999999), (-99999999), (0), (0))
        return (int(min), int(max), int(n), int(sum))

    def process_file_step2(self, filepath):
        try:
            config = Config()
            config.run_device = torch.device("cpu")
            spectrogram_generator = SpectrogramGenerator(config)
            samples, sample_rate = torchaudio.backend.sox_io_backend.load(filepath, normalize=False)
            samples = samples.to(torch.device("cpu"))
            samples = samples.view(1, 1, -1)
            spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize=False)
            spectrogram = spectrogram - self.mean
            spectrogram = torch.pow(spectrogram, 2)
            sum = torch.sum(spectrogram)
            n = torch.numel(spectrogram)
            return (n, sum)
        except Exception as e:
            print(f"Failed reading file: {filepath}... check it out bro")
            print(e)
            return (0, 0)

    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        paths = list(DiskStorage("/home/andrius/git/bakis/data/resampled_2019").get_audio_file_paths())
        paths.extend(list(DiskStorage("/home/andrius/git/bakis/data/spotifyTop10000").get_audio_file_paths()))
        shuffle(paths)
        # paths = paths[:1000]
        N = 0
        SUM = 0
        MIN = 9999999
        MAX = -MIN
        VARIATION_SUM = 0
        VARIATION_N = 0
        self.mean = 0
        with Pool(3) as p:
            pbar = tqdm(p.imap_unordered(self.process_file_step1, paths), total=len(paths))
            for min_v, max_v, n, sum in pbar:
                SUM += sum
                N += n
                if MIN > min_v:
                    MIN = min_v
                if MAX < max_v:
                    MAX = max_v 
                mean = SUM/N
                pbar.set_description_str(f"min={MIN:3.2f} max={MAX:3.4f} mean={mean:3.4f}")
            
            print(f"Mean={SUM}/{N}={mean}")
            print(f"Min={MIN}; Max={MAX}")
            
            mean = SUM/N
            self.mean = mean
            pbar = tqdm(p.imap_unordered(self.process_file_step2, paths), total=len(paths))
            for n, sum in pbar:
                VARIATION_SUM += sum
                VARIATION_N += n
                if(VARIATION_N == 0):
                    print("Skipping iteration because variation size is zero!")
                    continue
                var = VARIATION_SUM/VARIATION_N
                pbar.set_description_str(f"var={var:3.4f} std={math.sqrt(var):3.4f}")
            
        print(f"Mean={SUM}/{N}={mean}")
        print(f"Min={MIN}; Max={MAX}")
        VAR = VARIATION_SUM/VARIATION_N
        print(f"variation={VARIATION_SUM}/{VARIATION_N}={VAR}; stdev={math.sqrt(VAR)}")
            

    def help_str(self):
        return """Calculates the spectrogram mean and standard deviation across a couple of datasets."""