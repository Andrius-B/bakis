import logging
import os
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.datasets.diskds.disk_storage import DiskStorage
from src.datasets.diskds.sqlite_storage import SQLiteStorage, SQLiteStorageObject
from src.config import Config
from src.util.mutagen_utils import read_mp3_metadata,read_flac_metadata
from src.runners.run_parameter_keys import R
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
import librosa
import torchaudio
from torch import Tensor
from matplotlib import pyplot as plt
import os

class SpectrogramAnalysisExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DISKDS_WINDOW_LENGTH: str(2**17)
        }

    def get_statistics_of_spectrogram(self, spectrogram: Tensor):
        return f"""Value statistics:\nmin={spectrogram.min()}\nmax={spectrogram.max()}\nmean={spectrogram.mean()}\nstd={spectrogram.std()}"""

    def draw_spectrogram(self, ax, spectrogram, sample_rate, title):
        ax.set_title(title)
        numpy_spec = spectrogram.numpy()[0][0] # unpack from the batch version
        print(f"Spectrogram shape: {numpy_spec.shape}")
        librosa.display.specshow(numpy_spec, sr=sample_rate, hop_length=1024,
                             x_axis='time', y_axis='mel', ax=ax)
        ax.text(0, 0, self.get_statistics_of_spectrogram(numpy_spec))

    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        config = Config()
        fig = plt.figure(figsize=(6, 6))
        subplots = (2, 2)
        spectrogram_generator = SpectrogramGenerator(config)
        filepath = "/media/andrius/FastBoi/bakis_data/final22k/train/Adele - Hello.mp3"
        if not os.path.exists(filepath):
            log.error(f"Requested test file not found at: {filepath}")
            raise RuntimeError(f"File not found: {filepath}")
        
        samples, sample_rate = torchaudio.backend.sox_backend.load(
            filepath,
            offset=config.sample_rate*49,
            num_frames=int(run_params.get(R.DISKDS_WINDOW_LENGTH)),
            normalization=False,
        )
        samples = samples[0] # only take one channel.
        samples = samples.view(1,1,-1)
        log.info(f"Loaded samples reshaped to: {samples.shape}")
        raw_samples = samples[0][0].cpu().numpy()
        non_augmented_spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize_mag=False).cpu()
        normalized_spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize_mag=True).cpu()
        random_spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize_mag=True, frf_mimic=False, frf_mimic_prob=0, random_poly_cut=False, inverse_poly_cut=False, add_noise=0, random_highpass=True, random_bandcut=True).cpu()
        # random_spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize_mag=True, frf_mimic=False, frf_mimic_prob=0, random_poly_cut=True, inverse_poly_cut=True, add_noise=0).cpu()
        # random_spectrogram = spectrogram_generator.generate_spectrogram(samples, normalize_mag=True, frf_mimic=True, frf_mimic_prob=1, random_poly_cut=False, inverse_poly_cut=False, add_noise=0).cpu()

        ax1 = fig.add_subplot(subplots[0], subplots[1], 1)
        ax1.set_title("File Waveform")
        librosa.display.waveplot(raw_samples, sr=sample_rate, ax=ax1)

        ax2 = fig.add_subplot(subplots[0], subplots[1], 2)
        self.draw_spectrogram(ax2, non_augmented_spectrogram, sample_rate, "Non-augmented spectrogram")

        ax3 = fig.add_subplot(subplots[0], subplots[1], 3)
        self.draw_spectrogram(ax3, normalized_spectrogram, sample_rate, "Normalized spectrogram")

        ax4 = fig.add_subplot(subplots[0], subplots[1], 4)
        self.draw_spectrogram(ax4, random_spectrogram, sample_rate, "Naive frequency masking")
        fig.tight_layout()
        plt.show()

    def help_str(self):
        return """An experiment that loads a file from disk and displays variuos different variations of spectrograms"""