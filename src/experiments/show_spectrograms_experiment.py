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

log = logging.getLogger(__name__)

class ShowSpectrogramsExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DISKDS_WINDOW_LENGTH: str(2**17)
        }

    def get_statistics_of_spectrogram(self, spectrogram: Tensor):
        return f"""Value statistics:\nmin={spectrogram.min()}\nmax={spectrogram.max()}"""

    def draw_spectrogram(self, ax, spectrogram, sample_rate, title):
        ax.set_title(title)
        numpy_spec = spectrogram.numpy()[0][0] # unpack from the batch version
        print(f"Spectrogram shape: {numpy_spec.shape}")
        librosa.display.specshow(numpy_spec, sr=sample_rate, hop_length=1024,
                             x_axis='time', y_axis='mel', ax=ax, cmap='plasma')
        ax.text(0, 0, self.get_statistics_of_spectrogram(numpy_spec))

    def load_spectrogram(self, filepath, offset_frames, length_frames, spectrogram_generator: SpectrogramGenerator, run_params: RunParameters):
        if not os.path.exists(filepath):
            log.error(f"Requested test file not found at: {filepath}")
            raise RuntimeError(f"File not found: {filepath}")
        
        samples, sample_rate = torchaudio.backend.sox_backend.load(
            filepath,
            offset=offset_frames,
            num_frames=length_frames,
            normalization=False,
        )
        # sample_path = os.path.join(os.path.dirname(filepath), "samples", f"sample_{os.path.basename(filepath)}")
        # print(f"Would save file to: {sample_path}")
        # torchaudio.backend.sox_backend.save(sample_path, samples, 41000)
        samples = samples[0] # only take one channel.
        samples = samples.view(1,1,-1)
        log.info(f"Loaded samples reshaped to: {samples.shape}")
        raw_samples = samples[0][0].cpu().numpy()
        return (spectrogram_generator.generate_spectrogram(samples, normalize_mag=True, random_poly_cut=False, inverse_poly_cut=False).cpu(), sample_rate)

    def run(self):
        run_params = super().get_run_params()
        config = Config()
        fig = plt.figure(figsize=(8, 8))
        subplots = (3, 2)
        spectrogram_generator = SpectrogramGenerator(config)
        lenth_frames = 2**17
        files = [
            # { "filepath": "/home/andrius/git/bakis/data/test_data/New Order - Blue Monday.mp3", "offset_frames":130*41000, "length_frames":lenth_frames },
            # { "filepath": "/home/andrius/git/bakis/data/test_data/blue_monday_desktop_mic_headphone_spotify.mp3", "offset_frames":69*41000, "length_frames":lenth_frames },
            # { "filepath": "/home/andrius/git/bakis/data/test_data/blue_monday_desktop_mic_headphone_spotify_resampled.mp3", "offset_frames":69*41000, "length_frames":lenth_frames },
            # { "filepath": "/home/andrius/git/bakis/data/test_data/blue_monday_desktop_mic_phone_youtube.mp3", "offset_frames":134*41000, "length_frames":lenth_frames },
            # { "filepath": "/home/andrius/git/bakis/data/test_data/blue_monday_desktop_mic_speaker_youtube.mp3", "offset_frames":0, "length_frames":lenth_frames },
            # { "filepath": "/home/andrius/git/bakis/data/test_data/blue_monday_desktop_mic_speaker_youtube_resampled.mp3", "offset_frames":0, "length_frames":lenth_frames },
            # { "filepath": "/home/andrius/git/bakis/data/test_data/blue_monday_pitchbend_audacity.mp3", "offset_frames":75*41000, "length_frames":lenth_frames },
            # { "filepath": "/home/andrius/git/bakis/data/test_data/New Order - Blue Monday (experiment-modified).mp3", "offset_frames":8*41000, "length_frames":lenth_frames },
            { "filepath": "/media/andrius/FastBoi/bakis_data/top10000_meta_22k/!!! - Heart Of Hearts.mp3", "offset_frames":8*41000, "length_frames":lenth_frames }
        ]


        for i, f in enumerate(files):
            ax = fig.add_subplot(subplots[0], subplots[1], i+1)
            short_filename = os.path.basename(f["filepath"])
            (spectrogram, sample_rate) = self.load_spectrogram(f["filepath"], f["offset_frames"], f["length_frames"], spectrogram_generator, run_params)
            self.draw_spectrogram(ax, spectrogram, 41000, short_filename)
        fig.tight_layout()
        plt.show()


    def help_str(self):
        return """An experiment that loads a couple of files and generates spectrograms from it"""