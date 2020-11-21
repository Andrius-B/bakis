import logging
import os
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.datasets.diskds.disk_storage import DiskStorage
from src.datasets.diskds.sqlite_storage import SQLiteStorage, SQLiteStorageObject
from src.config import Config
from src.util.mutagen_utils import read_mp3_metadata,read_flac_metadata
import torchaudio
class Tester(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {}
    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        config = Config()
        spectrogram_t = torchaudio.transforms.Spectrogram(
            n_fft=2048, win_length=2048, hop_length=512, power=None
        ).to(config.run_device) # generates a complex spectrogram
        mel_t = torchaudio.transforms.MelScale(n_mels=129, sample_rate=44100).to(config.run_device)
        norm_t = torchaudio.transforms.ComplexNorm(power=2).to(config.run_device)
        ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(config.run_device)
        storage = DiskStorage("/home/andrius/git/bakis/data/spotifyTop10000")
        sqlStorage = SQLiteStorage("test.db")
        for filepath in storage.get_audio_file_paths():
            samples, sample_rate = torchaudio.backend.sox_io_backend.load(filepath)
            print(f"Read file samples:{samples.shape}, sr: {sample_rate}, {os.path.basename(filepath)}")
            samples = samples.to(config.run_device)
            spectrogram = spectrogram_t(samples)
            spectrogram = norm_t(spectrogram)
            spectrogram = mel_t(spectrogram)
            spectrogram = ampToDb_t(spectrogram)
            spectrogram = spectrogram.cpu().numpy()
            metadata = None
            if(filepath.lower().endswith("mp3")):
                metadata = read_mp3_metadata(filepath)
            elif(filepath.lower().endswith("flac")):
                metadata = read_flac_metadata(filepath)
            print(f"Writing spectrogram of shape: {spectrogram.shape}")
            o = SQLiteStorageObject(
                metadata["artist"], metadata["album"], metadata["title"],
                sample_rate, samples.shape[1], filepath,
                spectrogram
            )
            sqlStorage.insert_new_track(o)
        item = sqlStorage.get_all()
        print(item)


        

    def help_str(self):
        return """This experiment load a dataset from disk and iterates though it once"""