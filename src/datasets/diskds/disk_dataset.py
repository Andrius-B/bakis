from src.datasets.diskds.base_dataset import BaseDataset
from typing import List
from src.datasets.diskds.disk_storage import DiskStorage, SpecificAudioFileWindow, RandomAudioFileWindow, RandomSubsampleWindowGenerationStrategy
from mutagen import File
from src.config import Config
from src.util.mutagen_utils import read_flac_metadata, read_mp3_metadata
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from numba import njit
import random
import torchaudio
from logging import getLogger
from scipy import signal
import librosa
import torchaudio
import numpy as np
import torch


log = getLogger(__name__)

class DiskDataset(BaseDataset):
    def __init__(
            self,
            root_directory,
            file_limit=-1,
            size_limit=-1,
            features=["metadata"],
            storage_type=DiskStorage,
            config: Config = Config(),
            **kwargs):
        self._disk_storage = storage_type(root_directory, **kwargs)
        self._disk_storage.limit_size(size_limit)
        self._disk_storage.limit_files(file_limit)
        log.info("Generating sample idx's ...")
        self._idx_list = list(self._disk_storage.idx_generator())
        self._features = features
        self._file_list: List[str] = None
        self._samples_length_runnging = None
        self._config = config

        self.spectrogram_t = torchaudio.transforms.Spectrogram(
            n_fft=2048, win_length=2048, hop_length=512, power=None
        ).to(self._config.dataset_device) # generates a complex spectrogram
        self.time_stretch_t = torchaudio.transforms.TimeStretch(hop_length=512, n_freq=1025).to(self._config.dataset_device)
        self.low_pass_t = torchaudio.functional.lowpass_biquad
        self.high_pass_t = torchaudio.functional.highpass_biquad
        self.mel_t = torchaudio.transforms.MelScale(n_mels=129, sample_rate=44100).to(self._config.dataset_device)
        self.norm_t = torchaudio.transforms.ComplexNorm(power=2).to(self._config.dataset_device)
        self.ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(self._config.dataset_device)

    def __len__(self):
        return len(self._idx_list)

    def get_file_list(self) -> List[str]:
        return list(self._disk_storage.get_audio_file_paths())

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._load_item_by_window(x) for x in self._idx_list[key]]
        else:
            return self._load_item_by_window(self._idx_list[key])

    def _load_item_by_window(self, window):
        window = window
        output = {}
        if("data" in self._features):
            item_data = self.read_item_data(window)
            if("spectrogram" in self._features):
                samples_flat = item_data["samples"].numpy().reshape((-1))
                spectrogram = self.generate_mel_spectrogram(samples_flat)
                item_data = {**item_data, **spectrogram}
                item_data["samples"] = np.array([]) # reduce the memory footprint
                item_data["filepath"] = np.array([])
            output = {**output, **item_data}
        if("metadata" in self._features):
            metadata = self.read_item_metadata(window)
            output = {**output, **metadata}
        if("onehot" in self._features):
            onehotdata = self.generate_onehot(window)
            output = {**output, **onehotdata}
        return output

    def read_item_data(self, window):
        if(isinstance(window, SpecificAudioFileWindow)):
            offset = window.window_start
            duration = window.window_end - window.window_start
        elif(isinstance(window, RandomAudioFileWindow)):
            duration = window.window_len
            metadata = torchaudio.backend.sox_io_backend.info(window.get_filepath())
            file_duration = metadata.num_frames # calculate duration in seconds
            # max_offset = min(30, (file_duration-duration))
            max_offset = file_duration-duration
            offset = int(random.uniform(0.0, max_offset))
        else:
            raise AttributeError(f"Unknown type of window: {type(window)}")
        win_len_frames = duration
        # the sox_io_backend is slow as fuuuuuu, I can't update
        samples, sample_rate = torchaudio.backend.sox_backend.load(
            window.get_filepath(),
            offset=int(offset),
            num_frames=win_len_frames
        )
        if(samples.shape[1] != win_len_frames):
            metadata = torchaudio.backend.sox_io_backend.info(window.get_filepath())
            if isinstance(window, SpecificAudioFileWindow):
                window_type = "SpecificAudioFileWindow"
            elif isinstance(window, RandomAudioFileWindow):
                window_type = "RandomAudioFileWindow"
            else:
                window_type = "unknown"
            difference = win_len_frames - samples.shape[1]
            samples2, _ = torchaudio.backend.sox_backend.load(
                window.get_filepath(),
                offset=(offset - difference), # shift back the requested offset a bit.. this might be a bug in pytorch
                num_frames=win_len_frames,
            )

            info = {
                'win_len_frames': win_len_frames,
                'file_len_frames': metadata.num_frames,
                'initially_requested_offset': offset,
                'file_remainder_after_initial_offset': metadata.num_frames - offset,
                'retry_offset_shifted_by': -difference,
                'window_type': window_type,                
                'file_name': window.get_filepath()
            }

            if(samples2.shape[1] == win_len_frames):
                # means we recovered .. sort of..
                # log.warn(f"Had to shift window start to get thre required amount of frames.. This might be a bug in torchaudio, additional context:{info}")
                samples = samples2
            else:
                log.error(f"Additional info for context: {info}")
                log.error(f"After a retry of re-reading the samples at a shifted offset, received: {samples2.shape[1]} samples")
                raise Exception(f"The amount of samples that was read is not the requested amount! requested: {win_len_frames} got: {samples.shape[1]}")
        # librosa.effects.time_stretch(samples, random.uniform(0.93, 1.07))
        # samples = samples[:int(duration*44100)]
        samples = samples.float().reshape((1,  -1))
        try:
            samples = samples.narrow(1, 0, win_len_frames) # make the float error less prominent by reading a round amount of frames
        except RuntimeError as e:
            print(f"Failed loading a sample from: {window.get_filepath()} at: {int(offset*44100)}")
        if(self._samples_length_runnging == None):
            self._samples_length_runnging = samples.shape[1]
        else:
            if(self._samples_length_runnging != samples.shape[1]):
                print(f"Read a sample of different length than the running size: {samples.shape}, expected: {self._samples_length_runnging}")
        samples = samples.reshape((1, -1)).float().to(self._config.dataset_device)
        w_start = offset,
        w_end = offset + duration
        return {
            "samples": samples,
            "sample_rate": torch.tensor(sample_rate).type(torch.LongTensor),
            "filepath": window.get_filepath(),
            "window_start": torch.tensor(w_start).type(torch.FloatTensor),
            "window_end": torch.tensor(w_end).type(torch.FloatTensor)
        }

    def read_item_metadata(self, window):
        if(window.get_filepath().endswith(".mp3")):
            metadata = read_mp3_metadata(window.get_filepath())
        elif(window.get_filepath().endswith(".flac")):
            metadata = read_flac_metadata(window.get_filepath())
        else:
            metadata = File(window.get_filepath())
            print("""
            File that is not MP3 or FLAC is being read
            for metadata, this behaviuor is not yet defined.
            metadata read:
            """)
            if(metadata):
                print(metadata.pprint())
        return metadata

    def generate_onehot(self, window):
        """This implementation of one-hot encoding assumes that each file is unique
        and a window is mapped directly to a file, i.e. the onehot vector has length that is equal
        to the count of files for this dataset."""
        # file_list = self.get_file_list()
        index = window.file_index
        if(index < 0):
            raise RuntimeError(f"""
            Can not generate One-Hot encoding for window: {str(window)}
            Not found in dataset idx_list""")
        # onehot = np.zeros(len(file_list))
        # onehot[index] = 1
        # onehot = torch.tensor(onehot).to(Config.device)
        # return {"onehot": onehot.long()}
        # looks like pytorch only wants the actual index:
        onehot = torch.LongTensor([index]).to(self._config.dataset_device)
        return {"onehot": onehot}



    def generate_mel_spectrogram(self, samples):
        samples = torch.tensor(samples).to(self._config.dataset_device)
        samples = samples.reshape((1,-1))
        # log.info(f"Samples: {samples.shape}")
        # if(random.random() < 0.1):
        #     samples = self.low_pass_t(samples, 41000, random.randint(100, 1000))
        # if(random.random() < 0.1):
        #     samples = self.high_pass_t(samples, 41000, random.randint(2000, 10000))
        # spectrogram = self.spectrogram_t(samples)
        # # log.info(f"Spectrogram: {spectrogram.shape}")
        if(random.random() > 0.5): # half of the samples get a timestretch
            # this is because
            spectrogram = self.time_stretch_t(spectrogram, random.uniform(0.93, 1.07))
            # log.info(f"Spectrogram stretched: {spectrogram.shape}")
        spectrogram = spectrogram.narrow(2, 0, 129)
        # log.info(f"Spectrogram narrowed: {spectrogram.shape}")
        # print(f"Spectrogram after slowing down: {spectrogram.shape}")
        spectrogram = self.norm_t(spectrogram)
        spectrogram = self.mel_t(spectrogram)
        spectrogram = self.ampToDb_t(spectrogram)
        # log.info(f"Final spectrogram: {spectrogram.shape}")
        return {"spectrogram": spectrogram.to(self._config.dataset_device)}

class RandomReadDiskDataset(DiskDataset):
    def __init__(
            self,
            root_directory,
            length,
            file_limit=-1,
            size_limit=-1,
            features=["metadata"],
            overread=1,
            storage_type=DiskStorage,
            config: Config = Config(),
            **kwargs):
        if('window_generation_strategy' in kwargs):
            raise RuntimeError("RanodmReadDiskDataset does not support window_generation_stategy (it's random allways)")
        self._window_generation_strategy = RandomSubsampleWindowGenerationStrategy()
        kwargs['window_generation_strategy'] = self._window_generation_strategy
        self._disk_storage = storage_type(root_directory, **kwargs)
        self._disk_storage.limit_size(size_limit)
        self._disk_storage.limit_files(file_limit)
        self._features = features
        self._file_list: List[str] = None
        self._samples_length_runnging = None
        self._config = config
        self._length_override = length

    def __len__(self):
        return self._length_override

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise RuntimeError("Random read dataset does not support slice indexing")
        else:
            files = self._disk_storage._file_array  
            if(len(self._disk_storage._file_array) <= 0):
                files = list(self._disk_storage.get_audio_file_paths())
            random_file = random.choice(files)
            return self._load_item_by_window(self._window_generation_strategy.generate_random_window(random_file, self._disk_storage))