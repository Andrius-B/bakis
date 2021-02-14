from .base_dataset import BaseDataset
from typing import Generator, List
from .disk_storage import SpecificAudioFileWindow, RandomAudioFileWindow, UniformReadWindowGenerationStrategy, RandomSubsampleWindowGenerationStrategy
from src.config import Config
import tempfile
import io
import logging
import torch
import torchaudio
import random
import os

log = logging.getLogger(__name__)
class MemoryFileDiskStorage(BaseDataset):
    """
    Event though this class is named MemoryFileDiskStorage, it actually stores the file to disk and then re-reads it,
    because torchaudio does not support decoding the raw data itself. See discussion at https://github.com/pytorch/audio/issues/800
    Should be good with the next release though, since the feature was merged into master here: https://github.com/pytorch/audio/pull/1158
    """
    def __init__(
        self,
        memory_file: io.BytesIO,
        format: str,
        window_generation_strategy=None,
        features: List[str] = ["metadata"],
        config: Config = Config()
    ):
        if window_generation_strategy is None:
            self.window_generation_strategy = UniformReadWindowGenerationStrategy(overread=1)
        else:
            self.window_generation_strategy = window_generation_strategy
        self.memory_file = memory_file
        log.info(f"Memory file passed to storage has: {self.memory_file}")
        self.disk_file = tempfile.NamedTemporaryFile(mode='w+b', suffix=f".{format}", dir="./data/server_downloads")
        # data = self.memory_file.read()
        # log.info(f"Read data from memory file: {data}")
        reader = io.BufferedReader(self.memory_file)
        self.disk_file.write(self.memory_file.read())
        log.info(f"Memory file dumped to: {self.disk_file.name}")
        self._file_array = [self.disk_file.name]
        self._length = -1
        self.windows = list(self.generate_windows())
        log.info(f"Generated windows: {len(self.windows)}")
        self._features = features
        self._config = config

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        return self._load_item_by_window(self.windows[index])

    def _load_item_by_window(self, window):
        window = window
        output = {}
        if("data" in self._features):
            item_data = self.read_item_data(window)
            if("spectrogram" in self._features):
                raise RuntimeError("Spectrogram not supported by memory file!")
            output = {**output, **item_data}
        if("metadata" in self._features):
            raise RuntimeError("Metadata not supported by memory file!")
        if("onehot" in self._features):
            raise RuntimeError("Onehot not supported by memory file!")
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
        win_len_frames = int(duration)
        # the sox_io_backend is slow as fuuuuuu, I can't update
        samples, sample_rate = torchaudio.backend.sox_backend.load(
            filepath = window.get_filepath(),
            offset = int(offset),
            num_frames = win_len_frames,

        )
        # librosa.effects.time_stretch(samples, random.uniform(0.93, 1.07))
        # samples = samples[:int(duration*44100)]
        samples = samples.float().reshape((1,  -1))
        try:
            samples = samples.narrow(1, 0, win_len_frames) # make the float error less prominent by reading a round amount of frames
        except RuntimeError:
            print(f"Failed loading a sample from: {window.get_filepath()} at: {int(offset*44100)}")
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

    def generate_windows(self):
        log.info("Generating windows from in-memory file")
        for window in self.window_generation_strategy.generate_windows(self.disk_file.name, self):
            yield window

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


    def close(self):
        log.info("Deleting temporary memory storage file")
        self.disk_file.close()

    # def read_item_data(self, window):
    #     if(isinstance(window, SpecificAudioFileWindow)):
    #         offset = window.window_start
    #         duration = window.window_end - window.window_start
    #     elif(isinstance(window, RandomAudioFileWindow)):
    #         duration = window.window_len
    #         metadata = torchaudio.backend.sox_io_backend.info(window.get_filepath())
    #         file_duration = metadata.num_frames # calculate duration in seconds
    #         # max_offset = min(30, (file_duration-duration))
    #         max_offset = file_duration-duration
    #         offset = int(random.uniform(0.0, max_offset))
    #     else:
    #         raise AttributeError(f"Unknown type of window: {type(window)}")
    #     win_len_frames = duration
    #     # the sox_io_backend is slow as fuuuuuu, I can't update
    #     samples, sample_rate = torchaudio.backend.sox_backend.load(
    #         window.get_filepath(),
    #         offset=int(offset),
    #         num_frames=win_len_frames
    #     )
    #     if(samples.shape[1] != win_len_frames):
    #         metadata = torchaudio.backend.sox_io_backend.info(window.get_filepath())
    #         if isinstance(window, SpecificAudioFileWindow):
    #             window_type = "SpecificAudioFileWindow"
    #         elif isinstance(window, RandomAudioFileWindow):
    #             window_type = "RandomAudioFileWindow"
    #         else:
    #             window_type = "unknown"
    #         difference = win_len_frames - samples.shape[1]
    #         samples2, _ = torchaudio.backend.sox_backend.load(
    #             window.get_filepath(),
    #             offset=(offset - difference), # shift back the requested offset a bit.. this might be a bug in pytorch
    #             num_frames=win_len_frames,
    #         )

    #         info = {
    #             'win_len_frames': win_len_frames,
    #             'file_len_frames': metadata.num_frames,
    #             'initially_requested_offset': offset,
    #             'file_remainder_after_initial_offset': metadata.num_frames - offset,
    #             'retry_offset_shifted_by': -difference,
    #             'window_type': window_type,                
    #             'file_name': window.get_filepath()
    #         }

    #         if(samples2.shape[1] == win_len_frames):
    #             # means we recovered .. sort of..
    #             # log.warn(f"Had to shift window start to get thre required amount of frames.. This might be a bug in torchaudio, additional context:{info}")
    #             samples = samples2
    #         else:
    #             log.error(f"Additional info for context: {info}")
    #             log.error(f"After a retry of re-reading the samples at a shifted offset, received: {samples2.shape[1]} samples")
    #             raise Exception(f"The amount of samples that was read is not the requested amount! requested: {win_len_frames} got: {samples.shape[1]}")
    #     # librosa.effects.time_stretch(samples, random.uniform(0.93, 1.07))
    #     # samples = samples[:int(duration*44100)]
    #     samples = samples.float().reshape((1,  -1))
    #     try:
    #         samples = samples.narrow(1, 0, win_len_frames) # make the float error less prominent by reading a round amount of frames
    #     except RuntimeError as e:
    #         print(f"Failed loading a sample from: {window.get_filepath()} at: {int(offset*44100)}")
    #     if(self._samples_length_runnging == None):
    #         self._samples_length_runnging = samples.shape[1]
    #     else:
    #         if(self._samples_length_runnging != samples.shape[1]):
    #             print(f"Read a sample of different length than the running size: {samples.shape}, expected: {self._samples_length_runnging}")
    #     samples = samples.reshape((1, -1)).float().to(self._config.dataset_device)
    #     w_start = offset,
    #     w_end = offset + duration
    #     return {
    #         "samples": samples,
    #         "sample_rate": torch.tensor(sample_rate).type(torch.LongTensor),
    #         "filepath": window.get_filepath(),
    #         "window_start": torch.tensor(w_start).type(torch.FloatTensor),
    #         "window_end": torch.tensor(w_end).type(torch.FloatTensor)
    #     }