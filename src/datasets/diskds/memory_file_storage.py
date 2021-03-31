from .base_dataset import BaseDataset
from typing import Generator, List
from .disk_storage import SpecificAudioFileWindow, RandomAudioFileWindow, UniformReadWindowGenerationStrategy, RandomSubsampleWindowGenerationStrategy
from src.runners.run_parameters import RunParameters
from src.datasets.diskds.sox_transforms import FileLoadingSoxEffects
from src.runners.run_parameter_keys import R
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
        run_params: RunParameters,
        window_generation_strategy=None,
        features: List[str] = ["metadata"],
        config: Config = Config()
    ):
        if window_generation_strategy is None:
            self.window_generation_strategy = UniformReadWindowGenerationStrategy(
                overread=1.1, window_len=int(run_params.getd(R.DISKDS_WINDOW_LENGTH, 2**16)),
                window_hop=int(run_params.getd(R.DISKDS_WINDOW_HOP_TRAIN, 2**15))
            )
        else:
            self.window_generation_strategy = window_generation_strategy
        self.memory_file = memory_file
        log.info(f"Memory file passed to storage has: {self.memory_file}")
        if isinstance(memory_file, str):
            self.disk_file = open(memory_file, 'r+b')
        else:
            self.disk_file = tempfile.NamedTemporaryFile(mode='w+b', suffix=f".{format}", dir="./data/server_downloads")
            # data = self.memory_file.read()
            # log.info(f"Read data from memory file: {data}")
            reader = io.BufferedReader(self.memory_file)
            self.disk_file.write(self.memory_file.read())
            log.info(f"Memory file dumped to: {self.disk_file.name}")
        info = torchaudio.backend.sox_io_backend.info(self.disk_file.name)
        log.info(f"Received file metadata: {info.__dict__}")
        self._file_array = [self.disk_file.name]
        self._length = -1
        self.windows = list(self.generate_windows())
        log.info(f"Generated windows: {len(self.windows)}")
        self._features = features
        self._config = config
        self.sox_effects = FileLoadingSoxEffects(random_pre_resampling=False)

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
        else:
            raise AttributeError(f"Unknown type of window: {type(window)}")
        win_len_frames = int(duration)
        # the sox_io_backend is slow as fuuuuuu, I can't update
        samples, sample_rate = torchaudio.backend.sox_backend.load(
            filepath = window.get_filepath(),
            offset = int(offset),
            num_frames = win_len_frames,

        )
        samples, sample_rate = self.sox_effects(samples)
        samples = samples.float().reshape((1,  -1))
        try:
            samples = samples.narrow(1, 0, win_len_frames) # make the float error less prominent by reading a round amount of frames
        except RuntimeError:
            log.error(f"Failed loading a sample from: {window.get_filepath()} at: {int(offset)}")
            log.error(f"Window: {window.__dict__}")
            #14467446 14467072
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