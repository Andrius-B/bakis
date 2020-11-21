import os
import librosa
from typing import Generator, List
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen import File
import torch
from torch.utils.data import IterableDataset
from src.config import Config
from scipy import signal
import itertools
import numpy as np


class IterativeDiskDataset(IterableDataset):
    def __init__(
        self,
        directory: str,
        features: List[str],  # = ["data"]
        formats: List[str],  # = [".mp3", ".flac", ".wav"]
        file_limit: int,  # = -1
        n_preloaded_files: int,  # = 2
        window_size: int = 2**16,
        window_hop: int = 2**16/4
    ):
        self._root_dir = directory
        self._file_limit = file_limit
        self._file_count = -1
        self._length = -1
        self._formats = formats
        self._features = features

        self._window_size = window_size
        self._window_hop = window_hop
        self._current_file_index = 0

        self._n_preloaded_files = n_preloaded_files
        self._loaded_files: List[AudioFileWindowIterator] = [None]*n_preloaded_files
        self._circular_file_buf_index = 0
        self._last_loaded_file = -1
        self._len_estimate = -1

        if(file_limit < n_preloaded_files):
            raise IndexError("Can not use fewer files than preloaded, please reduce the preloaded file count")

    def __iter__(self):
        self._last_loaded_file = -1
        for i in range(self._n_preloaded_files):
            self._load_file_to_pos(i)
        return self

    def __next__(self):
        return self._get_item_and_load_if_needed()

    def __len__(self):
        if(self._len_estimate < 0):
            w_cnt = 0
            for fp in self.get_audio_file_paths_all():
                file_duration = self.get_file_duration(fp)
                windows = file_duration / (self._window_hop / 44100)
                windows = int(windows)
                # print(f"Duration for file: {fp} {file_duration} windows est: {windows}")
                w_cnt += windows
            self._len_estimate = w_cnt
        return self._len_estimate

    def _get_item_and_load_if_needed(self):
        """Loads items from in-memory storage using a circular buffer
        for files.
        For example if there are three files in  the storage, items will be interleaved like so:
        w1f1,w1f2,w1f3,w2f1,w2f2,w2f3.. (w1f1 - window#1 from file#1)
        """
        try:
            item = next(self._loaded_files[self._circular_file_buf_index])
            # print(f"Got item from iter at: {self._circular_file_buf_index}")
            self._circular_file_buf_index += 1
            if(self._circular_file_buf_index >= self._n_preloaded_files):
                self._circular_file_buf_index = 0
            return item
        except StopIteration:
            # print(f"Loading another file to position: {self._circular_file_buf_index}")
            if(self._last_loaded_file >= self.file_count() - 1):
                return self._get_item_no_load()
            self._load_file_to_pos(self._circular_file_buf_index)
            return self._get_item_and_load_if_needed()

    def _get_item_no_load(self):
        """Returns an item from the preloaded files
        if no such item is found, raises a StopIteration
        """
        for index, iterator in enumerate(self._loaded_files):
            try:
                item = next(iterator)
                return item
            except StopIteration:
                pass
        raise StopIteration

    def _load_file_to_pos(self, pos):
        """
        Loads an AudioFileWindowIterator for the next filepath in the list.
        This increments the self._last_loaded_file counter
        """
        self._last_loaded_file += 1
        filepath_it = itertools.islice(self.get_audio_file_paths_all(), self._last_loaded_file, None)
        filepath = next(filepath_it)
        # print(f"Loading file {filepath} to pos: {pos}")
        self._loaded_files[pos] = AudioFileWindowIterator(
            filepath,
            features=self._features,
            ohe_index=self._last_loaded_file,
            window_length=self._window_size,
            window_hop=self._window_hop
        )

    def file_count(self):
        if(self._file_count > 0):
            return self._file_count
        else:
            self._file_count = 0
            for _ in self.get_audio_file_paths_all():
                pass
            return self._file_count

    def get_audio_file_paths_all(self):
        file_count = 0
        for root, directory, files in os.walk(self._root_dir):
            for file in files:
                filepath = os.path.normpath(os.path.join(root, file))
                _, ext = os.path.splitext(filepath)
                if ext in self._formats:
                    yield filepath
                    file_count += 1
                if(self._file_limit > 0 and file_count >= self._file_limit):
                    self._file_count = file_count
                    return

    def get_file_duration(self, filepath):
        return librosa.core.get_duration(filename=filepath)


class AudioFileWindowIterator:
    def __init__(
        self,
        audio_file_path: str,
        features: List[str],
        ohe_index: int,
        window_length: int,
        window_hop: int,
    ):
        self._audio_file_path = audio_file_path
        self._features = features
        self._current_window_offset = 0
        self._window_length = window_length
        self._window_hop = window_hop
        self._samples, self._sample_rate = librosa.core.load(
            self._audio_file_path,
            sr=None,
            mono=True,
        )
        # in memory cache for metadata so it only has to be read once
        self._metadata_cache = None
        self._ohe_index = ohe_index
        if(self._ohe_index < 0):
            raise RuntimeError(f"""
            Can not generate One-Hot encoding for window: {str(self._audio_file_path)}
            Illegal index given: {self._ohe_index}""")

    def __iter__(self):
        self._current_window_offset = 0
        return self

    def __next__(self):
        output = {}
        if("data" in self._features):
            item_data = self.get_window_data()
            if("spectrogram" in self._features):
                samples_flat = item_data["samples"].numpy().reshape((-1))
                spectrogram = self.generate_mel_spectrogram(samples_flat)
                item_data = {**item_data, **spectrogram}
            output = {**output, **item_data}
        if("metadata" in self._features):
            metadata = self.read_item_metadata()
            output = {**output, **metadata}
        if("onehot" in self._features):
            onehotdata = self.generate_onehot()
            output = {**output, **onehotdata}
        return output

    def get_window_data(self):
        wstart = int(self._current_window_offset)
        wend = int(wstart + self._window_length)
        if(wend > len(self._samples)):
            raise StopIteration
        self._current_window_offset += self._window_hop
        samples = self._samples[wstart:wend]
        samples = torch.tensor(samples)
        samples = samples.reshape((1, -1)).float().to(Config().dataset_device)
        return {
            "samples": samples,
            "sample_rate": self._sample_rate,
            "filepath": self._audio_file_path,
            "window_start": wstart/self._sample_rate,
            "window_end": wend/self._sample_rate
        }

    def read_item_metadata(self):
        if(self._metadata_cache != None):
            return self._metadata_cache
        if(self._audio_file_path.endswith(".mp3")):
            metadata = self._read_mp3_metadata(self._audio_file_path)
        elif(self._audio_file_path.endswith(".flac")):
            metadata = self._read_flac_metadata(self._audio_file_path)
        else:
            metadata = File(self._audio_file_path)
            print("""
            File that is not MP3 or FLAC is being read
            for metadata, this behaviuor is not yet defined.
            metadata read:
            """)
            if(metadata):
                print(metadata.pprint())
        self._metadata_cache = metadata
        return metadata

    def generate_onehot(self):
        onehot = torch.LongTensor([self._ohe_index]).to(Config().dataset_device)
        return {"onehot": onehot}

    def _read_mp3_metadata(self, filepath):
        id3 = ID3(filepath)
        return {
            "artist": id3["TPE1"],
            "album": id3["TALB"],
            "title": id3["TIT2"]
        }

    def _read_flac_metadata(self, filepath):
        meta = FLAC(filepath)
        return {
            "artist": meta["artist"],
            "album": meta["album"],
            "title": meta["title"]
        }

    def generate_mel_spectrogram(self, samples):
        w_len = 2048
        hop = int(w_len/4)
        window = signal.windows.triang(w_len)
        mel_spec = librosa.feature.melspectrogram(samples, sr=44100, n_fft=w_len,
                                                  hop_length=hop,
                                                  n_mels=129)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return {"spectrogram": torch.tensor(mel_spec)}
