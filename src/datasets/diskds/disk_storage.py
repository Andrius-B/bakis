import os
import librosa
import random
from torchaudio.backend import sox_io_backend
from logging import getLogger
from typing import Generator, List, Optional
from tqdm import tqdm

log = getLogger(__name__)

class SpecificAudioFileWindow:
    """
    SpecificAudioFileWindow fully describes the window that of a file - it contains
    information about the file, length and offset of the window. This makes it usefull
    when full reproduction is required for a dataset.
    file_index - index of the target file in the storage_ref
    window_start - offset in file (in samples)
    window_end - offset in file (in samples)
    """

    def __init__(self, storage_ref, file_index: int, window_start: int, window_end: int):
        self.storage = storage_ref
        self.file_index = file_index
        self.window_start = window_start
        self.window_end = window_end

    def __repr__(self):
        # print(f"file_index: {self.file_index} files: {self.storage._file_array}")
        return f"{self.file_index}|({self.window_start},{self.window_end})"

    def get_filepath(self) -> str:
        return self.storage._file_array[self.file_index]

    @classmethod
    def from_string(cls, s, storage):
        file_id, window = s.split("|")
        w_start, w_end = map(float, window[1:-1].split(","))
        return SpecificAudioFileWindow(storage, int(file_id), w_start, w_end)


class RandomAudioFileWindow:
    """
    RandomAudioFileWindow only contains information about the filepath to be read and has an index from that file
    with a specified duration,
    but is does not have information about the offset of the window. This is intended to 
    be a random read of the specified length from the given file.
    file_index - index of the target file in the storage_ref
    window_len - length in samples
    window_index - integer index, should be unique per file.
    """

    def __init__(self, storage_ref, file_index: int, window_index: int, window_len: int):
        self.storage = storage_ref
        self.file_index = file_index
        self.window_len = window_len
        self.window_index = window_index

    def __repr__(self):
        # print(f"file_index: {self.file_index} files: {self.storage._file_array}")
        return f"{self.file_index}|({self.window_index},{self.window_len})"

    def get_filepath(self) -> str:
        return self.storage._file_array[self.file_index]

    @classmethod
    def from_string(cls, s, storage):
        file_id, window = s.split("|")
        window_index = int(window[1:-1].split(",")[0])
        window_len = float(window[1:-1].split(",")[1])
        return RandomAudioFileWindow(storage, int(file_id), window_index, window_len)


class DiskStorage:
    def __init__(
        self,
        directory: str,
        formats: List[str] = [".mp3", ".flac", ".wav"],
        window_generation_strategy=None,
        skip_files=0
    ):
        self._root_dir = directory
        self._size_limit = -1
        self._file_limit = -1
        self._file_count = -1
        self._skip_files = skip_files
        self._length = -1
        self._formats = formats
        self._file_array = []
        if(window_generation_strategy == None):
            self.window_generation_strategy = UniformReadWindowGenerationStrategy(2**16, 2**16 / 4)
        else:
            self.window_generation_strategy = window_generation_strategy

    def limit_size(self, size):
        self._size_limit = size
        return self

    def limit_files(self, file_limit):
        self._file_limit = file_limit
        return self

    def idx_generator(self):
        idxs_generated = 0
        files_iterated = 0
        skipped_files = 0
        running_window_length = -1
        file_paths = list(self.get_audio_file_paths())
        for file in file_paths:
            if(skipped_files < self._skip_files):
                skipped_files += 1
                continue
            for window in self.generate_windows(file):
                # Sanity test - check that windows are of the same length!
                if(isinstance(window, SpecificAudioFileWindow)):
                        window_len = window.window_end - window.window_start
                elif(isinstance(window, RandomAudioFileWindow)):
                    window_len = window.window_len

                if(running_window_length < 0):
                    running_window_length = window_len
                else:
                    if(abs(running_window_length - window_len) != 0):
                        raise Exception(f"Expected window lengths to match: (running length: {running_window_length}, actual:{window_len})")
                # sanity passed..
                yield window
                idxs_generated += 1
                if(self._size_limit > 0 and idxs_generated >= self._size_limit):
                    return
            files_iterated += 1
            if(self._file_limit > 0 and files_iterated >= self._file_limit):
                return
        self._length = idxs_generated

    def __len__(self):
        current_calc = max(self._size_limit, self._length)
        # recalculate length if currently unknown..
        if(current_calc < 0):
            self._length = 0
            for _ in self.idx_generator():
                self._length += 1
            current_calc = self._length
        return current_calc

    def file_count(self):
        if(self._file_count > 0):
            return self._file_count
        else:
            for _ in self.get_audio_file_paths():
                pass
            return self._file_count

    def generate_windows(self, file):
        for window in self.window_generation_strategy.generate_windows(file, self):
            yield window

    def get_audio_file_paths(self):
        self._file_count = 0
        self._file_array = []
        for root, directories, files in os.walk(self._root_dir):
            directories.sort()
            files.sort()
            for file in files:
                # print(file)
                filepath = os.path.normpath(os.path.join(root, file))
                _, ext = os.path.splitext(filepath)
                if ext in self._formats:
                    self._file_array.append(filepath)
                    yield filepath
                    self._file_count += 1
                    if(self._file_limit > 0 and self._file_count >= self._file_limit):
                        return


class UniformReadWindowGenerationStrategy:
    """
    Window generation strategy that reads windows from a file with a uniform step
    This generates A LOT of windows.
    """

    def __init__(
        self,
        window_len: int = 2**16,
        window_hop: int = (2**16 / 4),
        overread: int = 1.08
    ):
        """Parameters here are sample counts"""
        self.window_len: int = window_len
        self.window_hop: int = window_hop
        self.overread = overread

    def generate_windows(self, file, storage, file_format: Optional[str] = None):
        # print(f"getting info for file of type: {file_format}")
        metadata = sox_io_backend.info(file, file_format)
        # print(f"Metadata of the file: {metadata.__dict__}")
        duration = metadata.num_frames
        if(metadata.num_frames == 0 and not isinstance(file, str)):
            samples, sr = sox_io_backend.load(file, format=file_format)
            duration = samples.shape[-1]

        window_size = self.window_len
        i = 0  # index measued in samples
        step = self.window_hop  # leave overlap.
        while(i < duration - (window_size * self.overread)):
            f_id = -1
            if isinstance(file, str):
                f_id = storage._file_array.index(file)
            # w = SpecificAudioFileWindow(storage, f_id, i, i + window_size)
            # print(f"Generated window: {str(w)} max offset: {duration - window_size} current offset: {i}")
            window_end = i + int(window_size * self.overread)
            yield SpecificAudioFileWindow(storage, f_id, i, window_end)
            i += step


class RandomSubsampleWindowGenerationStrategy:
    """
    Window generation strategy that generates a preset amount of windows per file depending on duration,
    it only lists the index of the window. This is intended to be used for a random read of the required length
    to generate less windows.
    """

    def __init__(
        self,
        window_len: int = 2**16,
        average_hop: int = 2**16,
        overread: float = 1.08
    ):
        """Parameters here are sample counts"""
        self.window_len: int = int(window_len * overread)
        self.average_hop: int = average_hop

    def generate_windows(self, file, storage):
        metadata = sox_io_backend.info(file)
        duration = metadata.num_frames
        window_size = self.window_len
        i = 0  # index measued in samples
        step = self.average_hop
        generated_windows = 0
        f_id = storage._file_array.index(file)
        while(i < duration - window_size):
            yield RandomAudioFileWindow(storage, f_id, generated_windows, window_size)
            i += step
            generated_windows += 1