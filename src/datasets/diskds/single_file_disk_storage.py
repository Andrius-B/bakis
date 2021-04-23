from src.datasets.diskds.disk_storage import DiskStorage
from typing import Generator, List
import os

class SingleFileDiskStorage(DiskStorage):

    def __init__(
        self,
        filepath: str,
        formats: List[str] = [".mp3", ".flac", ".wav"],
        window_generation_strategy=None
    ):
        directory = os.path.dirname(filepath)
        self.filepath = filepath
        super().__init__(
            directory = directory,
            formats=formats,
            window_generation_strategy=window_generation_strategy,
            skip_files=0)

    def idx_generator(self):
        if(len(self._file_array) == 0):
            # "We need to initialize the storage file array manually -- since idx generator can not do it
            self._file_array = self.get_audio_file_paths()
        idxs_generated = 0
        if(self.filepath not in self._file_array):
            print(f"File: {self.filepath} not found in the storage file array:")
            print(f"Full file array:")
            for f in self._file_array:
                print(f)
        for window in self.generate_windows(self.filepath):
            yield window
            idxs_generated += 1
            if(self._size_limit > 0 and idxs_generated >= self._size_limit):
                return
        self._length = idxs_generated