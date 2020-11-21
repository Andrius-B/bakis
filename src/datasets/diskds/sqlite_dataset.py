from .base_dataset import BaseDataset
from .disk_storage import UniformReadWindowGenerationStrategy
from typing import List
from .sqlite_storage import SQLiteStorage, AbstractSQLiteWindow
from src.config import Config

class SQLiteDataset(BaseDataset):
    def __init__(
            self,
            db_file: str,
            features=["metadata"],
            overread=1,
            config: Config = Config(),
            window_generation_strategy=None,
        ):
        if(window_generation_strategy == None):
            self.window_generation_strategy = UniformReadWindowGenerationStrategy(2**16, 2**16 / 4)
        else:
            self.window_generation_strategy = window_generation_strategy
        self._storage = SQLiteStorage(
            db_file,
            window_generation_strategy=window_generation_strategy
            )
        self._overread = overread
        self._config = config
        self._db_file = db_file
        self._length = -1
        self._window_list = list(self._storage.generate_windows())

    def __len__(self):
        current_calc = self._length
        # recalculate length if currently unknown..
        if(current_calc < 0):
            self._length = 0
            for _ in self.idx_generator():
                self._length += 1
            current_calc = self._length
        return current_calc

    def idx_generator(self):
        return range(len(self._window_list))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._load_item_from_window(x) for x in self._window_list[key]]
        else:
            return self._load_item_from_window(self._window_list[key])

    def _load_item_from_window(self, window: AbstractSQLiteWindow):
        # TODO: Loading the whole blob here consumes way too much memory.
        # split the blobs into smaller pieces and then load in pairs or something
        data = window.read_data()
        return {
            'spectrogram': data.spectrogram,
            'data_id': data.metadata.data_id,
            'metadata_id': data.metadata.metadata_id,
        }
