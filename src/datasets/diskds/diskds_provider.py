import re
from logging import getLogger
from torch.utils.data import DataLoader
from .disk_storage import UniformReadWindowGenerationStrategy, RandomSubsampleWindowGenerationStrategy
from src.datasets.diskds.sox_transforms import FileLoadingSoxEffects
from src.runners.run_parameter_keys import R
from src.datasets.diskds.disk_dataset import DiskDataset, RandomReadDiskDataset
from src.datasets.diskds.disk_storage import DiskStorage
import src.runners.run_parameters
from src.config import Config
from typing import List, Tuple

log = getLogger(__name__)

class DiskDsProvider:
    def __init__(self, run_params):
        dataset_name_string = run_params.get_dataset_name()
        if not re.match(r"disk-ds\(.+\)", dataset_name_string):
            raise Exception(f"Disk dataset name not in correct format: `{dataset_name_string}` should be `disk-ds($PATH)`")
        self.path = dataset_name_string.split("(")[1][:-1]
        self.w_len = float(run_params.getd(R.DISKDS_WINDOW_LENGTH, str(2**16)))
        self.num_files = int(run_params.getd(R.DISKDS_NUM_FILES, str(-1)))
        self.run_params = run_params

    def get_disk_dataset(self, batch_sizes: Tuple(int, int), shuffle: Tuple(bool, bool), config: Config = Config()):
        
        train_features_str = self.run_params.getd(R.DISKDS_TRAIN_FEATURES, "data,onehot")
        train_features = [x.strip() for x in train_features_str.split(",")]

        valid_features_str = self.run_params.getd(R.DISKDS_VALID_FEATURES, train_features_str)
        valid_features = [x.strip() for x in valid_features_str.split(",")]

        train_window_hop = int(self.run_params.getd(R.DISKDS_WINDOW_HOP_TRAIN, str((2**16)*8)))
        validation_window_hop = int(self.run_params.getd(R.DISKDS_WINDOW_HOP_VALIDATION, str((2**16)*8)))

        use_random_pre_sampling_train = bool(self.run_params.getd(R.DISKDS_USE_SOX_RANDOM_PRE_SAMPLING_TRAIN, "False"))
        use_random_pre_sampling_valid = bool(self.run_params.getd(R.DISKDS_USE_SOX_RANDOM_PRE_SAMPLING_VALID, "False"))

        file_subet = self.run_params.getd(R.DISKDS_FILE_SUBSET, None)
        if file_subet is not None:
            file_subet = [int(x) for x in file_subet.split(',')]

        formats_str = self.run_params.getd(R.DISKDS_FORMATS, ".flac,.mp3")
        formats = [x.strip() for x in formats_str.split(",")]
        log.info(f"Loading disk dataset from {self.path}")
        generation_strategy = RandomSubsampleWindowGenerationStrategy(window_len=self.w_len, average_hop=train_window_hop, overread=1.08)
        # generation_strategy = UniformReadWindowGenerationStrategy(window_len=w_len, window_hop=(2**16)*10)
        log.info("Creating train dataset..")
        train_ds = DiskDataset(
            self.path,
            file_limit=self.num_files,
            features=train_features,
            formats=formats,
            window_generation_strategy=generation_strategy,
            sox_effects = FileLoadingSoxEffects(initial_sample_rate=config.sample_rate, final_sample_rate=config.sample_rate, random_pre_resampling=use_random_pre_sampling_train),
            file_subset=file_subet
        )
        log.info("Creating validation dataset..")
        valid_sampling_strategy = UniformReadWindowGenerationStrategy(window_len=self.w_len, window_hop=validation_window_hop, overread=1.09)
        valid_ds = DiskDataset(
            self.path,
            file_limit=self.num_files,
            features=valid_features,
            formats=formats,
            window_generation_strategy=valid_sampling_strategy,
            sox_effects = FileLoadingSoxEffects(initial_sample_rate=config.sample_rate, final_sample_rate=config.sample_rate, random_pre_resampling=use_random_pre_sampling_valid),
            file_subset=file_subet
        )

        loader = DataLoader(train_ds, shuffle=shuffle[0], batch_size=batch_sizes[0], num_workers=22)
        valid_loader = DataLoader(valid_ds, shuffle=shuffle[1], batch_size=batch_sizes[1], num_workers=24)
        log.info(f"Imported dataset sizes -> train_ds: {len(train_ds)} valid_ds: {len(valid_ds)}")
        return (loader, batch_sizes[0], valid_loader, batch_sizes[1])
    
    def get_file_list(self) -> List[str]:
        return DiskStorage(self.path).limit_files(self.num_files).get_audio_file_paths()