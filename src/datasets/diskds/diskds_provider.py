import re
from logging import getLogger
from torch.utils.data import DataLoader
from .disk_storage import UniformReadWindowGenerationStrategy, RandomSubsampleWindowGenerationStrategy
from src.runners.run_parameter_keys import R
from src.datasets.diskds.disk_dataset import DiskDataset, RandomReadDiskDataset
import src.runners.run_parameters


class DiskDsProvider:
    def get_disk_dataset(self, run_params, batch_sizes: (int, int), shuffle: (bool, bool)):
        log = getLogger(__name__)

        dataset_name_string = run_params.get_dataset_name()
        if not re.match(r"disk-ds\(.+\)", dataset_name_string):
            raise Exception(f"Disk dataset name not in correct format: `{dataset_name_string}` should be `disk-ds($PATH)`")
        path = dataset_name_string.split("(")[1][:-1]
        w_len = float(run_params.getd(R.DISKDS_WINDOW_LENGTH, str(2**16)))
        num_files = int(run_params.getd(R.DISKDS_NUM_FILES, str(-1)))
        features_str = run_params.getd(R.DISKDS_FEATURES, "data,onehot")
        features = [x.strip() for x in features_str.split(",")]
        formats_str = run_params.getd(R.DISKDS_FORMATS, ".flac,.mp3")
        formats = [x.strip() for x in formats_str.split(",")]
        log.info(f"Loading disk dataset from {path}")
        generation_strategy = RandomSubsampleWindowGenerationStrategy(window_len=w_len, average_hop=(2**16)*2, overread=1.08)
        # generation_strategy = UniformReadWindowGenerationStrategy(window_len=w_len, window_hop=(2**16)*10)
        log.info("Creating train dataset..")
        train_ds = DiskDataset(
            path,
            file_limit=num_files,
            features=features,
            formats=formats,
            window_generation_strategy=generation_strategy
        )
        log.info("Creating validation dataset..")
        valid_sampling_strategy = UniformReadWindowGenerationStrategy(window_len=w_len, window_hop=(2**16)*20, overread=1.09)
        valid_ds = DiskDataset(
            path,
            file_limit=num_files,
            features=features,
            formats=formats,
            window_generation_strategy=valid_sampling_strategy
        )

        loader = DataLoader(train_ds, shuffle=shuffle[0], batch_size=batch_sizes[0], num_workers=6)
        valid_loader = DataLoader(train_ds, shuffle=shuffle[1], batch_size=batch_sizes[1], num_workers=15)
        log.info(f"Imported dataset sizes -> train_ds: {len(train_ds)} valid_ds: {len(valid_ds)}")
        return (loader, batch_sizes[0], valid_loader, batch_sizes[1])