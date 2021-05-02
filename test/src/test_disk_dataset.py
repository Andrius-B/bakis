from unittest import TestCase
from torch.utils.data import DataLoader
from src.datasets.diskds.disk_dataset import DiskDataset, SpecificAudioFileWindow
from src.datasets.diskds.disk_storage import RandomSubsampleWindowGenerationStrategy, UniformReadWindowGenerationStrategy
from src.datasets.diskds.single_file_disk_storage import SingleFileDiskStorage
from torch.utils.data import DataLoader
import os
import numpy as np


class DiskDatasetTests(TestCase):

    def test_diskdataset(self):
        dataset = DiskDataset("test/test_data")
        idxs = dataset.get_idx_list()
        print(f"Loaded idx's: {idxs}")
        self.assertGreater(len(dataset), 100)

    def test_diskdataset_read_data(self):
        dataset = DiskDataset("test/test_data", features=["data"])
        for item in dataset[:5]:
            self.assertGreater(item["samples"].shape[1], 2**16)
            self.assertEqual(44100, item["sample_rate"])

    def test_diskdataset_read_data_uniform_windows(self):
        generation_strategy = UniformReadWindowGenerationStrategy(window_len=2**16, window_hop=2**17, overread=1)
        dataset = DiskDataset(
            "test/test_data",
            features=["data"],
            window_generation_strategy=generation_strategy,
        )
        for item in dataset[:5]:
            self.assertGreater(item["samples"].shape[1], 2**16-1)
            self.assertEqual(44100, item["sample_rate"])
        loader = DataLoader(dataset, batch_size=16)
        for item in loader:
            self.assertEqual(item["samples"].shape[0], 16)
            break

    def test_diskdataset_read_data_random_windows(self):
        generation_strategy = RandomSubsampleWindowGenerationStrategy(window_len=2**16, average_hop=2**17)
        dataset = DiskDataset(
            "test/test_data",
            features=["data"],
            window_generation_strategy=generation_strategy
        )
        for item in dataset[:5]:
            self.assertGreater(item["samples"].shape[1], 2**16)
            self.assertEqual(44100, item["sample_rate"])

    def test_diskdataset_read_data_sampling_len_equals(self):
        num_files = 1
        uniform_generation_strategy = UniformReadWindowGenerationStrategy(window_len=2**16, window_hop=2**16)
        uniform_ds = DiskDataset(
            "test/test_data",
            file_limit=num_files,
            features=["data"],
            formats=[".flac"],
            window_generation_strategy=uniform_generation_strategy
        )
        uniform_loader = DataLoader(uniform_ds, batch_size=3)
        random_generation_strategy = RandomSubsampleWindowGenerationStrategy(window_len=2**16, average_hop=2**16)
        random_ds = DiskDataset(
            "test/test_data",
            file_limit=num_files,
            features=["data"],
            formats=[".flac"],
            window_generation_strategy=random_generation_strategy
        )
        random_loader = DataLoader(random_ds, batch_size=3)
        print(f"Total windows in test files: {len(uniform_ds)}")
        self.assertEqual(len(uniform_ds), len(random_ds))
        self.assertEqual(len(random_loader), len(uniform_loader))
        print(f"Size of loaders: {len(random_loader)}")
        uniform_batch_count = 0
        for batch in uniform_loader:
            uniform_batch_count += 1
        random_batch_count = 0
        for batch in random_loader:
            random_batch_count += 1
        self.assertEqual(random_batch_count, uniform_batch_count)

    # def test_diskdataset_read_metadata_flac(self):
    #     dataset = DiskDataset("test/test_data", features=["metadata"])
    #     flac_meta = dataset.read_item_metadata(SpecificAudioFileWindow(dataset._disk_storage, 1, 0, 1))
    #     self.assertEqual(["Kolaps"], flac_meta['title'])
    #     self.assertEqual(["Chino"], flac_meta["artist"])
    #     self.assertEqual(["Kolaps"], flac_meta["album"])

    def test_diskdataset_read_metadata_mp3(self):
        dataset = DiskDataset("test/test_data/04_Lord.mp3", formats=[".mp3"], features=["metadata"], storage_type=SingleFileDiskStorage)
        window = SpecificAudioFileWindow(dataset._disk_storage, 0, 0, 1)
        mp3_meta = dataset.read_item_metadata(window)
        self.assertEqual("Lord", mp3_meta['title'])
        self.assertEqual("Tv.Out", mp3_meta["artist"])
        self.assertEqual("Dusk till Dawn", mp3_meta["album"])

    def test_diskdataset_generate_onehot(self):
        dataset = DiskDataset("test/test_data", features=["onehot"])
        # need to use the exact same path that os.normcase would give here.

        onehot1 = dataset.generate_onehot(SpecificAudioFileWindow(dataset._disk_storage, 0, 0, 1))
        self.assertEqual(onehot1["onehot"].item(), 0)

        onehot2 = dataset.generate_onehot(SpecificAudioFileWindow(dataset._disk_storage, 1, 0, 1))
        self.assertEqual(onehot2["onehot"].item(), 1)

        onehot3 = dataset.generate_onehot(SpecificAudioFileWindow(dataset._disk_storage, 2, 0, 1))
        self.assertEqual(onehot3["onehot"].item(), 2)

    # def test_diskdataset_single_file(self):

    #     dataset = DiskDataset("test/test_data/1 - Chino - Kolaps.flac", features=["onehot"], storage_type=SingleFileDiskStorage)
    #     print(dataset._disk_storage)
    #     onehot1 = dataset.generate_onehot(SpecificAudioFileWindow(dataset._disk_storage, 0, 0, 1))
    #     self.assertEqual(onehot1["onehot"].item(), 1)
    #     # check that only windows from the one file are read.
    #     for i in range(len(dataset)):
    #         item = dataset[i]
    #         self.assertEqual(item["onehot"].item(), 1)

    def test_diskdataset_single_file_read_data_random_windows(self):
        generation_strategy = UniformReadWindowGenerationStrategy(window_len=2**16, window_hop=2**16)
        dataset = DiskDataset(
            "test/test_data/1 - Chino - Kolaps.flac",
            features=["data"],
            storage_type=SingleFileDiskStorage,
            window_generation_strategy=generation_strategy
        )
        for item in dataset[:5]:
            self.assertGreater(item["samples"].shape[1], 2**16)
            self.assertEqual(44100, item["sample_rate"])
