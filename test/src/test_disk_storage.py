from unittest import TestCase
from src.datasets.diskds.disk_storage import DiskStorage, SpecificAudioFileWindow, RandomAudioFileWindow, RandomSubsampleWindowGenerationStrategy
import os


class DiskStorageTests(TestCase):
    def test_directory_list(self):
        ds = DiskStorage("test/test_data")
        file_list = [os.path.basename(x) for x in ds.get_audio_file_paths()]
        self.assertIn("63.wav", file_list)
        self.assertIn("04_Lord.mp3", file_list)
        self.assertIn("1 - Chino - Kolaps.flac", file_list)

    def test_generate_windows_for_file(self):
        ds = DiskStorage("test/test_data")
        for file in ds.get_audio_file_paths():
            windows = list(ds.generate_windows(file))
            self.assertGreater(len(windows), 0)

    def test_RandomAudioFileWindow_string(self):
        ds = DiskStorage("test/test_data")
        test_window = RandomAudioFileWindow(ds, 1, 5, 10.123)
        self.assertEqual("1|(5,10.123)", str(test_window))
        self.assertEqual(vars(test_window), vars(RandomAudioFileWindow.from_string("1|(5,10.123)", ds)))

    def test_SpecificAudioFileWindow_string(self):
        ds = DiskStorage("test/test_data")
        test_window = SpecificAudioFileWindow(ds, 1, 10.123, 420.5)
        self.assertEqual("1|(10.123,420.5)", str(test_window))
        self.assertEqual(vars(test_window), vars(SpecificAudioFileWindow.from_string("1|(10.123,420.5)", ds)))

    def test_idx_iteration(self):
        ds = DiskStorage("test/test_data")
        l = sum([1 for x in ds.idx_generator()])
        self.assertGreater(l, 3)

    def test_idx_iteration_random_window(self):
        generation_strategy = RandomSubsampleWindowGenerationStrategy(window_len=2**16, average_hop=2**17)
        ds = DiskStorage("test/test_data", window_generation_strategy=generation_strategy)
        l = sum([1 for x in ds.idx_generator()])
        self.assertGreater(l, 3)

    def test_size_limit(self):
        ds = DiskStorage("test/test_data").limit_size(100)
        l = sum([1 for x in ds.idx_generator()])
        self.assertEqual(l, 100)

    def test_file_limit(self):
        size = 2
        ds = DiskStorage("test/test_data").limit_files(size)
        l = sum([1 for _ in ds.get_audio_file_paths()])
        self.assertEqual(l, size)
        self.assertEqual(ds.file_count(), size)

    def test_len(self):
        dsl = DiskStorage("test/test_data").limit_size(100)
        ds = DiskStorage("test/test_data")
        self.assertEqual(len(dsl), 100)
        self.assertGreater(len(ds), len(dsl))
