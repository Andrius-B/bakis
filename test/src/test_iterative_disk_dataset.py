import unittest
import time
from src.datasets.diskds.iterative_disk_dataset import IterativeDiskDataset, AudioFileWindowIterator


class TestIterativeDataset(unittest.TestCase):
    def test_iterative_dataset_file_limit(self):
        ds = IterativeDiskDataset(
            "test/test_data",
            file_limit=2,
            features=["data", "onehot"],
            formats=[".flac"],
            n_preloaded_files=2
        )
        ds._load_file_to_pos(0)
        ds._load_file_to_pos(1)
        self.assertEqual(2, ds.file_count())

    def test_iterative_dataset_len(self):
        ds = IterativeDiskDataset(
            "test/test_data",
            file_limit=2,
            features=["data", "onehot"],
            formats=[".flac"],
            n_preloaded_files=1
        )
        print(f"Loader length estimate: {len(ds)}")
        # self.assertEqual(27, ds.file_count())

    def test_iterative_dataset_file_limit2(self):
        num_files = 2
        w_len = 2**16
        w_hop = 2**16*6
        ds = IterativeDiskDataset(
            "test/test_data",
            file_limit=num_files,
            features=["data", "onehot"],
            formats=[".flac"],
            n_preloaded_files=2,
            window_size=w_len,
            window_hop=w_hop
        )
        self.assertEqual(num_files, ds.file_count())
        cnt_ds = 0
        for item in ds:
            # print(f"Item: {item['filepath']} ({item['window_start']};{item['window_end']})")
            cnt_ds += 1

        fps = list(ds.get_audio_file_paths_all())
        print(f"Counting elements from these files: {fps}")

        cnt_it = 0
        for fp in fps:
            it = AudioFileWindowIterator(
                fp,
                features=["data"],
                ohe_index=1,
                window_length=w_len,
                window_hop=w_hop)
            for item in it:
                # print(f"Item: {item['filepath']} ({item['window_start']};{item['window_end']})")
                cnt_it += 1

        self.assertEqual(cnt_it, cnt_ds)

    def test_iterative_dataset_AudioFileWindowIterator(self):
        it = AudioFileWindowIterator(
            "test/test_data/1 - Chino - Kolaps.flac",
            features=["data", "onehot", "metadata", "spectrogram"],
            ohe_index=1,
            window_length=2**16,
            window_hop=2**16/4)
        item = next(it)
        self.assertEqual(it._current_window_offset, 2**16/4)
        self.assertEqual(item["samples"].shape[1], 2**16)
        item = next(it)
        self.assertEqual(it._current_window_offset, (2**16/4)*2)
        self.assertEqual(item["samples"].shape[1], 2**16)
        self.assertEqual(1, item["onehot"].item())
        self.assertEqual(["Chino"], item["artist"])
        self.assertEqual((129, 129), item["spectrogram"].shape)

    def test_iterative_dataset_AudioFileWindowIterator(self):
        it = AudioFileWindowIterator(
            "test/test_data/1 - Chino - Kolaps.flac",
            # features=["data", "onehot", "metadata", "spectrogram"],
            features=["data", "onehot"],
            ohe_index=1,
            window_length=2**16,
            window_hop=2**16/4)

        t1 = time.time()
        cnt = 0
        for item in it:
            cnt += 1
            # if(cnt >= 50):
            #     break
        t2 = time.time()
        dt = t2 - t1
        print(f"Iteration time={dt}")
        print(f"Average time per window: {dt/cnt}")
        print(f"Total window count: {cnt}")
