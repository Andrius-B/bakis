import random
from src.datasets.base_dataset import BaseDataset
from typing import List


class SplitDatasetBuilder:
    def __init__(self, base_dataset: BaseDataset, shuffle=False):
        super().__init__()
        self.base_dataset = base_dataset
        self.base_idx_list = base_dataset.get_idx_list()
        if(shuffle):
            random.shuffle(self.base_idx_list)
        self.splits = []

    def split_by_pct(self, probability: float, shuffle: (bool, bool) = None):
        first_idx_list = []
        second_idx_list = []
        for idx in self.base_dataset.get_idx_list():
            if(random.random() < probability):
                first_idx_list.append(idx)
            else:
                second_idx_list.append(idx)
        self.splits.append(SplitDatasetProxy(self.base_dataset, first_idx_list))
        self.splits.append(SplitDatasetProxy(self.base_dataset, second_idx_list))
        return self

    def get_datasets(self) -> List[BaseDataset]:
        return self.splits


class SplitDatasetProxy(BaseDataset):

    def __init__(self, base_dataset: BaseDataset, idx_list: List[str]):
        self.base_dataset = base_dataset
        self.idx_list = idx_list

    def get_idx_list(self) -> List[str]:
        return self.idx_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        idx = self.idx_list[index]
        idx_index = self.base_dataset.get_idx_list().index(idx)
        return self.base_dataset[idx_index]

    def get_file_list(self):
        return self.base_dataset.get_file_list()
