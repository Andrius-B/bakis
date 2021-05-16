from typing import List, Dict, Optional
from dataclasses import dataclass
import os
import pandas as pd
import logging

log = logging.getLogger(__name__)

@dataclass
class SongSubset:
    filepath: str
    # the following are meant to be percentages of time of the song, so for example
    # part [0;0.7) of a song goes to the train set and [0.7;1) goes to the test set.
    start: float
    end: float

class DatasetSplitIndex:

    def __init__(self, datasets: Dict[str, List[SongSubset]]) -> None:
        self.datasets = datasets
    
    @staticmethod
    def from_directory(directory: str):
        target_file = os.path.join(directory, "dataset_index.csv")
        if not os.path.isfile(target_file):
            log.error(f"Dataset index file not found at: {target_file}")
            return None
        df = pd.read_csv(target_file)
        dataset_names = df['dataset'].unique()
        dataset_names = sorted(dataset_names)
        datasets = {}
        for dataset_name in dataset_names:
            dataset_objects = []
            for row in df[df["dataset"] == dataset_name].itertuples():
                # print(row)
                dataset_objects.append(SongSubset(row.filepath, float(row.start), float(row.end)))
            datasets[dataset_name] = dataset_objects
        log.info(f"Dataset index loaded, containing the following sets: {datasets.keys()}")
        return DatasetSplitIndex(datasets)
        

    def save_to_diectory(self, directory: str):
        target_file = os.path.join(directory, "dataset_index.csv")
        if os.path.isfile(target_file):
            log.warn(f"Overwritting dataset index file at {target_file}")
        data = {
            'dataset': [],
            'filepath': [],
            'start': [],
            'end': [],
        }
        for dataset_name in self.datasets:
            for song_subset in self.datasets[dataset_name]:
                data['dataset'].append(dataset_name)
                data['filepath'].append(song_subset.filepath)
                data['start'].append(song_subset.start)
                data['end'].append(song_subset.end)
        df = pd.DataFrame(data=data)
        df.to_csv(target_file)

    def get_dataset_subsets(self, dataset_name) -> List[SongSubset]:
        dataset = self.datasets[dataset_name]
        log.info(f"Requested dataset {dataset_name} contains {len(dataset)} songs")
        return dataset