import os
from src.datasets.diskds.dataset_index import SongSubset, DatasetSplitIndex
from src.tools.base_tool import BaseTool
import logging
import random

log = logging.getLogger(__name__)

class SetSplitter(BaseTool):
    def iterate_audio_files(self, directory, extensions):
        for root, directory, files in os.walk(directory):
            # print("Files found:" + str(files))
            for file in files:
                filepath = os.path.join(root, file)
                _, ext = os.path.splitext(filepath)
                if ext[1:] in extensions:
                    yield filepath


    def configure_argument_parser(self, parser):
        parser.add_argument(
            "dataset_dir",
            metavar='DATASET-DIR',
            help="Dataset directory (will be iterate recusiveley) for placing generated split files")
        parser.add_argument(
            '--dataset-names',
            metavar='DATASET-NAMES', nargs='+', help='Dataset names', required=True)
        parser.add_argument(
            '--dataset-percentages',
            metavar='DATASET-PERCENTAGES', nargs='+', help='Percentage of items to split for sets', type=float, required=True)
        return parser

    def run(self, args):
        dataset_dir = args.dataset_dir
        dataset_names = args.dataset_names
        dataset_percentages = args.dataset_percentages
        dataset_percentages = [float(x) for x in dataset_percentages]
        if sum(dataset_percentages) < 0.999 or sum(dataset_percentages) > 1.001:
            raise RuntimeError("Specified percentages do not add to 1")
        datasets = {}
        for dataset_name in dataset_names:
            datasets[dataset_name] = []
        random.seed(1234)
        ranges = []
        running_p = 0
        for p in dataset_percentages:
            ranges.append((running_p, running_p+p))
            running_p += p
        print(ranges)
        for audio_file in self.iterate_audio_files(dataset_dir, ["mp3", "flac", "wav"]):
            p = random.random()
            assigned = False
            for i, range in enumerate(ranges):
                if p >= range[0] and p < range[1]:
                    if i == 0:
                        datasets[dataset_names[0]].append(SongSubset(audio_file, 0.0, 1))    
                    else:
                        # allways add first half of song to the first dataset, but the second half to the selected dataset
                        datasets[dataset_names[0]].append(SongSubset(audio_file, 0.0, 0.5))
                        datasets[dataset_names[i]].append(SongSubset(audio_file, 0.5, 1))
                    assigned = True
            if not assigned:
                datasets[dataset_names[0]].append(SongSubset(audio_file, 0.0, 1))
        index = DatasetSplitIndex(datasets)
        index.save_to_diectory(dataset_dir)