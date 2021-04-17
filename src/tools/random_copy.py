import shutil
from random import shuffle
from tqdm import tqdm
import pathlib
import os
import random

class RandomCopy:
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
            "src_dir",
            metavar='SRC-DIR',
            help="Source directory (will be iterate recusiveley)")
        parser.add_argument(
            "dest_dir",
            metavar='DEST-DIR',
            help="Output directory for copy targets"
        )
        parser.add_argument(
            "--count",
            help="How many random files should be copied from the original directory to the target directory?",
            default="10"
        )

    def run(self, args):
        src = args.src_dir
        dest = args.dest_dir
        if not os.path.isdir(dest):
            pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
        count = int(args.count)
        source_list = list(self.iterate_audio_files(src, ["mp3"]))
        random.shuffle(source_list)
        for file in tqdm(source_list[:count]):
            if not os.path.exists(os.path.join(dest, os.path.basename(file))):
                shutil.copy(file, dest)