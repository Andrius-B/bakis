import os
from subprocess import Popen, PIPE
import sys
import argparse
from typing import Callable, List, Tuple
from tqdm import tqdm
from src.tools.base_tool import BaseTool


class Resampler():
    def iterate_audio_files(self, directory, extensions):
        for root, directory, files in os.walk(directory):
            # print("Files found:" + str(files))
            for file in files:
                filepath = os.path.join(root, file)
                _, ext = os.path.splitext(filepath)
                if ext[1:] in extensions:
                    yield filepath


    def transcode_file(
        self, src_file, output_dir, fmt,
        channels, mp3_q, sample_rate, pbar
        ) -> Tuple[Popen, Callable]:
        launch_command = ["sox", "--norm"]
        file = os.path.basename(src_file)
        filename, ext = os.path.splitext(file)
        output_file = f"{filename}.{fmt}"
        output_file = os.path.join(output_dir, output_file)
        if(os.path.isfile(output_file)):
            pbar.write("Output file already exists, skipping!")
            return None
        launch_command.extend([str(src_file)])
        if(fmt == "mp3"):
            launch_command.extend(["-C", mp3_q])
        launch_command.extend(["-c", "1"])
        launch_command.extend(["-r", str(sample_rate)])
        launch_command.extend([str(output_file)])
        launch_command.extend(["remix", "-"])
        transcode_p = Popen(
            launch_command, stdout=PIPE, stderr=PIPE)
        def complete():
            stdout, stderr = transcode_p.communicate()
            stdout = stdout.decode('utf-8')
            stderr = stderr.decode('utf-8')
            if(len(stdout) > 0 or len(stderr) > 0):
                print(f"Output from resampling: {file}")
                print("========STDOUT=====")
                print(stdout)
                print("========STDERR=====")
                print(stderr)
        return (transcode_p, complete)


    def configure_argument_parser(self, parser):
        parser.add_argument(
            "src_dir",
            metavar='SRC-DIR',
            help="Source directory (will be iterate recusiveley)")
        parser.add_argument(
            "dest_dir",
            metavar='DEST-DIR',
            help="Output directory for all transcoded files (will be flat)"
        )
        parser.add_argument(
            "--format",
            help="Output format extension (mp3 / flac / wav)",
            default="mp3"
        )
        parser.add_argument(
            "--src-extensions",
            help="Comma separated file extensions to source for resampling",
            default="mp3,flac,wav"
        )
        parser.add_argument(
            "--channels",
            help="Num. channels in the output file",
            default="1"
        )
        parser.add_argument(
            "--sample-rate",
            help="Desired sample rate of the output file (in Hz)",
            default="41100"
        )
        parser.add_argument(
            "--mp3-q",
            metavar="MP3-QUALITY",
            help="""MP3 Quality parameter for sox: Because the −C value is a float, the fractional part is used to select quality.
            128.2 selects 128 kbps encoding with a quality of 2 (0 specifies highest quality but is very slow, while 9 selects poor quality, but is fast.)
            * This is only used if the output format is mp3..
            link: http://sox.sourceforge.net/soxformat.html
            """,
            default='198.2'
        )
        parser.add_argument(
            "--processes",
            help="How many resamping processes to run simultaniuosly",
            default="1"
        )
        return parser

    def run(self, args):
        src = args.src_dir
        dest = args.dest_dir
        extensions = [x.strip().replace('.','') for x in args.src_extensions.split(',')]
        fmt = args.format
        channels = str(int(args.channels))
        sample_rate = int(args.sample_rate)
        mp3_q = str(float(args.mp3_q))
        num_processes = int(args.processes)
        print(f"Starting resampler from {args.src_dir} to {args.dest_dir}")
        l = sum([1 for _ in self.iterate_audio_files(src, extensions)])
        it = tqdm(self.iterate_audio_files(src, extensions), total=l)
        queue:List[Tuple[Popen,Callable]] = []
        for file in it:
            it.write(f"Transcoding file to: {file}")
            future = self.transcode_file(
                src_file=file,
                output_dir=dest,
                fmt=fmt,
                channels=channels,
                mp3_q=mp3_q,
                sample_rate=sample_rate,
                pbar=it,
                )
            if future is None:
                continue
            while(len(queue) >= num_processes):
                to_be_removed = []
                for f in queue:
                    if(f[0].poll() != None):
                        # if one of the tracks has finished,
                        f[1]() # call the complete method
                        # and remove it from queue
                        to_be_removed.append(f)
                for x in to_be_removed:
                    queue.remove(x)
            queue.append(future)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    resampler = Resampler()
    resampler.configure_argument_parser(parser)
    args = parser.parse_args()
    resampler.run(args)
    