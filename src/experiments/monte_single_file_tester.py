import logging
import os
import torch
from torchsummary import summary
from src.runners.audio_runner import AudioRunner
from src.experiments.base_experiment import BaseExperiment
from src.runners.run_parameters import RunParameters
from src.datasets.diskds.disk_storage import RandomSubsampleWindowGenerationStrategy
from src.datasets.diskds.disk_dataset import DiskDataset
from src.datasets.diskds.single_file_disk_storage import SingleFileDiskStorage
from src.models.res_net_akamaster_audio import *
from src.models.working_model_loader import *
from src.datasets.diskds.sox_transforms import FileLoadingSoxEffects
from src.runners.run_parameter_keys import R
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
import matplotlib.pyplot as plt
import librosa
import torchaudio
from src.util.mutagen_utils import read_mp3_metadata
from src.config import Config
import matplotlib.cm

cmap = matplotlib.cm.get_cmap('plasma')

log = logging.getLogger(__name__)


class MonteSingleFileTester(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/final22k/train)',
            # R.DATASET_NAME: 'disk-ds(/home/andrius/git/searchify/resampled_music)',
            R.DISKDS_NUM_FILES: '9500',
            R.BATCH_SIZE_TRAIN: '75',
            R.CLUSTERING_MODEL: 'mass',
            R.MODEL_SAVE_PATH: 'zoo/9500massv2',
            R.EPOCHS: '2',
            R.BATCH_SIZE_VALIDATION: '150',
            R.DISKDS_WINDOW_HOP_TRAIN: str((2**17)),
            R.DISKDS_WINDOW_LENGTH: str((2**17)),
            R.DISKDS_TRAIN_FEATURES: 'data,onehot',
            R.DISKDS_USE_SOX_RANDOM_PRE_SAMPLING_TRAIN: 'False',
        }

    def generate_full_file_spectrogram(self, filepath: str, config: Config, ax):
        print("Generating full spectrogram")
        scale_hops = 10
        spectrogram_t_viz = torchaudio.transforms.Spectrogram(
            n_fft=2048*scale_hops, win_length=2048*scale_hops, hop_length=1024*scale_hops, power=None
        ).to(config.run_device)  # generates a complex spectrogram
        mel_t_viz = torchaudio.transforms.MelScale(n_mels=256, sample_rate=config.sample_rate).to(config.run_device)
        norm_t = torchaudio.transforms.ComplexNorm(power=2).to(config.run_device)
        ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(config.run_device)

        full_samples, sample_rate = torchaudio.load(filepath)
        full_samples = full_samples.to("cpu").view((1, -1))
        full_samples, sample_rate = FileLoadingSoxEffects(sample_rate, config.sample_rate, False).forward(full_samples)
        full_samples = full_samples.to(config.run_device)
        spectrogram = spectrogram_t_viz(full_samples.view(1, -1))
        spectrogram = norm_t(spectrogram)
        spectrogram = mel_t_viz(spectrogram)
        spectrogram = ampToDb_t(spectrogram)
        spectrogram = spectrogram.cpu().detach().numpy()[0]
        print(f"Spectrogram shape: {spectrogram.shape}")
        librosa.display.specshow(spectrogram, sr=config.sample_rate, hop_length=1024*scale_hops,
                                 x_axis='time', y_axis='mel', ax=ax, cmap=cmap)

    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        model_save_path = run_params.get(R.MODEL_SAVE_PATH)
        model, _ = load_working_model(run_params, model_save_path)
        config = Config()
        target_file = "/media/andrius/FastBoi/bakis_data/final22k/train/Adele - Hello.mp3"
        if not os.path.isfile(target_file):
            raise RuntimeError(f"Requested file not found at: {target_file}")
        file_info = torchaudio.backend.sox_io_backend.info(target_file)
        w_len = 2**17
        topn = 10
        generation_strategy = RandomSubsampleWindowGenerationStrategy(window_len=w_len, average_hop=int((w_len)*0.01))
        base_dataset = DiskDataset(
            target_file,
            file_limit=0,
            features=["data", "onehot"],
            formats=[".mp3"],
            window_generation_strategy=generation_strategy,
            storage_type=SingleFileDiskStorage,
            sox_effects=FileLoadingSoxEffects(initial_sample_rate=file_info.sample_rate, final_sample_rate=config.sample_rate, random_pre_resampling=False)
        )
        files = base_dataset.get_file_list()
        loader = DataLoader(base_dataset, shuffle=False, batch_size=64, num_workers=6)
        num_batches = len(loader)
        print(f"Files: {len(files)}")
        file_duration = int(file_info.num_frames/file_info.sample_rate)
        spectrogram_generator = SpectrogramGenerator(config)
        predicted_total = 0
        predicted_correctly = 0
        predicted_correctly_topk = 0
        top1_buckets = [[0, 0] for _ in range(file_duration)]
        topk_buckets = [[0, 0] for _ in range(file_duration)]
        num_epochs = int(run_params.get(R.EPOCHS))
        for i in range(num_epochs):
            pbar = tqdm(enumerate(loader), total=len(loader), leave=True)
            model.to(config.run_device)
            model.train(mode=False)
            for i, data in pbar:
                xb, yb = data["samples"], data["onehot"]
                start_times = data["window_start"]
                xb = xb.to(config.run_device)
                yb = yb.to(config.run_device)
                spectrogram = spectrogram_generator.generate_spectrogram(
                    xb, narrow_to=128,
                    timestretch=False, random_highpass=False,
                    random_bandcut=False, normalize_mag=True
                )
                outputs = model(spectrogram).detach()
                output_cat = outputs

                top_cats = output_cat.topk(topn)
                target_expanded = yb.expand_as(top_cats.indices).detach()
                topk_correct = target_expanded.eq(top_cats.indices)
                predicted_correctly_topk += topk_correct.sum().item()
                correct_topk = topk_correct.sum(dim=-1)
                output_cat = torch.argmax(outputs, dim=1)
                # print(f"Predicted category:{output_cat}")
                target = yb.detach().view(-1)
                correct = target.eq(output_cat).detach()
                correct_predictions_in_batch = correct.sum().item()
                predicted_total += len(target)
                predicted_correctly += correct_predictions_in_batch
                pbar.set_description(
                    f"running validation accuracy: TOP-1:{predicted_correctly/predicted_total:.3%}, TOP-{topn}: {predicted_correctly_topk/predicted_total:.3%}")
                for bi in range(len(target)):
                    # iterate over batch and add to the counters:
                    bi_correct = correct[bi].item()
                    bi_start_time = start_times.detach()[bi]
                    start_time_i = int(bi_start_time/file_info.sample_rate)
                    if(bi_correct):
                        # print(f"Correct prediction at time: {start_time_i}")
                        top1_buckets[start_time_i][0] = top1_buckets[start_time_i][0] + 1
                    top1_buckets[start_time_i][1] = top1_buckets[start_time_i][1] + 1

                    bi_topk_correct = correct_topk[bi]
                    if(bi_topk_correct):
                        topk_buckets[start_time_i][0] = topk_buckets[start_time_i][0] + 1
                    topk_buckets[start_time_i][1] = topk_buckets[start_time_i][1] + 1
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        self.generate_full_file_spectrogram(target_file, config, ax)
        lim_bottom_y, lim_top_y = ax.get_ylim()
        ax2 = ax.twinx()
        ax2.plot(range(file_duration), list([0 if x[1] == 0 else (x[0]/x[1])*100 for x in top1_buckets]), 'g', linewidth=3, label="top-1")
        ax2.plot(range(file_duration), list([0 if x[1] == 0 else (x[0]/x[1])*100 for x in topk_buckets]), 'm', linewidth=3, label=f"top-{topn}")
        plt.legend(loc="upper left")
        ax2.set_ylabel('Accuracy')
        metadata = read_mp3_metadata(target_file)
        display_name = metadata["artist"] + " - " + metadata["title"]
        fig.suptitle(f'Accuracy for \"{display_name}\"', fontsize=16)
        plt.show()

    @staticmethod
    def help_str():
        return """Tries to teach a simple cec resnet for classes read from disk"""
