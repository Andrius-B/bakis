from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.datasets.dataset_provider import DatasetProvider
from src.config import Config
from logging import getLogger
from matplotlib import pyplot as plt
import librosa.display
import matplotlib.cm
import torchaudio
import torch

class WindowToSpectro(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/home/andrius/git/bakis/data/spotifyTop10000)',
            R.DISKDS_NUM_FILES: '1',
            R.NUM_CLASSES: '10',
            R.BATCH_SIZE_TRAIN: '50',
            R.SHUFFLE_VALIDATION: 'False',
            R.BATCH_SIZE_VALIDATION: '1'
        }

    def run(self):
        self.dataset_provider = DatasetProvider()
        run_params = super().get_run_params()
        # load dataset:
        train_l, train_bs, valid_l, valid_bs = self.dataset_provider.get_datasets(run_params)
        log = getLogger(__name__)
        log.info("Running spectro test")

        # set up torchaudio transforms:
        self.config = Config()
        spectrogram_t = torchaudio.transforms.Spectrogram(
            n_fft=2048, win_length=2048, hop_length=1024, power=None
        ).to(self.config.run_device) # generates a complex spectrogram
        time_stretch_t = torchaudio.transforms.TimeStretch(hop_length=1024, n_freq=1025).to(self.config.run_device)
        freq_mask_t = torchaudio.transforms.FrequencyMasking(32)
        mel_t = torchaudio.transforms.MelScale(n_mels=64, sample_rate=self.config.sample_rate).to(self.config.run_device)
        norm_t = torchaudio.transforms.ComplexNorm(power=2).to(self.config.run_device)
        ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(self.config.run_device)

        test_item = None
        for item in valid_l:
             print(f"Item from validation: {str([item['filepath'], item['window_start'], item['window_end'], item['onehot']])}")
             test_item = item
             break
        
        samples = test_item["samples"].to(self.config.run_device)
        spectrogram = spectrogram_t(samples)
        spectrogram = spectrogram.narrow(3, 0, 64)
        spectrogram = norm_t(spectrogram)
        spectrogram = mel_t(spectrogram)
        spectrogram = ampToDb_t(spectrogram)
        print(f"Max in spectrogram: {torch.max(spectrogram)}, min: {torch.min(spectrogram)}")
        spectrogram[:, :, 32:64, :] = -100.0
        splice = spectrogram[:, :, 0:32, :]
        print(f"Splice shape: {splice.shape} --\n{splice}")
        spectrogram = spectrogram.cpu()[0][0]
        samples = samples.cpu().numpy()[0][0]

        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title(f"File Waveform ({test_item['filepath']})")
        librosa.display.waveplot(samples, sr=self.config.sample_rate, ax=ax1)


        ax2 = fig.add_subplot(2, 2, 3)
        ax2.set_title("Pytorch generated spectrogram")
        spectro = spectrogram.numpy()
        print(f"Spectrogram shape: {spectro.shape} --\n{spectro}")
        librosa.display.specshow(spectro, sr=self.config.sample_rate, hop_length=1024,
                             x_axis='time', y_axis='mel', ax=ax2)

        # full_samples, sample_rate = torchaudio.backend.sox_io_backend.load(item['filepath'][0])
        # print(f"full samples: {full_samples.shape}, sr:{sample_rate}")
        # samples = full_samples.to(self.config.run_device)
        # spectrogram = spectrogram_t(samples)
        # # spectrogram = spectrogram.narrow(3, 0, 129)
        # print(f"Complex spectrogram shape: {spectrogram.shape}")
        # tmp = torch.clone(spectrogram)
        # tmp = tmp[:, :, 100:520, :]
        # print(f"Sliced spectrogram shape: {tmp.shape}")
        # spectrogram = norm_t(spectrogram)
        # spectrogram = mel_t(spectrogram)
        # spectrogram = ampToDb_t(spectrogram)
        # print(f"after post processing spectrogram shape: {spectrogram.shape}")
        # spectrogram = spectrogram.cpu()[0]
        # full_samples = full_samples.cpu().numpy()[0]

        # splice_start = int(round(item['window_start'].item()*sample_rate, -1))
        # splice_end = int(round(item['window_end'].item()*sample_rate, -1))
        # splice_len = splice_end-splice_start
        # print(f"window start: {str(splice_start)}, end: {str(splice_end)}, len: {splice_len}")
        # print(f"full samples shape: {full_samples.shape} -- {full_samples}")
        # spliced_samples = full_samples[splice_start:splice_end]
        
        # # print(f"spliced samples size: {spliced_samples.shape} -- {spliced_samples}")
        # ax3 = fig.add_subplot(2, 2, 2)
        # ax3.set_title("Spliced File Waveform")
        # librosa.display.waveplot(spliced_samples, sr=sample_rate, ax=ax3)

        # print(f"full file spectrogram size: {spectrogram.shape} -- {spectrogram}")
        # spectro_splice_start = int(splice_start/512)
        # spectro_splice_end = int(splice_end/512)
        # spliced_spectrogram = spectrogram[:,spectro_splice_start:spectro_splice_end]
        # print(f"Spliced spectro shape: {spliced_spectrogram.shape} -- {spliced_spectrogram}")


        # ax4 = fig.add_subplot(2, 2, 4)
        # ax4.set_title("Pytorch generated spectrogram (spliced)")
        # spectro = spliced_spectrogram.numpy()
        # print(f"Spectrogram shape: {spectro.shape}")
        # librosa.display.specshow(spectro, sr=sample_rate, hop_length=512,
        #                      x_axis='time', y_axis='mel', ax=ax4)

        plt.show()
             
        

    def help_str(self):
        return """Run an experiment trying to pre-compute a spectrogram and select a window from it instead
        of reading a windows from data file and """