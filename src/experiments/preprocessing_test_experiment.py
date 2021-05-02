import logging
import os
import torch
from src.experiments.base_experiment import BaseExperiment
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.config import Config
import random
import torchaudio
log = logging.getLogger(__name__)


class PreprocessingTestExperiment(BaseExperiment):
    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/home/andrius/git/bakis/data/spotifyTop10000)',
        }

    def run(self):
        self.config = Config()
        spectrogram_t = torchaudio.transforms.Spectrogram(
            n_fft=2048, win_length=2048, hop_length=1024, power=None
        ).to(self.config.run_device)  # generates a complex spectrogram
        time_stretch_t = torchaudio.transforms.TimeStretch(hop_length=1024, n_freq=1025).to(self.config.run_device)
        low_pass_t = torchaudio.functional.lowpass_biquad
        high_pass_t = torchaudio.functional.highpass_biquad
        mel_t = torchaudio.transforms.MelScale(n_mels=64, sample_rate=self.config.sample_rate).to(self.config.run_device)
        norm_t = torchaudio.transforms.ComplexNorm(power=2).to(self.config.run_device)
        ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(self.config.run_device)
        low_pass_t = torchaudio.functional.lowpass_biquad
        high_pass_t = torchaudio.functional.highpass_biquad
        griffinLim_t = torchaudio.transforms.GriffinLim(n_fft=2048, win_length=2048, hop_length=1024)

        samples, sr = torchaudio.backend.sox_backend.load(
            "/home/andrius/git/bakis/data/test_data/New Order - Blue Monday.mp3", offset=41000*130, num_frames=41000*20)
        samples = samples.to(self.config.run_device)
        log.info(f"Loaded = {samples} (len:{len(samples[0])}) sr={sr} -> {len(samples[0])/sr} seconds of audio")
        cutoff = random.randint(60, 1000)
        log.info(f"Applying highpass filter with cutoff={cutoff}..")
        samples = high_pass_t(samples, sr, cutoff_freq=cutoff)
        log.info(f"Computing spectrogram..")
        spectrogram = spectrogram_t(samples)
        if(random.random() > 0.5):  # half of the samples get a timestretch
            # this is because
            log.info(f"Applying time stretch..")
            spectrogram = time_stretch_t(spectrogram, random.uniform(0.93, 1.07))
        # samples = griffinLim_t(spectrogram)
        log.info(f"Saving file..")
        samples = samples.to(self.config.dataset_device)
        torchaudio.backend.sox_backend.save("/home/andrius/git/bakis/data/test_data/New Order - Blue Monday (experiment-modified).mp3", samples, sample_rate=sr)

    @staticmethod
    def help_str():
        return """Experiment to test how well pytorch pre-processes the audio and what the net trains from."""
