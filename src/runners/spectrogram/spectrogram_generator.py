from src.config import Config
from torch import Tensor
import torchaudio
import torchvision
import random
import logging

logger = logging.getLogger(__name__)

class SpectrogramGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.spectrogram_t = torchaudio.transforms.Spectrogram(
            n_fft=2048, win_length=2048, hop_length=1024, power=None
        ).to(self.config.run_device) # generates a complex spectrogram
        self.time_stretch_t = torchaudio.transforms.TimeStretch(hop_length=1024, n_freq=1025).to(self.config.run_device)
        self.norm_t = torchaudio.transforms.ComplexNorm(power=2).to(self.config.run_device)
        self.mel_t = torchaudio.transforms.MelScale(n_mels=64, sample_rate=44100).to(self.config.run_device)
        self.ampToDb_t = torchaudio.transforms.AmplitudeToDB(stype='power').to(self.config.run_device)
        self.normalize_t = torchvision.transforms.Normalize([0.5], [0.5]).to(self.config.run_device)

    def generate_spectrogram(self, samples: Tensor, narrow_to=-1, timestretch=False, random_highpass=False, random_bandcut=False, normalize_stdev=True) -> Tensor:
        samples = samples.to(self.config.run_device)
        spectrogram = self.spectrogram_t(samples)
        # logger.info(f"Spectrogram size from window before narrowing and timestretch: {spectrogram.shape}")
        if timestretch and (random.random() > 0.5): # half of the samples get a timestretch
            # this is because
            spectrogram = self.time_stretch_t(spectrogram, random.uniform(0.93, 1.07))
        
        if narrow_to > 0:
            spectrogram = spectrogram.narrow(3, 0, narrow_to)
        # logger.info(f"Spectrogram size from window after narrowing: {spectrogram.shape}")
        spectrogram = self.norm_t(spectrogram)
        spectrogram = self.mel_t(spectrogram)
        spectrogram = self.ampToDb_t(spectrogram)
        spectrogram += 100
        mask_value = 0
        if normalize_stdev:
            # spectrogram -= spectrogram.min(2, keepdim=True)[0]
            # m = spectrogram.max(2, keepdim=True)[0]
            # if abs(m) > 0.00001:
            #     spectrogram = spectrogram /= m
            spectrogram -= 104.02396352970787
            spectrogram /= 26.794133651688224
            mask_value = 0
        # apply frequency masking
        if random_highpass and (random.random() < 0.5):
        # highpass type mask:
            bins_to_mask = random.randint(0,40) 
            spectrogram[:, :, 0:bins_to_mask, :] = mask_value
        if random_bandcut and (random.random() < 0.5):
            # band mask:
            bins_to_mask = random.randint(0,16)
            bin_start = random.randint(0,48)
            spectrogram[:, :, bin_start:bins_to_mask, :] = mask_value
        # logger.info(f"Final spectrogram size: {spectrogram.shape}")
        return spectrogram