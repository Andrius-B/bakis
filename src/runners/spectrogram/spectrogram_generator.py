from src.config import Config
from torch import Tensor
import torchaudio
import torchvision
import random
import torch
from typing import Tuple
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
    
    def generate_masking_polynomial(self, a_min, a_max) -> Tuple[int, int, int]:
        """
        Generates a random square polynomial that has a critical point
        that has coordinates (x;y) where 0 < x < 1 and 0 < y < 1
        Returns coefficiants for the polynomial: ax^2 + bx + c where a is in range [a_min; a_max]
        a MUST be positive -> the polynomial ir upright
        """
        # define a range as [a_min;a_max]
        a = random.uniform(a_min, a_max)
        # b must be in range [-2a;0]
        b_min, b_max = -2*a, 0
        b = random.uniform(b_min, b_max)
        # TODO: simplify this expression:
        tmp = - (b**2)/(4*a) + (b**2)/(2*a)
        c_min = 1 + tmp
        c_max = tmp
        c = random.uniform(c_min, c_max)
        return (a, b, c)
        # return (0, 1, 0)

    def generate_spectrogram(
            self, samples: Tensor,
            narrow_to=-1, timestretch=False,
            random_highpass=False, random_bandcut=False,
            random_poly_cut=False, inverse_poly_cut=False, # inverse here means that only the poly is left as was
            normalize_mag=True,
            clip_amplitude = 0, raise_to_power = 1
        ) -> Tensor:
        samples = samples.to(self.config.run_device)
        spectrogram = self.spectrogram_t(samples)
        # logger.info(f"Spectrogram size from window before narrowing and timestretch: {spectrogram.shape}")
        if timestretch and (random.random() > 0.5): # half of the samples get a timestretch
            # this is because
            spectrogram = self.time_stretch_t(spectrogram, random.uniform(0.93, 1.07))
        
        if narrow_to > 0:
            spectrogram = spectrogram.narrow(3, 0, narrow_to)
        spectrogram = self.norm_t(spectrogram)
        spectrogram = self.mel_t(spectrogram)
        # note: the size here will be: (batch_size, channel_count, height, width)
        # logger.info(f"Mel Spectrogram size: {spectrogram.shape}")
        # apply frequency masking
        if random_highpass and (random.random() < 0.5):
        # highpass type mask:
            bins_to_mask = random.randint(0,40) 
            spectrogram[:, :, 0:bins_to_mask, :] = -100
        if random_bandcut and (random.random() < 0.5):
            # band mask:
            bins_to_mask = random.randint(0,16)
            bin_start = random.randint(0,48)
            spectrogram[:, :, bin_start:bins_to_mask, :] = -100
        spectrogram = self.ampToDb_t(spectrogram)

        if random_poly_cut:
            multiplier = torch.ones_like(spectrogram)
            # iterate over batches
            spec_height = spectrogram.shape[-2]
            spec_width = spectrogram.shape[-1]
            for i, spectrogram_item in enumerate(spectrogram):
                (a, b, c) = self.generate_masking_polynomial(0, 20)
                sampling_x = torch.linspace(0,1, spec_height)
                filter_samples_y = a*(torch.pow(sampling_x, 2)) + b*sampling_x + c
                filter_samples_y[filter_samples_y > 1] = 1
                spectrogram_item_multiplier = filter_samples_y.view((spec_height, 1)).repeat((1, spec_width)).view(1, spec_height, spec_width)
                multiplier[i] = spectrogram_item_multiplier
            if(inverse_poly_cut):
                multiplier = 1 - multiplier
            spectrogram = spectrogram*multiplier

        if normalize_mag:
            spectrogram -= spectrogram.min(2, keepdim=True)[0]
            m = spectrogram.max(2, keepdim=True)[0]
            m[m==0] = 1
            spectrogram /= m
            # spectrogram -= 104.02396352970787
            # spectrogram /= 26.794133651688224
            # mask_value = 0

        if raise_to_power != 1:
            spectrogram = torch.pow(spectrogram, raise_to_power)
        if clip_amplitude > 0:
            spectrogram[spectrogram<clip_amplitude]=0
        # logger.info(f"Final spectrogram size: {spectrogram.shape}")
        return spectrogram