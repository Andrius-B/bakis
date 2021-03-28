from src.config import Config
from torch import Tensor
import torchaudio
import torchvision
import random
import torch
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
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
        self.poly_cut_cache_size = 100
        self.poly_cut_cache = {}
    
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

    def generate_non_differentiable_mask(self, start, stop, steps, polynomials):
        cache_index = random.randint(0, self.poly_cut_cache_size)
        if cache_index in self.poly_cut_cache and random.random() < 70:
            return self.poly_cut_cache[cache_index]
        x = torch.linspace(start, stop, steps)
        y = torch.ones_like(x)
        for i in range(polynomials):
            a,b,c = self.generate_masking_polynomial(0, 20)
            poly_y = a*(x**2) + b*x + c
            poly_y[poly_y>1] = 1
            y = torch.minimum(y, poly_y)
        self.poly_cut_cache[cache_index] = y
        return y

    def generate_mic_frf_mask(self, start_freq: int, end_freq: int, num_samples: int):
        points = [
            (20, 0.4),
            (100, 0.4),
            (start_freq, 0.9),
            (start_freq + 100, 0.9),
            (start_freq + 200, 0.9),
        ]
        freq = start_freq
        while freq < end_freq - 100:
            freq += 1000
            points.append((freq, 0.9))
        points.extend([
            (end_freq - 100, 0.9),
            (end_freq + 1000, 0.4),
            (end_freq + 2000, 0.3),
        ])
        freq = end_freq + 2000
        while freq < 20000:
            freq += 3000
            points.append((freq, 0.3))
        keys = set()
        points_final = []
        for p in points:
            if p[0] not in keys:
                points_final.append(p)
            keys.add(p[0])
        p = np.array(points_final)
        x_p, y_p = (p[:,0]/20000), p[:, 1]
        freq_mask_f = interpolate.interp1d(x_p, y_p, kind='cubic')
        x = np.linspace(0.001, 1, num_samples)
        mask_samples = freq_mask_f(x)
        mask_samples = torch.Tensor(mask_samples)
        mask_samples = mask_samples.view(-1, 1)
        return mask_samples

    def generate_spectrogram(
            self, samples: Tensor,
            narrow_to=-1, timestretch=False,
            random_highpass=False, random_bandcut=False,
            random_poly_cut=False, random_poly_cut_probability=1, inverse_poly_cut=False, # inverse here means that only the poly is left as was
            normalize_mag=True,
            clip_amplitude = 0, raise_to_power = 1,
            frf_mimic=False, frf_mimic_prob=1,
            add_noise=0,
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
        # print(f"Spectrogram stats: min={spectrogram.min()} max={spectrogram.max()} median={spectrogram.median()}")
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

        if frf_mimic and random.random() < frf_mimic_prob:
            mask_samples = self.generate_mic_frf_mask(random.randint(101, 1500), random.randint(5000, 10000), spectrogram.shape[-2])
            # print(f"Generated samples: {mask_samples.shape} -- \n {mask_samples}")
            # mask_samples -= torch.randn_like(mask_samples)/10
            # mask_samples = torch.rand_like(mask_samples)
            mask_samples = mask_samples.to(self.config.run_device)
            # plt.figure()
            # plt.plot(np.linspace(0, 1, mask_samples.shape[-2]), mask_samples.cpu().view(-1))

            mask_samples = mask_samples.view(-1, 1)
            mask_samples[mask_samples < 0.1] = 0.1
            mask_samples[mask_samples > 0.9] = 0.9
            # print(f"Using mimicing mask: {mask_samples.shape} -- \n{mask_samples}")
            # print(f"Test spectrogram: {spectrogram.shape} -- \n{spectrogram}")
            spectrogram = spectrogram * mask_samples

        if random_poly_cut and random.random() < random_poly_cut_probability:
            multiplier = torch.ones_like(spectrogram)
            # iterate over batches
            spec_height = spectrogram.shape[-2]
            spec_width = spectrogram.shape[-1]
            for i, spectrogram_item in enumerate(spectrogram):
                filter_samples_y = self.generate_non_differentiable_mask(0, 1, spec_height, 5)
                # filter_samples_y = filter_samples_y * 0.5
                spectrogram_item_multiplier = filter_samples_y.view((spec_height, 1)).repeat((1, spec_width)).view(1, spec_height, spec_width)
                multiplier[i] = spectrogram_item_multiplier
            if(inverse_poly_cut):
                multiplier = 1 - multiplier
            spectrogram = spectrogram*multiplier
        if add_noise > 0:
            spectrogram *= torch.ones_like(spectrogram) - torch.randn_like(spectrogram)*(add_noise)
        if normalize_mag:
            spectrogram = spectrogram - spectrogram.mean()
            spectrogram = spectrogram / spectrogram.std()
            spectrogram -= spectrogram.min(2, keepdim=True)[0]
            m = spectrogram.max(2, keepdim=True)[0]
            m[m==0] = 1
            spectrogram /= m

        if raise_to_power != 1:
            spectrogram = torch.pow(spectrogram, raise_to_power)
        if clip_amplitude > 0:
            spectrogram[spectrogram<clip_amplitude]=0
        # logger.info(f"Final spectrogram size: {spectrogram.shape}")
        return spectrogram