import torchaudio
import random
import torch

class FileLoadingSoxEffects(torch.nn.Module):
    def __init__(self, initial_sample_rate=44100, final_sample_rate=44100, random_pre_resampling=False):
        super().__init__()
        self.random_pre_resampling = random_pre_resampling
        self.effects_before = [
            ["remix", "-"],
            ["gain", "-n"],
        ]

        self.effects_after = []
        if initial_sample_rate != final_sample_rate or random_pre_resampling:
            self.effects_after.extend([
                ["rate", str(final_sample_rate)],
                ["pad", "0", "0.3"]
            ])
        self.initial_sample_rate = initial_sample_rate

    def forward(self, samples: torch.Tensor):
        effects = [*self.effects_before]
        if random.random() < 0.5:
            if self.random_pre_resampling:
                effects.append(["rate", str(random.randint(16000, 22000))])
            effects.extend(self.effects_after)
        return torchaudio.sox_effects.apply_effects_tensor(
            samples, self.initial_sample_rate, effects)