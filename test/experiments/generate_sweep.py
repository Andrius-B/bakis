import numpy as np
import matplotlib.pyplot as plt
import math
import torchaudio
import torch

if __name__ == "__main__":
    sample_rate = 41000
    len_seconds = 20
    len_frames = sample_rate*len_seconds
    x = np.linspace(0, len_seconds, len_frames)
    frequency = np.linspace(6, 50, len_frames)
    frequency = frequency*frequency
    y = np.sin(frequency * x *2*math.pi)
    samples = torch.tensor(y).view((1, -1))
    # plt.plot(x, y)
    # plt.show()
    # print(samples.shape)
    # print(frequency.max())
    torchaudio.backend.sox_backend.save("chirp.mp3", samples, sample_rate=sample_rate)