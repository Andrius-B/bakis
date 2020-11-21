import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from scipy import signal


def generate_spectrogram(samples, sample_rate):
    nperseg = 256
    window = signal.windows.triang(nperseg)
    frequencies, times, spectrogram = signal.spectrogram(
        samples,
        44100,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg//8,
        nfft=nperseg,
        detrend=False,
        return_onesided=True,
        scaling="density",
        mode="magnitude"
    )
    if(np.amin(spectrogram) > 0):
        spectrogram = np.log(spectrogram)
        return (frequencies, times, spectrogram)
    else:
        spectrogram = np.log(spectrogram+0.0000001)
        return (frequencies, times, spectrogram)


# trying to replicate:
# https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
if __name__ == "__main__":
    filename = 'test/test_data/2 - Go.flac'
    y, sr = librosa.core.load(
        filename,
        sr=None,
        mono=True,
        offset=5,
        duration=2**16/44100
    )
    fig = plt.figure(figsize=(16, 12))

    nperseg = 2048
    window = signal.windows.triang(nperseg)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Waveform")
    librosa.display.waveplot(y, sr=sr, ax=ax1)
    ax2 = fig.add_subplot(2, 2, 2)
    n_fft = nperseg
    n_mels = 129
    hop_length = int(n_fft/4)
    D = np.abs(librosa.stft(y, n_fft=n_fft, window=window,
                            hop_length=hop_length, center=False))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    print(f"Spectrogram shape: {DB.shape}")
    ax2.set_title("Normal Spectrogram")
    # ax2.pcolormesh(DB, cmap='magma')
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='log', ax=ax2)

    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    print(f"Mel spectrogram shape: {S_DB.shape}")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Mel Spectrogram")
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', ax=ax3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Previuosly used reference spectrogram")
    frequencies, times, s_old = generate_spectrogram(y, sr)
    # cmap = matplotlib.cm.get_cmap('magma')
    print(f"Old spectrogram shape: {s_old.shape}")
    ax4.pcolormesh(times, frequencies, s_old, cmap='magma')
    # librosa.display.specshow(s_old, sr=sr, hop_length=hop_length,
    #                          x_axis='time', y_axis='log', ax=ax2)
    # plt.colorbar()
    plt.show()

    y, sr = librosa.core.load(
        filename,
        sr=None,
        mono=True,
        offset=0,
        duration=15
    )
    fig = plt.figure(figsize=(6, 6))

    nperseg = 2048
    window = signal.windows.triang(nperseg)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("File Waveform")
    librosa.display.waveplot(y, sr=sr, ax=ax1)
    plt.show()
