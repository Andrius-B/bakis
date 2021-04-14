import torch
import logging
import torchaudio
import coloredlogs

class Config:
    def __init__(
        self,
        dataset_device = torch.device("cpu"),
        # run_device = torch.device("cpu")
        run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        torchaudio.set_audio_backend("sox_io")
        # to which device should the dataloader workers load the data?
        self.dataset_device = dataset_device
        # on which device should the model be placed / train or test samples moved to ?
        self.run_device = run_device

        self.sample_rate = 22050 # sample rate of the dataset
        # logging.basicConfig(level=logging.INFO)
        coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')
