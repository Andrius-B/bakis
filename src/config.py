import torch
import logging

class Config:
    def __init__(
        self,
        dataset_device = torch.device("cpu"),
        run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        # to which device should the dataloader workers load the data?
        self.dataset_device = dataset_device
        # on which device should the model be placed / train or test samples moved to ?
        self.run_device = run_device
        logging.basicConfig(level=logging.INFO)
