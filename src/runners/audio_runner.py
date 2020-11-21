import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import datetime
import torchaudio
import logging
import random
from typing import Tuple
from .abstract_runner import AbstractRunner
from .run_parameters import RunParameters
from src.datasets.dataset_provider import DatasetProvider
from src.config import Config
from src.runners.run_parameter_keys import R


logger = logging.getLogger(__name__)

class AudioRunner(AbstractRunner):
    
    def __init__(
        self,
        model:nn.Module,
        run_params: RunParameters,
        dataset_provider: DatasetProvider = DatasetProvider(),
        config: Config = Config(),
        tensorboard_prefix = '',
    ):
        super().__init__(model, run_params, dataset_provider, config, tensorboard_prefix)
        self.model = model
        self.run_params = run_params
        self.dataset_provider = dataset_provider
        self.config = Config()
        self.writer = SummaryWriter(self.find_log_folder(tensorboard_prefix), str(datetime.datetime.now()))
        

    def train(self):
        train_l, train_bs, valid_l, valid_bs = self.dataset_provider.get_datasets(self.run_params)
        self.writer.add_text('model', str(self.model))
        self.writer.add_text('hyper_parameters', str(self.run_params.all_params))
        if(self.run_params.getd(R.TEST_WITH_ONE_SAMPLE, 'False') == 'True'):
            dataiter = iter(train_l)
            dx, dy = dataiter.next()
            self.test_with_one_sample(dx, dy)

        spectrogram_t = torchaudio.transforms.Spectrogram(
            n_fft=2048, win_length=2048, hop_length=512, power=None
        ).to(self.config.run_device) # generates a complex spectrogram
        time_stretch_t = torchaudio.transforms.TimeStretch(hop_length=512, n_freq=1025).to(self.config.run_device)
        low_pass_t = torchaudio.functional.lowpass_biquad
        high_pass_t = torchaudio.functional.highpass_biquad
        mel_t = torchaudio.transforms.MelScale(n_mels=129, sample_rate=44100).to(self.config.run_device)
        norm_t = torchaudio.transforms.ComplexNorm(power=2).to(self.config.run_device)
        ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(self.config.run_device)


        # ////////////////////////////////////////////
        # Main training loop
        # ////////////////////////////////////////////
        optimizer = self.create_optimizer()
        criterion = self.create_criterion().to(self.config.run_device)
        epochs_count = int(self.run_params.getd(R.EPOCHS, 10))
        logger.info(f"Using: {epochs_count} epochs")
        logger.debug(self.run_params.all_params)
        measures = self.run_params.getd(R.MEASUREMENTS, 'loss')
        measures = [x.strip() for x in measures.split(',')]
        # logger.info(self.model)
        self.model.to(self.config.run_device)
        iteration = 0
        for epoch in range(epochs_count):  # loop over the dataset multiple times
            self.model.train(True)
            running_loss = 0.0
            predicted_correctly = 0
            predicted_total = 0
            record_loss = 'loss' in measures
            record_acc = 'accuracy' in measures
            metrics = {}
            pbar = tqdm(enumerate(train_l, 0), total=len(train_l), leave=True)
            for i, data in pbar:
                samples = data["samples"]

                samples = samples.to(self.config.run_device)
                spectrogram = spectrogram_t(samples)
                if(random.random() > 0.5): # half of the samples get a timestretch
                    # this is because
                    spectrogram = time_stretch_t(spectrogram, random.uniform(0.93, 1.07))
                spectrogram = spectrogram.narrow(3, 0, 129)
                # print(f"Spectrogram after slowing down: {spectrogram.shape}")
                spectrogram = norm_t(spectrogram)
                spectrogram = mel_t(spectrogram)
                spectrogram = ampToDb_t(spectrogram)

                labels = data["onehot"]
                labels = labels.to(self.config.run_device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(spectrogram)
                # logger.info(f"outputs: {outputs.shape}")
                # logger.info(f"labels: {labels.shape}")
                loss = criterion(outputs, labels.view(labels.shape[0],))
                loss.backward()
                optimizer.step()

                loss = loss.detach()
                output_cat = torch.argmax(outputs.detach(), dim=1)
                labels = labels.view(labels.shape[0],)
                correct = labels.eq(output_cat).detach()
                predicted_correctly += correct.sum().item()
                predicted_total += correct.shape[0]
                running_loss += loss.item()

                if record_loss:
                    metrics['loss'] = loss.item()
                    # metrics['running_loss'] = running_loss
                if record_acc:
                    metrics['accuracy'] = correct.sum().item()/correct.shape[0]
                    # metrics['accuracy_running'] = predicted_correctly/predicted_total
                self.writer.add_scalars('train_metrics', metrics, global_step=iteration)
                iteration += 1
                # print statistics
                
                pbar.set_description(f'[{epoch + 1}, {i + 1:03d}] loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}')
            self.get_validation_accuracy(self.model, valid_l, self.config)
            torch.save(self.model.state_dict(), "net.pth")
        print('Finished Training')

    def test_with_one_sample(self, dx, dy):
        logger.info("Should run test with a single sample..")
        logger.info(f"Shape of the sample input: {dx.shape} and sample output: {dy.shape}")

    def get_validation_accuracy(self, net: nn.Module, valid_loader: DataLoader, config: Config, topn:int = 5) -> Tuple[float, float]:
        net.train(mode=False)
        predicted_correctly = 0
        predicted_total = 0
        predicted_correctly_topk = 0
        num_batches = len(valid_loader)
        spectrogram_t = torchaudio.transforms.Spectrogram(
            n_fft=2048, win_length=2048, hop_length=512, power=None
        ).to(config.run_device) # generates a complex spectrogram
        # time_stretch_t = torchaudio.transforms.TimeStretch(hop_length=512, n_freq=1025).to(Config.run_device)
        # high_pass_t = torchaudio.functional.highpass_biquad
        mel_t = torchaudio.transforms.MelScale(n_mels=129, sample_rate=44100).to(config.run_device)
        norm_t = torchaudio.transforms.ComplexNorm(power=2).to(config.run_device)
        ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(config.run_device)

        pbar2 = tqdm(enumerate(valid_loader), total=num_batches, leave=False)
        for i, data in pbar2:
            
            samples = data["samples"]

            samples = samples.to(self.config.run_device)
            spectrogram = spectrogram_t(samples)
            spectrogram = spectrogram.narrow(3, 0, 129)
            # print(f"Spectrogram after slowing down: {spectrogram.shape}")
            spectrogram = norm_t(spectrogram)
            spectrogram = mel_t(spectrogram)
            spectrogram = ampToDb_t(spectrogram)

            labels = data["onehot"]
            labels = labels.to(self.config.run_device)

            # forward + backward + optimize
            output = self.model(spectrogram).detach()
            yb = labels.to(config.run_device).detach()

            cats = output
            top_cats = cats.topk(topn)
            target_expanded = yb.expand_as(top_cats.indices).detach()
            topk_correct = target_expanded.eq(top_cats.indices)
            # print("==============")
            # print(f"Cats shape: {cats.shape}")
            # print(f"top_cats shape: {top_cats.indices}")
            # print(f"Target: {target_expanded}")
            # print(f"Categories equal: {topk_correct}")
            # print(f"TOP-N correct in batch: {topk_correct.sum()}")
            predicted_correctly_topk += topk_correct.sum().item()
            output_cat = torch.argmax(output, dim=1)
            target = yb.detach().view(-1)
            diff = (target - output_cat).detach()
            correct_predictions_in_batch = (diff == 0).sum().item()
            predicted_total += len(target)
            predicted_correctly += correct_predictions_in_batch
            pbar2.set_description(f"running validation accuracy: TOP-1:{predicted_correctly/predicted_total:.3%}, TOP-{topn}:{predicted_correctly_topk/predicted_total:.3%}")
            # if(i > 5):
            #     break
        pbar2.close()
        return (predicted_correctly/predicted_total, predicted_correctly_topk/predicted_total)