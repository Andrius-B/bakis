import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import datetime
import torchaudio
import logging
import math
import random
from typing import Tuple
from .abstract_runner import AbstractRunner
from .run_parameters import RunParameters
from src.datasets.dataset_provider import DatasetProvider
from src.config import Config
from torch.optim.lr_scheduler import StepLR
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator 
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
        self.config = config
        self.spectrogram_generator = SpectrogramGenerator(self.config)
        self.train_l, self.train_bs, self.valid_l, self.valid_bs = self.dataset_provider.get_datasets(self.run_params)

        

    def train(self):
        self.writer.add_text('model', str(self.model))
        self.writer.add_text('hyper_parameters', str(self.run_params.all_params))
        if(self.run_params.getd(R.TEST_WITH_ONE_SAMPLE, 'False') == 'True'):
            dataiter = iter(self.train_l)
            dx, dy = dataiter.next()
            self.test_with_one_sample(dx, dy)


        # ////////////////////////////////////////////
        # Main training loop
        # ////////////////////////////////////////////
        lr = float(self.run_params.getd(R.LR, 1e-3))
        wd = float(self.run_params.getd(R.WEIGHT_DECAY, 0))
        optimizer = optim = torch.optim.Adam([
            { 'params': self.model.conv1.parameters() },
            { 'params': self.model.bn1.parameters() },
            { 'params': self.model.resnet_layers.parameters() },
            { 'params': self.model.layer6.parameters() },
            { 'params': self.model.classification[-1].g_constant  },
            { 'params': self.model.classification[-1].centroids  },
            { 'params': self.model.classification[-1].cluster_mass, 'weight_decay' : 0.1 }
            ], lr, weight_decay=wd)
        sheduler = StepLR(optimizer, step_size=30, gamma=0.1)
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
            try:
                self.model.train(True)
                running_loss = 0.0
                predicted_correctly = 0
                predicted_total = 0
                record_loss = 'loss' in measures
                record_acc = 'accuracy' in measures
                metrics = {}
                pbar = tqdm(enumerate(self.train_l, 0), total=len(self.train_l), leave=True)
                for i, data in pbar:
                    # with torch.no_grad():
                    samples = data["samples"]
                    samples = samples.to(self.config.run_device)
                    spectrogram = self.spectrogram_generator.generate_spectrogram(
                        samples, narrow_to=128,
                        timestretch=True, random_highpass=False,
                        random_bandcut=False, normalize_mag=True,
                        random_poly_cut=True, random_poly_cut_probability=0.35, inverse_poly_cut=False,
                        frf_mimic=False, frf_mimic_prob=0,
                        add_noise=0)
                    
                    labels = data["onehot"]
                    labels = labels.to(self.config.run_device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(spectrogram)
                    # logger.info(f"outputs: {outputs.shape} -- \n{outputs}")
                    # logger.info(f"labels: {labels.shape} -- \n {labels}")
                    loss = criterion(outputs, labels.view(labels.shape[0],))
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        loss = loss.detach()
                        # logger.warn(f"Loss after iteration: {loss}")
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
                    if math.isnan(loss.item()):
                        raise RuntimeError("stop")
                    # print statistics
                    training_summary = f'[{epoch + 1}, {i + 1:03d}] loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}'
                    pbar.set_description(training_summary)
                    if i == len(pbar) - 1:
                        # with torch.no_grad():
                        #     cluster_sizes = self.model.classification[-1].cluster_sizes
                        #     pbar.write(f"Median cluster size={torch.median(cluster_sizes)} mean={torch.mean(cluster_sizes)} min={torch.min(cluster_sizes)} max={torch.max(cluster_sizes)}")
                        k = 5
                        top_one, top_k = self.get_validation_accuracy(self.model, self.valid_l, self.config, topn = k)
                        valid_metrics = {
                            'top1': top_one,
                            f'top{k}': top_k
                        }
                        self.writer.add_scalars('validation_metrics', valid_metrics, global_step=iteration)
                        validation_summary = f"validation acc: top-1={top_one:.3%} top-{k}={top_k:.3%}"
                        pbar.set_description(f"{training_summary} | {validation_summary}")
                        sheduler.step()
                torch.save(self.model.state_dict(), "net.pth")
            except KeyboardInterrupt as e:
                temp_file = "net_c.pth"
                logger.error(f"Interrupted with control-c, saving temp file: {temp_file}")
                torch.save(self.model.state_dict(), temp_file)
                logger.error(f"Recovery file created for model parameters")
                raise KeyboardInterrupt("Canceling training - interrupted")
            except Exception as e:
                logger.error(f"Caught and exeption, skipping to next epoch")
                logger.exception(e)
        print('Finished Training')

    def get_validation_accuracy(
        self,
        net: nn.Module,
        valid_loader: DataLoader,
        config: Config,
        topn:int = 5,
    ) -> Tuple[float, float]:
        net.train(mode=False)
        predicted_correctly = 0
        predicted_total = 0
        predicted_correctly_topk = 0
        num_batches = len(valid_loader)
        with torch.no_grad():
            pbar2 = tqdm(enumerate(valid_loader), total=num_batches, leave=False)
            for _, data in pbar2:
                # spectrogram = data["spectrogram"].to(self.config.run_device)
                samples = data["samples"]
                samples = samples.to(self.config.run_device)
                spectrogram = self.spectrogram_generator.generate_spectrogram(
                        samples, narrow_to=128,
                        timestretch=False, random_highpass=False,
                        random_bandcut=False, normalize_mag=True)

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
                validation_summary = f"running validation accuracy: TOP-1:{predicted_correctly/predicted_total:.3%}, TOP-{topn}:{predicted_correctly_topk/predicted_total:.3%}"
                pbar2.set_description(validation_summary)
                # if(i > 5):
                #     break
            pbar2.close()
        top_one_acc = predicted_correctly/predicted_total
        topk_acc = predicted_correctly_topk/predicted_total
        return (top_one_acc, topk_acc)