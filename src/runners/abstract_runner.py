import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import Tuple
import os
import datetime
import logging

from src.runners.run_parameter_keys import R
from .run_parameters import RunParameters
from src.datasets.dataset_provider import DatasetProvider
from src.config import Config


logger = logging.getLogger(__name__)

class AbstractRunner:
    
    def __init__(
        self,
        model:nn.Module,
        run_params: RunParameters,
        dataset_provider: DatasetProvider = DatasetProvider(),
        config: Config = Config(),
        tensorboard_prefix = '',
    ):
        self.model = model
        self.run_params = run_params
        self.dataset_provider = dataset_provider
        self.config = Config()
        model_save_path = run_params.getd(R.MODEL_SAVE_PATH, 'default')
        model_name = os.path.basename(model_save_path)
        prefix = f"{tensorboard_prefix}-{model_name}"
        self.writer = SummaryWriter(self.find_log_folder(prefix), str(datetime.datetime.now()))
        self.validate_params()
        

    def train(self):
        train_l, train_bs, valid_l, valid_bs = self.dataset_provider.get_datasets(self.run_params)
        self.writer.add_text('model', str(self.model))
        self.writer.add_text('hyper_parameters', str(self.run_params.all_params))
        if(self.run_params.getd(R.TEST_WITH_ONE_SAMPLE, 'False') == 'True'):
            dataiter = iter(train_l)
            dx, dy = dataiter.next()
            self.test_with_one_sample(dx, dy)
        # ////////////////////////////////////////////
        # Main training loop
        # ////////////////////////////////////////////
        optimizer = self.create_optimizer()
        criterion = self.create_criterion().to(self.config.run_device)
        epochs_count = int(self.run_params.getd(R.EPOCHS, 10))
        logger.warn(f"Using: {epochs_count} epochs")
        print(self.run_params.all_params)
        measures = self.run_params.getd(R.MEASUREMENTS, 'loss')
        measures = [x.strip() for x in measures.split(',')]
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
                self.model.train(mode=True)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                inputs = inputs.to(self.config.run_device)
                labels = labels.to(self.config.run_device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss = loss.detach()
                output_cat = torch.argmax(outputs.detach(), dim=1)
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
                train_stats = f'[{epoch + 1}, {i + 1:03d}] loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}'
                pbar.set_description(train_stats)
                if i == len(train_l) - 1:
                    top_one, top_n = self.get_validation_accuracy(self.model, valid_l, self.config, topn=5)
                    valid_metrics = {
                        'top1': top_one,
                        f'top{5}': top_n
                    }
                    self.writer.add_scalars('validation_metrics', valid_metrics, global_step=iteration)
                    validation_stats = f"{train_stats} | top1: {top_one:.3%} | top5: {top_n:.3%}"
                    pbar.set_description(validation_stats)
            torch.save(self.model.state_dict(), "net.pth")
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
                inputs, labels = data
                inputs = inputs.to(config.run_device)

                # forward + backward + optimize
                output = self.model(inputs).detach()
                yb = labels.to(config.run_device).detach()

                cats = output
                top_cats = cats.topk(topn)
                target_expanded = yb.view((yb.shape[-1], 1)).expand_as(top_cats.indices).detach()
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
            pbar2.close()
        top_one_acc = predicted_correctly/predicted_total
        topk_acc = predicted_correctly_topk/predicted_total
        return (top_one_acc, topk_acc)
    
    def validate_params(self):
        self.run_params.validate_params()
        self.dataset_provider.is_valid_dataset(self.run_params.get_dataset_name())

    def create_optimizer(self) -> torch:
        optimType = self.run_params.getd(R.OPTIMIZER, 'adam')
        lr = float(self.run_params.getd(R.LR, 1e-3))
        wd = float(self.run_params.getd(R.WEIGHT_DECAY, 0))

        optim = None
        if(optimType == 'adam'):
            optim = torch.optim.Adam(self.model.parameters(), lr,
                            weight_decay=wd)
        elif(optimType == 'sgd'):
            optim = torch.optim.SGD(self.model.parameters(), lr,
                            weight_decay=wd)
        return optim
    
    def create_criterion(self):
        criterionType = self.run_params.getd(R.CRITERION, 'crossentropy')
        if criterionType == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif criterionType == 'mse':
            return nn.MSELoss()
        elif criterionType == 'nll':
            return nn.NLLLoss()

    def test_with_one_sample(self, dx, dy):
        logger.info("Should run test with a single sample..")
        logger.info(f"Shape of the sample input: {dx.shape} and sample output: {dy.shape}")
    
    def find_log_folder(self, prefix):
        folder = 'runs'
        run_i = 1
        path = f'{folder}/{prefix}-test_run{run_i}'
        while(os.path.exists(path)):
            run_i += 1
            path = f'{folder}/{prefix}-test_run{run_i}'
        logger.info(f"Writing run data to: {path}")
        return path
        
