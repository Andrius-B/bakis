import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
        self.writer = SummaryWriter(self.find_log_folder(tensorboard_prefix), str(datetime.datetime.now()))
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
                
                pbar.set_description(f'[{epoch + 1}, {i + 1:03d}] loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}')
            torch.save(self.model.state_dict(), "net.pth")
        print('Finished Training')
    
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
        
