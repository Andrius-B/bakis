import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.datasets.diskds.diskds_provider import DiskDsProvider
from src.datasets.diskds.sqliteds_provider import SQLiteDsProvider
from src.runners.run_parameter_keys import R
import src.runners.run_parameters
from typing import Tuple
import torchvision.datasets as datasets

class DatasetProvider:

    def __init__(self):
        self.available_datasets = ['cifar-10', 'cifar-100']

    def is_valid_dataset(self, dataset_name: str):
        return dataset_name in self.available_datasets
    
    def get_datasets(self,
        run_params,
        # batch_sizes: (int, int) = (512, 128),
        # shuffle: (bool, bool) = (True, False),
    ) -> Tuple[DataLoader, int, DataLoader, int]:
        batch_sizes = (int(run_params.getd(R.BATCH_SIZE_TRAIN, '512')), int(run_params.getd(R.BATCH_SIZE_VALIDATION, '128')))
        shuffle = (bool(run_params.getd(R.SHUFFLE_TRAIN, 'True')), bool(run_params.getd(R.SHUFFLE_VALIDATION, 'False')))
        dataset_name = run_params.get_dataset_name()
        if dataset_name == 'cifar-10':
            return self.get_cifar10(batch_sizes, shuffle)
        elif dataset_name == 'cifar-100':
            return self.get_cifar100(batch_sizes, shuffle)
        elif dataset_name.startswith("disk-ds"):
            return DiskDsProvider(run_params).get_disk_dataset(batch_sizes, shuffle)
        elif dataset_name.startswith("sqlite-ds"):
            return SQLiteDsProvider().get_sqlite_dataset(run_params, batch_sizes, shuffle)
            

    def get_cifar10(self,
        batch_sizes: Tuple[int, int],
        shuffle: Tuple[bool, bool]
        ) -> Tuple[DataLoader, int, DataLoader, int]:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/cifar10', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_sizes[0], shuffle=shuffle[0],
            num_workers=5, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_sizes[1], shuffle=shuffle[1],
            num_workers=5, pin_memory=True)
        return (train_loader, batch_sizes[0], val_loader, batch_sizes[1])

    def get_cifar100(self,
        batch_sizes: Tuple[int, int],
        shuffle: Tuple[bool, bool]
        ) -> Tuple[DataLoader, int, DataLoader, int]:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data/cifar100', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_sizes[0], shuffle=shuffle[0],
            num_workers=5, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data/cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_sizes[1], shuffle=shuffle[1],
            num_workers=5, pin_memory=True)
        return (train_loader, batch_sizes[0], val_loader, batch_sizes[1])