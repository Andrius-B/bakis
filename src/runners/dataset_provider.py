import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

class DatasetProvider:

    def __init__(self):
        self.available_datasets = ['cifar-10', 'cifar-100']

    def is_valid_dataset(self, dataset_name: str):
        return dataset_name in self.available_datasets
    
    def get_datasets(self,
        dataset_name: str,
        batch_sizes: (int, int) = (512, 128),
        shuffle: (bool, bool) = (True, False),
    ) -> (DataLoader, int, DataLoader, int):
        if dataset_name == 'cifar-10':
            return self.get_cifar10(batch_sizes, shuffle)
        elif dataset_name == 'cifar-100':
            return self.get_cifar100(batch_sizes, shuffle)


    def get_cifar10(self,
        batch_sizes: (int, int),
        shuffle: (bool, bool)
        ) -> (DataLoader, int, DataLoader, int):
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
        batch_sizes: (int, int),
        shuffle: (bool, bool)
        ) -> (DataLoader, int, DataLoader, int):
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