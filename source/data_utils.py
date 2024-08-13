import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader

import CONFIG
from source.CIFAR10C import CIFAR10C
from source.CIFAR100C import CIFAR100C
from source.SVHN import SVHN


def get_all_severity_dataloader(root, max_corrupt_level, data_name,
                            batch_size=16, num_workers=4, shuffle=True, eval=True):
    """
    Returns dataloaders for all severity levels (0 ~ max severity)

    Args:
        root (str): Root directory of the dataset.
        max_corrupt_level (int): Maximum severity level of corruption to generate dataloaders.
        data_name (str): Name of the dataset (e.g., cifar10, cifar100, imagenet).
        batch_size (int): Batch size for the dataloader.
        corrupt_type (str): Type of corruption applied to the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        shuffle (bool): Whether to shuffle the data.
        eval (bool): Indicates if the dataloaders are for evaluation or tuning.

    Returns:
        list: A list of torch DataLoader objects, one for each severity level.
    """

    all_loaders = []
    # level 0
    test_loader = get_dataloader(
        root = root,
        data_name=data_name,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        eval=eval
    )
    all_loaders.append(test_loader)

    # level 1~5
    for level in range(max_corrupt_level):
        type_data = {}
        for c_type in CONFIG.CORRUPTION_TYPE:
            type_data[c_type] = get_dataloader( # 1~5 each
                root = root,
                data_name=data_name+'C', ## corrupt dataname
                corrupt_level=level,
                batch_size=batch_size,
                corrupt_type = c_type,
                num_workers=num_workers,
                shuffle=shuffle,
                eval=eval
            ) 
        all_loaders.append(type_data)

    return all_loaders


def get_dataloader(root, data_name, corrupt_level=0, batch_size=16,
                   corrupt_type=None, num_workers=4, shuffle=True, eval=False, num_sample=None):
    """
    Returns a dataloader for a specified dataset.

    Args:
        root (str): Root directory of the dataset.
        data_name (str): Name of the dataset (e.g., cifar10, cifar100, imagenet, cifar10C, cifar100C, imagenetC).
        corrupt_level (int): Sepecify the severity level of corruption (0 to 5). 0 indicates in-distribution data with no corruption.
        batch_size (int): Batch size for the dataloader.
        corrupt_type (str): Type of corruption applied to the dataset (if applicable).
        num_workers (int): Number of subprocesses to use for data loading.
        shuffle (bool): Whether to shuffle the data.
        eval (bool): Indicates if the dataloader is for tuning or evaluation.
        num_sample (int): Number of semantic OOD samples for tuning.

    Returns:
        torch.utils.data.DataLoader: The configured dataloader object.
    """

    if data_name.lower() == 'cifar10':
        transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4821, 0.4464), (0.2472, 0.2437, 0.2617))])
        
        if eval:
            dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        else: # for tuning
            dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
            dataset = torch.utils.data.Subset(dataset, range(45000, 50000)) # only validation set


    elif data_name.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        
        if eval:
            dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

        else: # for tuning
            dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
            dataset = torch.utils.data.Subset(dataset, range(45000, 50000)) # only validation set
    

    elif data_name.lower() == 'cifar10c': # corrupted dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4821, 0.4464), (0.2472, 0.2437, 0.2617))])
        
        dataset = CIFAR10C(root=os.path.join(root, 'cifar10C'), level=corrupt_level, c_type=corrupt_type, transform=transform)

    elif data_name.lower() == 'cifar100c':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        
        dataset = CIFAR100C(root=os.path.join(root, 'cifar100C'), level=corrupt_level, c_type=corrupt_type, transform=transform)


    elif data_name.lower() == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4821, 0.4464), (0.2472, 0.2437, 0.2617))])
        
        dataset = SVHN(root=os.path.join(root, 'svhn'), split='test', transform=transform)

        if not eval:
            dataset = torch.utils.data.Subset(dataset, range(num_sample))


    elif data_name.lower() == 'texture':
        transform=transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32),
            transforms.ToTensor(), transforms.Normalize((0.4915, 0.4821, 0.4464), (0.2472, 0.2437, 0.2617))])
        
        dataset = dset.ImageFolder(root=os.path.join(root, "dtd"), transform=transform)

        if not eval:
            dataset = torch.utils.data.Subset(dataset, range(num_sample))


    elif data_name.lower() == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        dataset = torchvision.datasets.ImageNet(root=os.path.join(root, 'imagenet'),
                                                split='val', transform=transform)

        if eval:
            dataset = torch.utils.data.Subset(dataset, range(12500, 37500))

        else:
            dataset = torch.utils.data.Subset(dataset, range(12500))

    elif data_name.lower() == 'imagenetc':
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        dataset = dset.ImageFolder(
            root=os.path.join(root, 'imagenetC'),
            transform=transform)

        dataset = torch.utils.data.Subset(dataset, range(CONFIG.NUM_CORRUPTED))

    elif data_name == 'imagenetR':
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        dataset = dset.ImageFolder(
            root=os.path.join(root, 'imagenetR'),
            transform=transform)
    
    elif data_name == 'imagenetA':
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        dataset = dset.ImageFolder(
            root=os.path.join(root, 'imagenetA'),
            transform=transform)


    elif data_name == 'imagenetS':
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        dataset = dset.ImageFolder(
            root=os.path.join(root, 'imagenetS'),
            transform=transform)

    else:
        print('not supported dataset')
        sys.exit()

    dataset_loader = DataLoader(
        dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return dataset_loader
