from torch.utils import data
from torchvision import datasets, transforms
from .imagenet import ImageNet1KDataset
from torchvision.transforms import AutoAugment
from torchvision.transforms import AutoAugmentPolicy
import os

from timm.data.transforms_factory import create_transform


def get_SVHN(root):
    input_size = 224
    num_classes = 10

    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.SVHN(
        root + "/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_SVHN_testset(root):
    input_size = 224
    num_classes = 10

    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return test_dataset


def get_CIFAR10(root):
    data_config = {
        'input_size': 224,
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }

    train_dataset = datasets.CIFAR10(root + "/CIFAR10", train=True, download=True)
    test_dataset = datasets.CIFAR10(root + "/CIFAR10", train=False, download=False)

    return train_dataset, test_dataset, data_config


def get_CIFAR10_testset(root):
    input_size = 224
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize])
    test_dataset = datasets.CIFAR10(root + "/CIFAR10", train=False, transform=test_transform, download=False)

    return test_dataset


def get_CIFAR100(root):
    data_config = {
        'input_size': 224,
        'num_classes': 100,
        'mean': (0.5071, 0.4866, 0.4409),
        'std': (0.2673, 0.2564, 0.2762)
    }

    train_dataset = datasets.CIFAR100(root + "/CIFAR100", train=True, download=True)
    test_dataset = datasets.CIFAR100(root + "/CIFAR100", train=False, download=False)

    return train_dataset, test_dataset, data_config


def get_CIFAR100_testset(root):
    input_size = 224
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize])
    test_dataset = datasets.CIFAR100(root + "/CIFAR100", train=False, transform=test_transform, download=True)

    return test_dataset


def get_ImageNet1K(root):
    data_config = {
        'input_size': 224,
        'num_classes': 1000,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }

    train_dataset = ImageNet1KDataset(root, 'train', transform=None)
    val_dataset = ImageNet1KDataset(root, 'val', transform=None)

    # return input_size, num_classes, train_dataset, val_dataset
    return train_dataset, val_dataset, data_config


all_datasets = {
    "SVHN": get_SVHN,
    "SVHN/test": get_SVHN_testset,
    "CIFAR10": get_CIFAR10,
    "CIFAR10/test": get_CIFAR10_testset,
    "CIFAR100": get_CIFAR100,
    "CIFAR100/test": get_CIFAR100_testset,
    "ImageNet1K": get_ImageNet1K,
}


def get_dataset(dataset, root="./"):
    return all_datasets[dataset](root)
