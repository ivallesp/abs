import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from src.constants import CIFAR10_CLASSES, CIFAR100_CLASSES
import inspect
import sys
import torch


def get_dataset(name, params):
    """Looks for a dataset loader function, runs it if it exists in this module, and
    returns it if it exists. If it does not exist but the provided name is valid, the
    dataset will be downloaded.

    Args:
        name (str): name of the dataset loader function.
        params (dict): dictionary of parameters to pass to the dataset when called.


    Raises:
        ModuleNotFoundError: if the function does not exist, this exception is raised.

    Returns:
        function: the required dataset.
    """
    # Find the requested model by name
    cls_members = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
    if name not in cls_members:
        raise ModuleNotFoundError(f"Function {name} not found in module {__name__}")
    dataset_loader = cls_members[name]
    trainloader, testloader, classes, size, channels = dataset_loader(**params)
    return trainloader, testloader, classes, size, channels


def cifar10(batch_size_train=32, batch_size_test=128, n_workers=8):
    input_size = 32
    input_channels = 3
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=n_workers
    )

    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=n_workers
    )

    classes = CIFAR10_CLASSES
    return trainloader, testloader, classes, input_size, input_channels


def cifar100(batch_size_train=32, batch_size_test=128, n_workers=8):
    input_size = 32
    input_channels = 3
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR100(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=n_workers
    )

    testset = CIFAR100(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=n_workers
    )

    classes = CIFAR100_CLASSES
    return trainloader, testloader, classes, input_size, input_channels


def mnist(batch_size_train=32, batch_size_test=128, n_workers=8):
    input_size = 28
    input_channels = 1
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=n_workers
    )

    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=n_workers
    )

    classes = list(range(10))
    return trainloader, testloader, classes, input_size, input_channels
