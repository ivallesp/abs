import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from src.constants import CIFAR10_CLASSES, CIFAR100_CLASSES


def load_cifar10(batch_size_train=32, batch_size_test=128, n_workers=8):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=2
    )

    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=2
    )

    classes = CIFAR10_CLASSES
    return trainloader, testloader, classes


def load_cifar100(batch_size_train=32, batch_size_test=128, n_workers=8):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR100(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=2
    )

    testset = CIFAR100(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=2
    )

    classes = CIFAR100_CLASSES
    return trainloader, testloader, classes