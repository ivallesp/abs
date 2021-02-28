import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


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

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainloader, testloader, classes
