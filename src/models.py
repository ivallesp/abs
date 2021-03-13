import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import inspect
import sys
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# from src.mobilenetV2 import MobileNetV2
from src.vgg import VGG16
from src.activations import get_activation
from src.model_tools import GradualWarmupScheduler


def get_model(name, params, n_epochs):
    """Looks for a model class with the specified name, instantiates it with the
    provided parameters and defines the loss and the optimizer.

    Args:
        name (str): name of the model, defined as a class with that same name in the
        src.models module.
        params (dict): dictionary of parameters to pass to the model when instantiated.
        n_epochs(int): number of epochs to run. Used to adjust the lr_scheduler

    Raises:
        ModuleNotFoundError: if the model does not exist, this exception is raised.

    Returns:
        tuple: model, loss function, optimizer and lr_scheduler.
    """
    # Find the requested model by name
    cls_members = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))
    if name not in cls_members:
        raise ModuleNotFoundError(f"Class {name} not found in module {__name__}")
    model_class = cls_members[name]

    # Instantiate the model
    net = model_class(**params)

    # Define the loss and the optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=n_epochs + 1, eta_min=1e-6, last_epoch=-1, verbose=False
    )
    lr_scheduler = GradualWarmupScheduler(
        optimizer, multiplier=10, total_epoch=5, after_scheduler=lr_scheduler
    )

    return net, criterion, optimizer, lr_scheduler


class TestNet(nn.Module):
    def __init__(self, n_outputs, activation_name):
        super(TestNet, self).__init__()
        self.activation = get_activation(activation_name)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv6(nn.Module):
    def __init__(self, n_outputs, activation_name, input_size, input_channels):
        super(Conv6, self).__init__()
        self.flatten_img_dim = int(input_size / 2 / 2 / 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = get_activation(activation_name)
        self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.fc1 = nn.Linear(256 * self.flatten_img_dim ** 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        h = self.activation(self.conv1_1(x))
        h = self.pool(self.activation(self.conv1_2(h)))

        h = self.activation(self.conv2_1(h))
        h = self.pool(self.activation(self.conv2_2(h)))

        h = self.activation(self.conv3_1(h))
        h = self.pool(self.activation(self.conv3_2(h)))
        h = h.view(-1, 256 * self.flatten_img_dim ** 2)

        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.fc3(h)
        return h


class Conv4(nn.Module):
    def __init__(self, n_outputs, activation_name, input_size, input_channels):
        super(Conv4, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_img_dim = int(input_size / 2 / 2)

        self.activation = get_activation(activation_name)
        self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))

        self.fc1 = nn.Linear(128 * self.flatten_img_dim ** 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        h = self.activation(self.conv1_1(x))
        h = self.pool(self.activation(self.conv1_2(h)))

        h = self.activation(self.conv2_1(h))
        h = self.pool(self.activation(self.conv2_2(h)))

        h = h.view(-1, 128 * self.flatten_img_dim ** 2)

        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.fc3(h)
        return h


class Conv2(nn.Module):
    def __init__(self, n_outputs, activation_name, input_size, input_channels):
        super(Conv2, self).__init__()
        self.flatten_img_dim = int(input_size / 2)

        self.pool = nn.MaxPool2d(2, 2)
        self.activation = get_activation(activation_name)
        self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.fc1 = nn.Linear(64 * self.flatten_img_dim ** 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        h = self.activation(self.conv1_1(x))
        h = self.pool(self.activation(self.conv1_2(h)))

        h = h.view(-1, 64 * self.flatten_img_dim ** 2)

        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.fc3(h)
        return h


class FC(nn.Module):
    def __init__(self, n_outputs, activation_name, input_size, input_channels):
        super(FC, self).__init__()
        self.activation = get_activation(activation_name)
        self.input_size = (input_size ** 2) * input_channels
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        h = x.view(x.shape[0], -1)
        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.fc3(h)
        return h
