import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import inspect
import sys

from src.activations import get_activation


def get_model(name, params):
    """Looks for a model class with the specified name, instantiates it with the
    provided parameters and defines the loss and the optimizer.

    Args:
        name (str): name of the model, defined as a class with that same name in the
        src.models module.
        params (dict): dictionary of parameters to pass to the model when instantiated.

    Raises:
        ModuleNotFoundError: if the model does not exist, this exception is raised.

    Returns:
        tuple: model, loss function and optimizer.
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
    optimizer = optim.RMSprop(net.parameters(), lr=0.0001)
    return net, criterion, optimizer


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
