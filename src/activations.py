import torch
import inspect
import sys
from torch.nn.functional import (
    relu,
    leaky_relu,
    softplus,
    silu,
    elu,
    selu,
    gelu,
    sigmoid,
    tanh,
)


swish = silu


def get_activation(name):
    """Looks for a activation function and returns it if it exists in this module.

    Args:
        name (str): name of the activation function.

    Raises:
        ModuleNotFoundError: if the function does not exist, this exception is raised.

    Returns:
        function: the required activation function.
    """
    # Find the requested model by name
    cls_members = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
    if name not in cls_members:
        raise ModuleNotFoundError(f"Function {name} not found in module {__name__}")
    activation = cls_members[name]

    return activation


def logcosh4(x):
    return _logcosh(x=x, alpha=4.0)


def absolute(x):
    return torch.abs(x)


def _logcosh(x, alpha):
    return torch.log(torch.cosh(alpha * x))
