import torch
import torch.nn as nn
from src.activations import get_activation
from typing import cast, List


class VGG16(nn.Module):
    def __init__(
        self,
        n_outputs,
        activation_name,
        input_channels,
        init_weights=False,
        *args,
        **kwarfs
    ) -> None:
        super(VGG16, self).__init__()
        act = _make_activation(activation_name)
        self.features = make_layers(
            [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                # "M",
            ],
            activation_name,
            in_channels=input_channels,
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            act(),
            nn.Linear(4096, 4096),
            act(),
            nn.Linear(4096, n_outputs),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _make_activation(name):
    act_f = get_activation(name)

    class Activation(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation_f = act_f

        def forward(self, x):
            return self.activation_f(x)

    Activation.__name__ = name
    return Activation


def make_layers(cfg, activation_name, in_channels):
    act = _make_activation(activation_name)()
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)
