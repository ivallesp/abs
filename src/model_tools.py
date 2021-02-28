import torch
import random
import numpy as np
from torch import nn


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_weights(net):
    # https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
    for name, param in net.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param)
        elif "emb" in name:
            nn.init.kaiming_normal_(param)