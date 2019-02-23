#!/usr/bin/env python3

import torch as th
import numpy as np
import torch.nn as nn


def kostrikov_control_(module, gain=None):
    with th.no_grad():
        if gain is None:
            gain = np.sqrt(2.0)
        nn.init.orthogonal_(module.weight.data, gain=gain)
        nn.init.constant_(module.bias.data, 0.0)
        return module


def atari_init_(module, gain=None):
    if gain is None:
        gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0.0)
    return module
