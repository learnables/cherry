#!/usr/bin/env python3

import torch as th
import numpy as np
import torch.nn as nn


def pong_control_(module, bias=0.1):
    weight = module.weight
    size = weight.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    weight.data.uniform_(-bound, bound)
    module.bias.data.fill_(bias)
    return module


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
