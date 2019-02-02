#!/usr/bin/env python3

import torch.nn as nn
from cherry.nn import ControlLinear


class ControlMLP(nn.Module):

    def __init__(self, input_size, output_size, layer_sizes=None):
        super(ControlMLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        if len(layer_sizes) > 0:
            layers = [ControlLinear(input_size, layer_sizes[0]),
                      nn.Tanh()]
            for in_, out_ in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(ControlLinear(in_, out_))
                layers.append(nn.Tanh())
            layers.append(ControlLinear(layer_sizes[-1], output_size))
        else:
            layers = [ControlLinear(input_size, output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Actor(ControlMLP):

    def __init__(self, input_size, output_size, layer_sizes=None):
        super(ControlMLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        if len(layer_sizes) > 0:
            layers = [ControlLinear(input_size, layer_sizes[0]),
                      nn.Tanh()]
            for in_, out_ in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(ControlLinear(in_, out_))
                layers.append(nn.Tanh())
            layers.append(ControlLinear(layer_sizes[-1],
                                        output_size,
                                        gain=1.0))
        else:
            layers = [ControlLinear(input_size, output_size, gain=1.0)]
        self.layers = nn.Sequential(*layers)
