# -*- coding=utf-8 -*-

import torch


class MLP(torch.nn.Sequential):

    def _linear(self, input_size, output_size, bias):
        linear = torch.nn.Linear(input_size, output_size, bias=bias)
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(linear.weight.data, gain=gain)
        torch.nn.init.constant_(linear.bias.data, 0.0)
        return linear

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        activation=None,
        bias=True,
    ):
        if activation is None:
            activation = torch.nn.ReLU
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes, ]

        layers = []
        in_size = input_size
        for out_size in hidden_sizes[1:]:
            layers.append(self._linear(in_size, out_size, bias))
            layers.append(activation())
        layers.append(self._linear(hidden_sizes[-1], output_size, bias))
        super(MLP, self).__init__(*layers)
