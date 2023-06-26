# -*- coding=utf-8 -*-

import torch


class MLP(torch.nn.Sequential):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    Implements a simple multi-layer perceptron.

    ## Example

    ~~~python
    net = MLP(128, 1, [1024, 1024], activation=torch.nn.GELU)
    ~~~
    """

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
        """
        ## Arguments

        * `input_size` (int) - Input size of the MLP.
        * `output_size` (int) - Number of output units.
        * `hidden_sizes` (list of int) - Each int is the number of hidden units of a layer.
        * `activation` (callable) - Activation function to use for the MLP.
        * `bias` (bool, *optional*, default=True) - Whether the MLP uses bias terms.
        """
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
