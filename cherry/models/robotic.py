#!/usr/bin/env python3

import torch.nn as nn
from cherry.nn import RoboticLinear


class RoboticMLP(nn.Module):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotic.py)

    **Description**

    A multi-layer perceptron with proper initialization for robotic control.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **output_size** (int) - Size of output.
    * **layer_sizes** (list, *optional*, default=None) - A list of ints,
      each indicating the size of a hidden layer.
      (Defaults to two hidden layers of 64 units.)

    **Example**
    ~~~python
    target_qf = ch.models.robotic.RoboticMLP(23,
                                             34,
                                             layer_sizes=[32, 32])
    ~~~
    """

    def __init__(self, input_size, output_size, layer_sizes=None):
        super(RoboticMLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        if len(layer_sizes) > 0:
            layers = [RoboticLinear(input_size, layer_sizes[0]),
                      nn.Tanh()]
            for in_, out_ in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(RoboticLinear(in_, out_))
                layers.append(nn.Tanh())
            layers.append(RoboticLinear(layer_sizes[-1], output_size))
        else:
            layers = [RoboticLinear(input_size, output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RoboticActor(RoboticMLP):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotic.py)

    **Description**

    A multi-layer perceptron with initialization designed for choosing
    actions in continuous robotic environments.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **output_size** (int) - Size of action size.
    * **layer_sizes** (list, *optional*, default=None) - A list of ints,
      each indicating the size of a hidden layer.
      (Defaults to two hidden layers of 64 units.)

    **Example**
    ~~~python
    policy_mean = ch.models.robotic.Actor(28,
                                          8,
                                          layer_sizes=[64, 32, 16])
    ~~~
    """

    def __init__(self, input_size, output_size, layer_sizes=None):
        super(RoboticMLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        if len(layer_sizes) > 0:
            layers = [RoboticLinear(input_size, layer_sizes[0]),
                      nn.Tanh()]
            for in_, out_ in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(RoboticLinear(in_, out_))
                layers.append(nn.Tanh())
            layers.append(RoboticLinear(layer_sizes[-1],
                                        output_size,
                                        gain=1.0))
        else:
            layers = [RoboticLinear(input_size, output_size, gain=1.0)]
        self.layers = nn.Sequential(*layers)
