#!/usr/bin/env python3

import torch as th
import numpy as np
import torch.nn as nn


def pong_control_(module, bias=0.1):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/init.py)

    **Description**

    The default initialization for robotic control of RLkit.

    **Credit**

    Adapted from Vitchyr Pong's implementations.

    **Arguments**

    * **module** (nn.Module) - Module to initialize.
    * **bias** (float, *optional*, default=0.1) - Constant bias initialization.

    **Returns**

    * Module, whose weight and bias have been modified in-place.

    **Example**

    ~~~python
    linear = nn.Linear(23, 5)
    pong_control_(linear)
    ~~~

    """
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
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/init.py)

    **Description**

    Default initialization for robotic control.

    **Credit**

    Adapted from Ilya Kostrikov's implementation, itself inspired from OpenAI Baslines.

    **Arguments**

    * **module** (nn.Module) - Module to initialize.
    * **gain** (float, *optional*, default=sqrt(2.0)) - Gain of orthogonal initialization.

    **Returns**

    * Module, whose weight and bias have been modified in-place.

    **Example**

    ~~~python
    linear = nn.Linear(23, 5)
    kostrikov_control_(linear)
    ~~~

    """
    with th.no_grad():
        if gain is None:
            gain = np.sqrt(2.0)
        nn.init.orthogonal_(module.weight.data, gain=gain)
        nn.init.constant_(module.bias.data, 0.0)
        return module


def atari_init_(module, gain=None):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/init.py)

    **Description**

    Default initialization for Atari environments.

    **Credit**

    Adapted from Ilya Kostrikov's implementation, itself inspired from OpenAI Baslines.

    **Arguments**

    * **module** (nn.Module) - Module to initialize.
    * **gain** (float, *optional*, default=None) - Gain of orthogonal initialization.
    Default is computed for ReLU activation with `torch.nn.init.calculate_gain('relu')`.

    **Returns**

    * Module, whose weight and bias have been modified in-place.

    **Example**

    ~~~python
    linear = nn.Linear(23, 5)
    atari_init_(linear)
    ~~~

    """
    if gain is None:
        gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0.0)
    return module
