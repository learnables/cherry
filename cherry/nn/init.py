#!/usr/bin/env python3

import torch as th
import numpy as np
import torch.nn as nn


def robotics_init_(module, gain=None):
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
    kostrikov_robotics_(linear)
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
