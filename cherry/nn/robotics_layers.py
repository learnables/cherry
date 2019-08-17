#!/usr/bin/env python3

import torch.nn as nn
import cherry as ch


class RoboticsLinear(nn.Linear):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/robotics_layers.py)

    **Description**

    Akin to `nn.Linear`, but with proper initialization for robotic control.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**


    * **gain** (float, *optional*) - Gain factor passed to `robotics_init_` initialization.
    * This class extends `nn.Linear` and supports all of its arguments.

    **Example**

    ~~~python
    linear = ch.nn.Linear(23, 5, bias=True)
    action_mean = linear(state)
    ~~~

    """

    def __init__(self, *args, **kwargs):
        gain = kwargs.pop('gain', None)
        super(RoboticsLinear, self).__init__(*args, **kwargs)
        ch.nn.init.robotics_init_(self, gain=gain)
