#!/usr/bin/env python3

import torch.nn as nn
import cherry as ch


class ControlLinear(nn.Linear):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/control_layers.py)

    **Description**

    Akin to `nn.Linear`, but with proper initialization for robotic control.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**


    * **gain** (float, *optional*) - Gain factor passed to `kostrikov_control_` initialization.
    * This class extends `nn.Linear` and supports all of its arguments.

    **Example**

    ~~~python
    linear = ch.nn.Linear(23, 5, bias=True)
    action_mean = linear(state)
    ~~~

    """

    def __init__(self, *args, **kwargs):
        gain = kwargs.pop('gain', None)
        super(ControlLinear, self).__init__(*args, **kwargs)
        ch.nn.init.kostrikov_control_(self, gain=gain)
