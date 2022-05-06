# -*- coding=utf-8 -*-

import torch


class StateValue(torch.nn.Module):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/nn/state_value.py" class="source-link">[Source]</a>

    ## Description

    Abstract Module to represent V-value functions.

    ## Example

    ~~~python
    class VValue(StateValue):

        def __init__(self, state_size, action_size):
            super(QValue, self).__init__()
            self.mlp = MLP(state_size+action_size, 1, [1024, 1024])

        def forward(self, state):
            return self.mlp(state)

    state_value = VValue(128, 5)
    qvalue = state_value(state)
    ~~~
    """

    pass
