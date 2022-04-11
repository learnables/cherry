# -*- coding=utf-8 -*-

import torch


class ActionValue(torch.nn.Module):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/nn/action_value.py" class="source-link">[Source]</a>

    ## Description

    Abstract Module to represent Q-value functions.

    ## Example

    ~~~python
    class QValue(ActionValue):

        def __init__(self, state_size, action_size):
            super(QValue, self).__init__()
            self.mlp = MLP(state_size+action_size, 1, [1024, 1024])

        def forward(self, state, action):
            return self.mlp(torch.cat([state, action], dim=1))

    qf = QValue(128, 5)
    qvalue = qf(state, action)
    ~~~
    """

    def forward(self, state, action):
        raise NotImplementedError

    def all_action_values(self, state):
        raise NotImplementedError


class Twin(ActionValue):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/nn/action_value.py" class="source-link">[Source]</a>
    """

    def __init__(self, *action_values):
        super(Twin, self).__init__()
        self.action_values = torch.nn.ModuleList(action_values)

    def twin(self, state, action):
        return [qf(state, action) for qf in self.action_values]

    def forward(self, state, action):
        return torch.minimum(*self.twin(state, action))

    def all_action_values(self, state):

        return torch.minimum(*[
            qf.all_action_values(state) for qf in self.action_values
        ])
