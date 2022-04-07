# -*- coding=utf-8 -*-

import torch


class ActionValue(torch.nn.Module):

    """
    Boilerplate Modulel to represent Q-value functions.
    """

    def forward(self, state, action):
        raise NotImplementedError

    def all_action_values(self, state):
        raise NotImplementedError


class Twin(ActionValue):

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
