#!/usr/bin/env python3

import torch as th
import torch.nn as nn


class StateValueFunction(nn.Module):

    """
    **Description**

    Stores a table of state values, V(s), one for each state.

    Assumes that the states are one-hot encoded.

    **Arguments**

    **References**

    **Example**

    """

    def __init__(self, state_size, init=None):
        super(StateValueFunction, self).__init__()
        self.values = nn.Parameter(th.randn((state_size, 1)))
        self.state_size = state_size
        if init is not None:
            init(self.values)

    def forward(self, state):
        return state.view(-1, self.state_size) @ self.values


class ActionValueFunction(nn.Module):

    """
    **Description**

    Stores a table of action values, Q(s, a), one for each
    (state, action) pair.

    Assumes that the states and actions are one-hot encoded.

    **Arguments**

    **References**

    **Example**

    """

    def __init__(self, state_size, action_size, init=None):
        super(ActionValueFunction, self).__init__()
        self.values = nn.Parameter(th.randn((state_size, action_size),
                                            requires_grad=True))
        self.state_size = state_size
        self.action_size = action_size
        if init is not None:
            init(self.values)

    def forward(self, state, action=None):
        action_values = (state @ self.values).view(-1, self.action_size)
        if action is None:
            return action_values
        return th.sum(action * action_values, dim=1, keepdim=True)
