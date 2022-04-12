#!/usr/bin/env python3

import cherry
import torch
import torch.nn as nn


class StateValueFunction(nn.Module):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/tabular.py" class="source-link">[Source]</a>

    ## Description

    Stores a table of state values, V(s), one for each state.

    Assumes that the states are one-hot encoded.
    Also, the returned values are differentiable and can be used in
    conjunction with PyTorch's optimizers.

    ## References

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    ## Example

    ~~~python
    vf = StateValueFunction(env.state_size)
    state = env.reset()
    state = ch.onehot(state, env.state_size)
    state_value = vf(state)
    ~~~

    """

    def __init__(self, state_size, init=None):
        """
        ## Arguments

        * `state_size` (int) - The number of states in the environment.
        * `init` (function, *optional*, default=None) - The initialization scheme for
            the values in the table. (Default is 0.)
        """
        super(StateValueFunction, self).__init__()
        self.values = nn.Parameter(torch.zeros((state_size, 1)))
        self.state_size = state_size
        if init is not None:
            if isinstance(init, (float, int, torch.Tensor)):
                self.values.data.add_(init)
            else:
                init(self.values)

    def forward(self, state):
        """
        ## Description

        Returns the state value of a one-hot encoded state.

        ## Arguments

        * `state` (Tensor) - State to be evaluated.
        """
        return state.view(-1, self.state_size) @ self.values


class ActionValueFunction(cherry.nn.ActionValue):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/tabular.py" class="source-link">[Source]</a>

    ## Description

    Stores a table of action values, Q(s, a), one for each
    (state, action) pair.

    Assumes that the states and actions are one-hot encoded.
    Also, the returned values are differentiable and can be used in
    conjunction with PyTorch's optimizers.

    ## References

    1. Richard Sutton and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    ## Example

    ~~~python
    qf = ActionValueFunction(env.state_size, env.action_size)
    state = env.reset()
    state = ch.onehot(state, env.state_size)
    all_action_values = qf(state)
    action = ch.onehot(0, env.action_size)
    action_value = qf(state, action)
    ~~~

    """

    def __init__(self, state_size, action_size, init=None):
        """
        ## Arguments

        * `state_size` (int) - The number of states in the environment.
        * `action_size` (int) - The number of actions per state.
        * `init` (function, *optional*, default=None) - The initialization scheme for the values in the table. (Default is 0.)
        """
        super(ActionValueFunction, self).__init__()
        self.values = nn.Parameter(torch.zeros((state_size, action_size),
                                   requires_grad=True))
        self.state_size = state_size
        self.action_size = action_size
        if init is not None:
            if isinstance(init, (float, int, torch.Tensor)):
                self.values.data.add_(init)
            else:
                init(self.values)

    def forward(self, state, action=None):
        """
        ## Description

        Returns the action value of a one-hot encoded state and one-hot encoded action.

        ## Arguments

        * `state` (Tensor) - State to be evaluated.
        * `action` (Tensor) - Action to be evaluated.
        """
        action_values = (state @ self.values).view(-1, self.action_size)
        if action is None:
            return action_values
        return torch.sum(action * action_values, dim=1, keepdim=True)
